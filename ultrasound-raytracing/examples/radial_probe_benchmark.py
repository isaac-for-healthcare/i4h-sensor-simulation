# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check radial geometry and measure end-to-end performance without mesh assets.

The phantom is the union of four spheres. For a ray ``o + t d`` with unit direction
``d`` and a sphere centred at ``c`` with radius ``R``, let ``q = c - o``. The first
positive hit is

    t = dot(d, q) - sqrt(dot(d, q) ** 2 + R ** 2 - dot(q, q)).

The nearest valid hit gives the expected range for each A-line. Timings cover
``simulate_with_metadata()``, including all returned data. The benchmark writes no files.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import raysim.cuda as rs


@dataclass(frozen=True)
class RuntimeMetrics:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    fps: float
    scanlines_per_second: float
    real_time_factor: float


@dataclass(frozen=True)
class GeometryMetrics:
    detected_fraction: float
    range_bias_mm: float
    range_rmse_mm: float
    range_max_error_mm: float
    seam_residual_mm: float


@dataclass(frozen=True)
class TimestampMetrics:
    maximum_error_us: float
    maximum_spacing_jitter_us: float


@dataclass(frozen=True)
class ProbeGeometryMetrics:
    emitter_position_rmse_mm: float
    orbit_radius_rmse_mm: float
    direction_norm_max_error: float
    beam_tilt_max_error_degrees: float
    angular_pitch_max_error_degrees: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-contained radial probe geometry and performance benchmark"
    )
    parser.add_argument("--frames", type=int, default=100, help="Measured frames per phantom")
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up frames per phantom")
    parser.add_argument("--scanlines", type=int, default=256, help="A-lines per revolution")
    parser.add_argument("--buffer-size", type=int, default=4096, help="Depth samples per A-line")
    parser.add_argument("--image-size", type=int, default=512, help="Square B-mode image size")
    parser.add_argument("--t-far", type=float, default=30.0, help="Maximum range in millimetres")
    parser.add_argument(
        "--start-angle", type=float, default=0.0, help="First A-line angle in degrees"
    )
    parser.add_argument(
        "--dead-zone-radius", type=float, default=1.0, help="Central cut-out radius in millimetres"
    )
    parser.add_argument(
        "--rotation-period",
        type=float,
        default=1.0 / 30.0,
        help="Mechanical rotation period in seconds",
    )
    parser.add_argument(
        "--transducer-offset-radius",
        type=float,
        default=0.5,
        help="Emitter orbit radius around the catheter axis in millimetres",
    )
    parser.add_argument(
        "--beam-tilt",
        type=float,
        default=5.0,
        help="Signed beam tilt from the transverse plane in degrees",
    )
    parser.add_argument(
        "--rotation-direction",
        choices=("positive", "negative", "both"),
        default="both",
        help="Radial acquisition order(s) to benchmark",
    )
    parser.add_argument("--frequency", type=float, default=20.0, help="Probe frequency in MHz")
    parser.add_argument(
        "--sphere-radius", type=float, default=15.0, help="Reflector sphere radius in mm"
    )
    parser.add_argument(
        "--sphere-center-distance",
        type=float,
        default=20.0,
        help="Centred shell sphere-centre distance in mm",
    )
    parser.add_argument(
        "--offset-x", type=float, default=2.0, help="Offset shell x displacement in mm"
    )
    parser.add_argument(
        "--offset-z", type=float, default=-1.0, help="Offset shell z displacement in mm"
    )
    args = parser.parse_args()

    positive_values = {
        "frames": args.frames,
        "scanlines": args.scanlines,
        "buffer-size": args.buffer_size,
        "image-size": args.image_size,
        "t-far": args.t_far,
        "rotation-period": args.rotation_period,
        "frequency": args.frequency,
        "sphere-radius": args.sphere_radius,
        "sphere-center-distance": args.sphere_center_distance,
    }
    for name, value in positive_values.items():
        if value <= 0:
            parser.error(f"--{name} must be positive")
    if args.buffer_size != 4096:
        parser.error("--buffer-size must be 4096 (the Hilbert row length is compile-time fixed)")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.dead_zone_radius < 0:
        parser.error("--dead-zone-radius must be non-negative")
    if not math.isfinite(args.transducer_offset_radius) or args.transducer_offset_radius < 0:
        parser.error("--transducer-offset-radius must be finite and non-negative")
    if not math.isfinite(args.beam_tilt) or not -90.0 < args.beam_tilt < 90.0:
        parser.error("--beam-tilt must be finite and strictly between -90 and 90")

    centered_centers = _reflector_centers(args.sphere_center_distance, np.zeros(3))
    offset_centers = _reflector_centers(
        args.sphere_center_distance, np.array([args.offset_x, 0.0, args.offset_z])
    )
    try:
        reference_ranges = np.concatenate(
            [
                _expected_ranges(
                    centers,
                    args.sphere_radius,
                    4096,
                    args.start_angle,
                    args.transducer_offset_radius,
                    args.beam_tilt,
                    direction_sign,
                )
                for direction_sign in _selected_direction_signs(args.rotation_direction)
                for centers in (centered_centers, offset_centers)
            ]
        )
    except ValueError as exc:
        parser.error(str(exc))

    nearest_wall = float(np.min(reference_ranges))
    farthest_wall = float(np.max(reference_ranges))
    if args.dead_zone_radius >= nearest_wall:
        parser.error("the dead zone must be smaller than the nearest reflector")
    if args.t_far <= farthest_wall:
        parser.error("--t-far must extend beyond the farthest reflector")

    return args


def _selected_direction_signs(selection: str) -> tuple[int, ...]:
    if selection == "positive":
        return (1,)
    if selection == "negative":
        return (-1,)
    return (1, -1)


def _ray_geometry(
    scanlines: int,
    start_angle_degrees: float,
    transducer_offset_radius: float,
    beam_tilt_degrees: float,
    direction_sign: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = math.radians(start_angle_degrees) + direction_sign * np.arange(scanlines) * (
        2.0 * math.pi / scanlines
    )
    radial = np.column_stack((np.sin(angles), np.zeros(scanlines), np.cos(angles)))
    origins = transducer_offset_radius * radial
    tilt = math.radians(beam_tilt_degrees)
    directions = math.cos(tilt) * radial
    directions[:, 1] = math.sin(tilt)
    return angles, origins, directions


def _unpack_result(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalise tuple, mapping and object results to NumPy arrays."""
    if isinstance(result, dict):
        values = (
            result["b_mode"],
            result["rf_data"],
            result["scanlines"],
            result["scanline_timestamps"],
        )
    elif all(
        hasattr(result, name)
        for name in ("b_mode", "rf_data", "scanlines", "scanline_timestamps")
    ):
        values = (result.b_mode, result.rf_data, result.scanlines, result.scanline_timestamps)
    else:
        try:
            b_mode, raw_rf_data, scanlines, timestamps = result
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "simulate_with_metadata() must return b_mode, rf_data, scanlines, and "
                "scanline_timestamps"
            ) from exc
        values = (b_mode, raw_rf_data, scanlines, timestamps)

    b_mode_array = np.asarray(values[0])
    raw_rf_array = np.asarray(values[1])
    scanline_array = np.asarray(values[2])
    timestamp_array = np.asarray(values[3], dtype=np.float64)
    return b_mode_array, raw_rf_array, scanline_array, timestamp_array


def _reflector_centers(distance: float, offset: np.ndarray) -> np.ndarray:
    return distance * np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]]
    ) + np.asarray(offset)


def _expected_ranges(
    centers: np.ndarray,
    radius: float,
    scanlines: int,
    start_angle_degrees: float,
    transducer_offset_radius: float,
    beam_tilt_degrees: float,
    direction_sign: int,
) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"expected reflector centres with shape [N, 3], got {centers.shape}")
    _, origins, directions = _ray_geometry(
        scanlines,
        start_angle_degrees,
        transducer_offset_radius,
        beam_tilt_degrees,
        direction_sign,
    )
    relative_centers = centers[np.newaxis, :, :] - origins[:, np.newaxis, :]
    relative_norm_squared = np.sum(relative_centers**2, axis=2)
    if np.any(relative_norm_squared <= radius**2):
        raise ValueError("all reflector spheres must exclude every emitter position")

    projection = np.sum(directions[:, np.newaxis, :] * relative_centers, axis=2)
    discriminant = projection**2 + radius**2 - relative_norm_squared
    near_intersection = projection - np.sqrt(np.maximum(discriminant, 0.0))
    valid = (discriminant >= 0.0) & (near_intersection > 0.0)
    nearest = np.min(np.where(valid, near_intersection, np.inf), axis=1)
    if not np.all(np.isfinite(nearest)):
        raise ValueError("the reflector spheres do not cover the full radial field of view")
    return nearest


def _measure_geometry(
    scanlines: np.ndarray, expected_ranges: np.ndarray, t_far: float
) -> GeometryMetrics:
    if scanlines.ndim != 2:
        raise RuntimeError(f"expected scanlines with shape [lines, depth], got {scanlines.shape}")
    if scanlines.shape[0] != expected_ranges.size:
        raise RuntimeError(
            f"expected {expected_ranges.size} scanlines, got shape {scanlines.shape}"
        )
    if scanlines.shape[1] < 2:
        raise RuntimeError("at least two depth samples are required")

    values = np.abs(scanlines.astype(np.float64))
    finite_values = np.where(np.isfinite(values), values, -np.inf)
    peak_indices = np.argmax(finite_values, axis=1)
    peak_values = finite_values[np.arange(finite_values.shape[0]), peak_indices]

    line_floor = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=1)
    contrast = peak_values - line_floor
    finite_global = values[np.isfinite(values)]
    if finite_global.size == 0:
        detected = np.zeros(values.shape[0], dtype=bool)
    else:
        global_span = float(np.max(finite_global) - np.min(finite_global))
        contrast_threshold = max(global_span * 1e-6, np.finfo(np.float64).eps)
        detected = (
            np.isfinite(peak_values)
            & np.isfinite(contrast)
            & (contrast > contrast_threshold)
        )

    estimated_ranges = peak_indices.astype(np.float64) * t_far / (scanlines.shape[1] - 1)
    errors = estimated_ranges - expected_ranges
    detected_errors = errors[detected]

    if detected_errors.size == 0:
        bias = math.inf
        rmse = math.inf
        max_error = math.inf
        seam_residual = math.inf
    else:
        bias = float(np.mean(detected_errors))
        rmse = float(np.sqrt(np.mean(detected_errors**2)))
        max_error = float(np.max(np.abs(detected_errors)))
        seam_residual = (
            float(abs(errors[0] - errors[-1])) if detected[0] and detected[-1] else math.inf
        )

    return GeometryMetrics(
        detected_fraction=float(np.mean(detected)),
        range_bias_mm=bias,
        range_rmse_mm=rmse,
        range_max_error_mm=max_error,
        seam_residual_mm=seam_residual,
    )


def _validate_scan_conversion(
    b_mode: np.ndarray, dead_zone_radius: float, t_far: float
) -> None:
    if b_mode.ndim != 2:
        raise RuntimeError(f"expected a 2D B-mode image, got shape {b_mode.shape}")
    if not np.issubdtype(b_mode.dtype, np.floating):
        raise RuntimeError(f"expected floating-point B-mode data, got {b_mode.dtype}")
    if not np.all(np.isfinite(b_mode)):
        raise RuntimeError("B-mode scan conversion produced non-finite values")

    height, width = b_mode.shape
    diameter_pixels = min(width - 1, height - 1)
    if diameter_pixels <= 0:
        raise RuntimeError("B-mode dimensions must both be at least two pixels")
    pixel_spacing = 2.0 * t_far / diameter_pixels
    x = (np.arange(width) - (width - 1) * 0.5) * pixel_spacing
    z = (np.arange(height) - (height - 1) * 0.5) * pixel_spacing
    radius = np.hypot(z[:, np.newaxis], x[np.newaxis, :])
    sentinel = np.finfo(b_mode.dtype).min

    # Exclude a half-pixel band at each annulus boundary to allow for CPU/GPU round-off.
    definitely_masked = (radius < dead_zone_radius - 0.5 * pixel_spacing) | (
        radius > t_far + 0.5 * pixel_spacing
    )
    definitely_visible = (radius > dead_zone_radius + pixel_spacing) & (
        radius < t_far - pixel_spacing
    )
    if np.any(b_mode[definitely_masked] != sentinel):
        raise RuntimeError("radial scan conversion did not preserve the circular/dead-zone mask")
    if not np.any(b_mode[definitely_visible] > sentinel):
        raise RuntimeError("radial scan conversion produced no visible annulus pixels")


def _measure_timestamps(
    timestamps: np.ndarray, scanlines: int, rotation_period: float
) -> TimestampMetrics:
    timestamps = np.ravel(timestamps)
    if timestamps.size != scanlines:
        raise RuntimeError(f"expected {scanlines} timestamps, got {timestamps.size}")
    expected = np.arange(scanlines, dtype=np.float64) * rotation_period / scanlines
    maximum_error_us = float(np.max(np.abs(timestamps - expected)) * 1e6)

    wrapped = np.concatenate((timestamps, [timestamps[0] + rotation_period]))
    expected_spacing = rotation_period / scanlines
    maximum_spacing_jitter_us = float(np.max(np.abs(np.diff(wrapped) - expected_spacing)) * 1e6)
    return TimestampMetrics(maximum_error_us, maximum_spacing_jitter_us)


def _measure_probe_geometry(
    probe: Any, args: argparse.Namespace, direction_sign: int
) -> ProbeGeometryMetrics:
    _, expected_origins, _ = _ray_geometry(
        args.scanlines,
        args.start_angle,
        args.transducer_offset_radius,
        args.beam_tilt,
        direction_sign,
    )
    actual_origins = np.stack(
        [probe.get_element_position(index) for index in range(args.scanlines)]
    ).astype(np.float64)
    actual_directions = np.stack(
        [probe.get_element_direction(index) for index in range(args.scanlines)]
    ).astype(np.float64)
    actual_angles = np.array(
        [probe.get_a_line_angle(index) for index in range(args.scanlines)], dtype=np.float64
    )

    emitter_errors = actual_origins - expected_origins
    emitter_position_rmse_mm = float(np.sqrt(np.mean(np.sum(emitter_errors**2, axis=1))))
    orbit_radii = np.hypot(actual_origins[:, 0], actual_origins[:, 2])
    orbit_radius_rmse_mm = float(
        np.sqrt(np.mean((orbit_radii - args.transducer_offset_radius) ** 2))
    )
    direction_norm_max_error = float(
        np.max(np.abs(np.linalg.norm(actual_directions, axis=1) - 1.0))
    )
    actual_tilts = np.degrees(np.arcsin(np.clip(actual_directions[:, 1], -1.0, 1.0)))
    beam_tilt_max_error_degrees = float(np.max(np.abs(actual_tilts - args.beam_tilt)))

    closed_angles = np.concatenate((actual_angles, [actual_angles[0] + direction_sign * 360.0]))
    expected_pitch = direction_sign * 360.0 / args.scanlines
    angular_pitch_max_error_degrees = float(np.max(np.abs(np.diff(closed_angles) - expected_pitch)))
    return ProbeGeometryMetrics(
        emitter_position_rmse_mm=emitter_position_rmse_mm,
        orbit_radius_rmse_mm=orbit_radius_rmse_mm,
        direction_norm_max_error=direction_norm_max_error,
        beam_tilt_max_error_degrees=beam_tilt_max_error_degrees,
        angular_pitch_max_error_degrees=angular_pitch_max_error_degrees,
    )


def _measure_runtime(
    frame_times_ms: np.ndarray, scanlines: int, rotation_period: float
) -> RuntimeMetrics:
    mean_ms = float(np.mean(frame_times_ms))
    p50_ms, p95_ms, p99_ms = np.percentile(frame_times_ms, (50, 95, 99))
    fps = 1000.0 / mean_ms
    return RuntimeMetrics(
        mean_ms=mean_ms,
        p50_ms=float(p50_ms),
        p95_ms=float(p95_ms),
        p99_ms=float(p99_ms),
        fps=fps,
        scanlines_per_second=scanlines * fps,
        real_time_factor=rotation_period / (mean_ms / 1000.0),
    )


def _run_case(
    name: str,
    centers: np.ndarray,
    radius: float,
    args: argparse.Namespace,
    direction_sign: int,
) -> tuple[
    RuntimeMetrics,
    GeometryMetrics,
    TimestampMetrics,
    ProbeGeometryMetrics,
    tuple[int, ...],
    int,
    np.ndarray,
]:
    materials = rs.Materials()
    world = rs.World("water")
    for center in centers:
        world.add(rs.Sphere(center.astype(np.float32), radius, materials.get_index("bone")))

    simulator = rs.RaytracingUltrasoundSimulator(world, materials)
    pose = rs.Pose(
        position=np.zeros(3, dtype=np.float32),
        rotation=np.zeros(3, dtype=np.float32),
    )
    probe = rs.RadialProbe(
        pose,
        num_scanlines=args.scanlines,
        start_angle=args.start_angle,
        dead_zone_radius=args.dead_zone_radius,
        rotation_period=args.rotation_period,
        frequency=args.frequency,
        elevational_height=0.5,
        num_el_samples=1,
        f_num=1.0,
        speed_of_sound=1.54,
        pulse_duration=2.0,
        transducer_offset_radius=args.transducer_offset_radius,
        beam_tilt=args.beam_tilt,
        rotation_direction=(
            rs.RadialRotationDirection.POSITIVE
            if direction_sign > 0
            else rs.RadialRotationDirection.NEGATIVE
        ),
    )

    sim_params = rs.SimParams()
    sim_params.t_far = args.t_far
    sim_params.buffer_size = args.buffer_size
    # Depth 2 records the first interface response but suppresses the next.
    sim_params.max_depth = 2
    sim_params.use_scattering = False
    sim_params.conv_psf = False
    sim_params.median_clip_filter = False
    sim_params.b_mode_size = (args.image_size, args.image_size)
    sim_params.enable_cuda_timing = False

    for _ in range(args.warmup):
        simulator.simulate_with_metadata(probe, sim_params)

    frame_times_ms = np.empty(args.frames, dtype=np.float64)
    last_result: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    for frame_index in range(args.frames):
        start_ns = time.perf_counter_ns()
        result = simulator.simulate_with_metadata(probe, sim_params)
        frame_times_ms[frame_index] = (time.perf_counter_ns() - start_ns) / 1e6
        last_result = _unpack_result(result)

    if last_result is None:
        raise RuntimeError(f"{name}: benchmark produced no frames")
    b_mode, raw_rf_data, scanline_data, timestamps = last_result
    _validate_scan_conversion(b_mode, args.dead_zone_radius, args.t_far)

    expected = _expected_ranges(
        centers,
        radius,
        args.scanlines,
        args.start_angle,
        args.transducer_offset_radius,
        args.beam_tilt,
        direction_sign,
    )
    return (
        _measure_runtime(frame_times_ms, args.scanlines, args.rotation_period),
        _measure_geometry(raw_rf_data, expected, args.t_far),
        _measure_timestamps(timestamps, args.scanlines, args.rotation_period),
        _measure_probe_geometry(probe, args, direction_sign),
        b_mode.shape,
        b_mode.nbytes + raw_rf_data.nbytes + scanline_data.nbytes + timestamps.nbytes,
        b_mode,
    )


def _print_case(
    name: str,
    runtime: RuntimeMetrics,
    geometry: GeometryMetrics,
    timestamps: TimestampMetrics,
    probe_geometry: ProbeGeometryMetrics,
    b_mode_shape: tuple[int, ...],
    result_bytes: int,
) -> None:
    print(f"\n{name} reflector shell")
    print(f"  B-mode shape:              {b_mode_shape}")
    print(f"  Detected A-lines:           {geometry.detected_fraction * 100.0:8.3f} %")
    print(f"  Range bias:                {geometry.range_bias_mm:8.4f} mm")
    print(f"  Range RMSE:                {geometry.range_rmse_mm:8.4f} mm")
    print(f"  Maximum range error:       {geometry.range_max_error_mm:8.4f} mm")
    print(f"  Angular seam residual:     {geometry.seam_residual_mm:8.4f} mm")
    print(f"  Timestamp max error:       {timestamps.maximum_error_us:8.4f} us")
    print(f"  Timestamp spacing jitter:  {timestamps.maximum_spacing_jitter_us:8.4f} us")
    print(f"  Emitter position RMSE:     {probe_geometry.emitter_position_rmse_mm:8.6f} mm")
    print(f"  Orbit radius RMSE:         {probe_geometry.orbit_radius_rmse_mm:8.6f} mm")
    print(f"  Direction norm max error:  {probe_geometry.direction_norm_max_error:8.6g}")
    print(f"  Beam tilt max error:       {probe_geometry.beam_tilt_max_error_degrees:8.6f} deg")
    print(
        "  Angular pitch max error:   "
        f"{probe_geometry.angular_pitch_max_error_degrees:8.6f} deg"
    )
    print(f"  Mean frame time:           {runtime.mean_ms:8.3f} ms")
    print(
        f"  p50 / p95 / p99:           {runtime.p50_ms:.3f} / {runtime.p95_ms:.3f} / "
        f"{runtime.p99_ms:.3f} ms"
    )
    print(f"  Throughput:                {runtime.fps:8.2f} FPS")
    print(f"  A-line throughput:         {runtime.scanlines_per_second:8.0f} lines/s")
    print(f"  Real-time factor:          {runtime.real_time_factor:8.2f} x")
    print(f"  Result payload:            {result_bytes / (1024.0 * 1024.0):8.3f} MiB")


def main() -> None:
    args = _parse_args()
    centered_centers = _reflector_centers(
        args.sphere_center_distance, np.zeros(3, dtype=np.float32)
    )
    offset_centers = _reflector_centers(
        args.sphere_center_distance,
        np.array([args.offset_x, 0.0, args.offset_z], dtype=np.float32),
    )

    print("Radial probe geometry and performance benchmark")
    print(f"  Frames / warm-ups:         {args.frames} / {args.warmup} per phantom")
    print(f"  Scanlines / depth samples: {args.scanlines} / {args.buffer_size}")
    print(f"  Range / cut-out:           {args.t_far:.1f} / {args.dead_zone_radius:.1f} mm")
    print(f"  Frequency:                 {args.frequency:.1f} MHz")
    print(f"  Rotation period:           {args.rotation_period * 1000.0:.3f} ms")
    print(f"  Transducer orbit radius:   {args.transducer_offset_radius:.3f} mm")
    print(f"  Beam tilt:                 {args.beam_tilt:.3f} deg")
    print(f"  Rotation direction:        {args.rotation_direction}")

    results = []
    for direction_sign in _selected_direction_signs(args.rotation_direction):
        direction_name = "Positive" if direction_sign > 0 else "Negative"
        for shell_name, centers in (("Centred", centered_centers), ("Asymmetric", offset_centers)):
            case_name = f"{direction_name} / {shell_name}"
            result = _run_case(
                case_name, centers, args.sphere_radius, args, direction_sign
            )
            _print_case(case_name, *result[:-1])
            results.append((case_name, result))

    combined_times = np.array([result[0].mean_ms for _, result in results])
    range_error_budget = max(0.01, 2.0 * args.t_far / (args.buffer_size - 1))
    timestamp_error_budget_us = max(
        0.01,
        4.0 * np.finfo(np.float32).eps * args.rotation_period * 1e6,
    )
    position_error_budget_mm = 1e-5
    angular_error_budget_degrees = 1e-4
    direction_norm_error_budget = 1e-6
    failures = []
    for name, result in results:
        geometry = result[1]
        timestamps = result[2]
        probe_geometry = result[3]
        if geometry.detected_fraction < 1.0:
            failures.append(f"{name}: only {geometry.detected_fraction:.3%} of A-lines detected")
        if geometry.range_max_error_mm > range_error_budget:
            failures.append(
                f"{name}: {geometry.range_max_error_mm:.4f} mm maximum range error exceeds "
                f"{range_error_budget:.4f} mm"
            )
        if geometry.seam_residual_mm > range_error_budget:
            failures.append(
                f"{name}: {geometry.seam_residual_mm:.4f} mm seam residual exceeds "
                f"{range_error_budget:.4f} mm"
            )
        if timestamps.maximum_error_us > timestamp_error_budget_us:
            failures.append(
                f"{name}: {timestamps.maximum_error_us:.4f} us timestamp error exceeds "
                f"{timestamp_error_budget_us:.4f} us"
            )
        if timestamps.maximum_spacing_jitter_us > timestamp_error_budget_us:
            failures.append(
                f"{name}: {timestamps.maximum_spacing_jitter_us:.4f} us timestamp jitter "
                f"exceeds {timestamp_error_budget_us:.4f} us"
            )
        if probe_geometry.emitter_position_rmse_mm > position_error_budget_mm:
            failures.append(
                f"{name}: {probe_geometry.emitter_position_rmse_mm:.6f} mm emitter-position "
                f"RMSE exceeds {position_error_budget_mm:.6f} mm"
            )
        if probe_geometry.orbit_radius_rmse_mm > position_error_budget_mm:
            failures.append(
                f"{name}: {probe_geometry.orbit_radius_rmse_mm:.6f} mm orbit-radius RMSE "
                f"exceeds {position_error_budget_mm:.6f} mm"
            )
        if probe_geometry.direction_norm_max_error > direction_norm_error_budget:
            failures.append(
                f"{name}: {probe_geometry.direction_norm_max_error:.6g} direction-norm error "
                f"exceeds {direction_norm_error_budget:.6g}"
            )
        if probe_geometry.beam_tilt_max_error_degrees > angular_error_budget_degrees:
            failures.append(
                f"{name}: {probe_geometry.beam_tilt_max_error_degrees:.6f} degree tilt error "
                f"exceeds {angular_error_budget_degrees:.6f} degrees"
            )
        if probe_geometry.angular_pitch_max_error_degrees > angular_error_budget_degrees:
            failures.append(
                f"{name}: {probe_geometry.angular_pitch_max_error_degrees:.6f} degree pitch error "
                f"exceeds {angular_error_budget_degrees:.6f} degrees"
            )

    handedness_max_residual = None
    handedness_rmse = None
    mirrored_rmse = None
    if args.rotation_direction == "both":
        result_by_name = dict(results)
        positive_source = result_by_name["Positive / Asymmetric"][-1]
        negative_source = result_by_name["Negative / Asymmetric"][-1]
        sentinel = np.finfo(positive_source.dtype).min
        positive_b_mode = positive_source.astype(np.float64)
        negative_b_mode = negative_source.astype(np.float64)
        jointly_visible = (positive_source != sentinel) & (negative_source != sentinel)
        handedness_delta = positive_b_mode[jointly_visible] - negative_b_mode[jointly_visible]
        handedness_max_residual = float(np.max(np.abs(handedness_delta)))
        handedness_rmse = float(np.sqrt(np.mean(handedness_delta**2)))

        mirrored_source = np.fliplr(negative_source)
        mirror_visible = (positive_source != sentinel) & (mirrored_source != sentinel)
        mirrored_delta = positive_b_mode[mirror_visible] - mirrored_source.astype(np.float64)[
            mirror_visible
        ]
        mirrored_rmse = float(np.sqrt(np.mean(mirrored_delta**2)))
        handedness_error_budget = 5e-3
        if handedness_max_residual > handedness_error_budget:
            failures.append(
                f"positive/negative asymmetric B-mode residual {handedness_max_residual:.6g} "
                f"exceeds {handedness_error_budget:.6g}"
            )
        if handedness_rmse >= 0.1 * mirrored_rmse:
            failures.append(
                f"positive/negative asymmetric B-mode RMSE {handedness_rmse:.6g} is not "
                f"distinct from the mirrored-control RMSE {mirrored_rmse:.6g}"
            )

    print("\nSummary")
    print(f"  Mean latency across cases: {np.mean(combined_times):.3f} ms")
    print(f"  Range error budget:        {range_error_budget:.4f} mm")
    print(f"  Timestamp error budget:    {timestamp_error_budget_us:.4f} us")
    print(f"  Position error budget:     {position_error_budget_mm:.6f} mm")
    print(f"  Angular error budget:      {angular_error_budget_degrees:.6f} deg")
    if handedness_max_residual is not None:
        print(f"  Handedness max residual:   {handedness_max_residual:8.6g}")
        print(f"  Handedness RMSE:           {handedness_rmse:8.6g}")
        print(f"  Mirrored-control RMSE:     {mirrored_rmse:8.6g}")
    print(f"  Validation:                {'PASS' if not failures else 'FAIL'}")
    print("  Files written:             none")
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
