# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main Xray Simulator class.

This module provides the xray_simulator class - the primary interface for
generating simulated fluoroscopy (X-ray) images from preprocessed CT volumes.
"""

from __future__ import annotations

import time
import warnings
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .config import DisplaySettings, SimulatorConfig, resolve_display_settings
from .display import apply_display, calibrate_display
from .geometry import (
    clinical_angles_to_rotation,
    rotation_to_clinical_angles,
    view_frame_warning,
    view_rotation,
    volume_center_xyz_mm,
)
from .volume import PreprocessedVolume


def _as_display_settings(appearance: str | DisplaySettings) -> DisplaySettings:
    """Accept either a preset name or explicit settings."""
    if isinstance(appearance, DisplaySettings):
        return appearance
    return DisplaySettings.preset(appearance)


@dataclass
class Pose:
    """6-DOF pose for C-arm positioning.

    Rotations follow the ZXY Euler convention of the renderer, in which the beam travels
    along the C-arm's local +Z axis. See :mod:`xray_simulator.geometry` for how the world
    axes relate to patient anatomy; the classmethods here are the recommended way to reach
    a radiographic view, since the identity pose is an axial projection rather than a
    clinical one.

    Attributes:
        rotation: Euler angles (rx, ry, rz) in radians.
        translation: Translation (tx, ty, tz) in mm, displacing the isocenter from the
            center of the volume.
        view: Optional label describing an anatomically meaningful view, e.g. "AP" or
            "LAO 30 / CRAN 20". Set by the presets below and carried through to rendered
            frames. It does not affect geometry, but its presence tells the simulator the
            pose is only meaningful for a volume in the canonical patient frame.
    """

    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    view: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Pose":
        """Create from dictionary."""
        return cls(
            rotation=tuple(d.get("rotation", (0.0, 0.0, 0.0))),
            translation=tuple(d.get("translation", (0.0, 0.0, 0.0))),
            view=d.get("view"),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "rotation": list(self.rotation),
            "translation": list(self.translation),
            "view": self.view,
        }

    @classmethod
    def ap(cls, translation: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> "Pose":
        """Anteroposterior view: beam enters the front, patient left on the viewer's right."""
        return cls(rotation=view_rotation("ap"), translation=translation, view="AP")

    @classmethod
    def pa(cls, translation: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> "Pose":
        """Posteroanterior view: beam enters the back, as with a tube under the table."""
        return cls(rotation=view_rotation("pa"), translation=translation, view="PA")

    @classmethod
    def lateral(
        cls,
        side: str = "left",
        translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "Pose":
        """Lateral view, named by the side the detector is on.

        Args:
            side: "left" or "right".
            translation: Isocenter offset in mm.

        Returns:
            Pose for the requested lateral view.

        Raises:
            ValueError: If side is neither "left" nor "right".
        """
        key = side.strip().lower()
        if key not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right'; got {side!r}")
        return cls(
            rotation=view_rotation(f"lateral_{key}"),
            translation=translation,
            view=f"{key.capitalize()} lateral",
        )

    @classmethod
    def from_clinical_angles(
        cls,
        primary_deg: float = 0.0,
        secondary_deg: float = 0.0,
        translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "Pose":
        """Build a pose from cath-lab C-arm angles, relative to the PA view.

        Args:
            primary_deg: LAO/RAO angle in degrees, positive toward LAO.
            secondary_deg: CRAN/CAUD angle in degrees, positive toward cranial.
            translation: Isocenter offset in mm.

        Returns:
            Pose for the requested angulation.
        """
        primary_label = "LAO" if primary_deg >= 0 else "RAO"
        secondary_label = "CRAN" if secondary_deg >= 0 else "CAUD"
        return cls(
            rotation=clinical_angles_to_rotation(primary_deg, secondary_deg),
            translation=translation,
            view=(
                f"{primary_label} {abs(primary_deg):g} / "
                f"{secondary_label} {abs(secondary_deg):g}"
            ),
        )

    def clinical_angles(self) -> tuple[float, float]:
        """Report this pose as (LAO/RAO, CRAN/CAUD) degrees relative to the PA view."""
        return rotation_to_clinical_angles(self.rotation)


@dataclass
class Frame:
    """A rendered fluoroscopy frame.

    Attributes:
        image: 2D numpy array of pixel values (H, W), float32 in [0, 1].
        pose: The pose at which this frame was rendered.
        frame_idx: Frame index in a sequence (0 for single frames).
        timestamp_ms: Render timestamp in milliseconds.
        intensity: Pre-display intensity, kept when ``OutputSettings.keep_intensity`` is set,
            which is what allows :meth:`with_appearance` to switch modes after rendering.
        i0: Incident beam intensity the render used, needed to interpret ``intensity``.
    """

    image: np.ndarray
    pose: Pose
    frame_idx: int = 0
    timestamp_ms: float = 0.0
    intensity: np.ndarray | None = None
    i0: float = 1.0

    def with_appearance(self, appearance: str | DisplaySettings) -> "Frame":
        """Return the same render shown in another mode, e.g. fluoroscopy as an X-ray.

        Args:
            appearance: Preset name (``"fluoro"``, ``"xray"``, ...) or explicit settings.

        Returns:
            A new Frame with the remapped image; the original is unchanged.

        Raises:
            ValueError: If the intensity was not kept, or the preset name is unknown.
        """
        if self.intensity is None:
            raise ValueError(
                "this frame did not keep its intensity, so it cannot be remapped; render "
                "with SimulatorConfig.with_output(keep_intensity=True), or switch modes "
                "before rendering with simulator.set_appearance(...)"
            )

        settings = _as_display_settings(appearance)
        return replace(self, image=apply_display(self.intensity, settings, i0=self.i0))

    def save(self, path: str | Path) -> None:
        """Save frame to disk as PNG or NPY."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix.lower() == ".npy":
            np.save(path, self.image)
        else:
            # Save as PNG
            self._save_png(path)

    def _save_png(self, path: Path) -> None:
        """Save as 8-bit grayscale PNG."""
        img = np.clip(self.image, 0.0, 1.0)
        u8 = (img * 255.0 + 0.5).astype(np.uint8)
        try:
            import matplotlib.pyplot as plt

            plt.imsave(str(path), u8, cmap="gray", vmin=0, vmax=255)
        except ImportError:
            # Fallback to PGM format
            pgm_path = path.with_suffix(".pgm")
            header = f"P5\n{u8.shape[1]} {u8.shape[0]}\n255\n".encode("ascii")
            with pgm_path.open("wb") as f:
                f.write(header)
                f.write(u8.tobytes())


@dataclass
class CineSequence:
    """A sequence of rendered fluoroscopy frames.

    Attributes:
        frames: List of Frame objects.
        fps: Frames per second.
    """

    frames: list[Frame]
    fps: float = 30.0

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Frame:
        return self.frames[idx]

    def __iter__(self) -> Iterator[Frame]:
        return iter(self.frames)

    def save_all(self, output_dir: str | Path, format: str = "png") -> list[Path]:
        """Save all frames to disk.

        Args:
            output_dir: Directory to save frames.
            format: Output format ("png" or "npy").

        Returns:
            List of saved file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for frame in self.frames:
            path = output_dir / f"frame_{frame.frame_idx:04d}.{format}"
            frame.save(path)
            paths.append(path)

        return paths

    def to_numpy(self) -> np.ndarray:
        """Return all frames as a numpy array (N, H, W)."""
        return np.stack([f.image for f in self.frames], axis=0)


@dataclass
class SimulatorMetrics:
    """Performance metrics from the simulator.

    Attributes:
        fps: Average frames per second.
        frame_times_ms: List of individual frame render times.
        gpu_memory_mb: GPU memory usage in MB (if available).
        gpu_utilization: GPU utilization percentage (if available).
        jitter_ms: Frame timing jitter (std dev of frame times).
    """

    fps: float = 0.0
    frame_times_ms: list[float] = None
    gpu_memory_mb: float | None = None
    gpu_utilization: float | None = None
    jitter_ms: float = 0.0

    def __post_init__(self):
        if self.frame_times_ms is None:
            self.frame_times_ms = []

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "fps": self.fps,
            "avg_frame_time_ms": np.mean(self.frame_times_ms) if self.frame_times_ms else 0.0,
            "jitter_ms": self.jitter_ms,
            "gpu_memory_mb": self.gpu_memory_mb,
            "gpu_utilization": self.gpu_utilization,
        }


class xray_simulator:
    """Main fluoroscopy simulator class.

    This class provides the primary interface for generating simulated X-ray
    images from preprocessed CT volumes using GPU-accelerated Slang DiffDRR rendering.

    Features:
    - Single frame rendering
    - Cine sequence generation
    - Real-time streaming
    - Performance metrics collection
    - Realism post-processing (noise, blur)
    - Differentiable rendering (gradients via Slang autodiff)

    Example:
        >>> from xray_simulator import xray_simulator, SimulatorConfig, PreprocessedVolume
        >>>
        >>> # Load preprocessed volume
        >>> volume = PreprocessedVolume.load("/tmp/fluoro_cache")
        >>>
        >>> # Create simulator
        >>> config = SimulatorConfig()
        >>> simulator = xray_simulator(volume, config)
        >>>
        >>> # Render single frame
        >>> frame = simulator.render_frame(rotation=(0, 0, 0), translation=(0, 0, 0))
        >>>
        >>> # Render cine sequence
        >>> poses = [Pose(rotation=(i * 0.01, 0, 0)) for i in range(100)]
        >>> cine = simulator.render_cine(poses, fps=30)
    """

    def __init__(
        self,
        volume: PreprocessedVolume,
        config: SimulatorConfig | None = None,
    ):
        """Initialize the fluoroscopy simulator.

        Args:
            volume: Preprocessed CT volume (μ values).
            config: Simulator configuration. Uses defaults if not provided.
        """
        self._volume = volume
        self._config = config or SimulatorConfig()
        self._display = resolve_display_settings(self._config.physics, self._config.display)
        self._renderer = None
        self._metrics = SimulatorMetrics()
        self._frame_times = deque(maxlen=100)  # Rolling window for FPS
        self._frame_warning_issued = False

        # Initialize the renderer
        self._init_renderer()

    def _init_renderer(self) -> None:
        """Initialize the Slang DiffDRR rendering backend."""
        cfg = self._config

        try:
            from xray_simulator.rendering.diffdrr_slang_renderer import SlangDiffDRRConfig, SlangDiffDRRRenderer

            # The renderer hands back raw intensity; polarity and scaling are applied by
            # the display stage so that realism sees physical values.
            slang_cfg = SlangDiffDRRConfig(
                det_height_px=cfg.geometry.detector_height_px,
                det_width_px=cfg.geometry.detector_width_px,
                pixel_spacing_mm=cfg.geometry.pixel_spacing_mm,
                source_to_detector_mm=cfg.geometry.source_to_detector_mm,
                source_to_isocenter_mm=cfg.geometry.source_to_isocenter_mm,
                step_mm=cfg.physics.step_mm,
                i0=cfg.physics.i0,
                normalize=False,
                invert=False,
            )

            self._renderer = SlangDiffDRRRenderer(
                mu_volume=self._volume.mu_volume,
                spacing_zyx_mm=self._volume.spacing_zyx_mm,
                origin_xyz_mm=self._volume.metadata.origin_xyz_mm or (0.0, 0.0, 0.0),
                cfg=slang_cfg,
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Slang DiffDRR renderer: {e}\n"
                "Make sure slangpy is installed: pip install slangpy"
            ) from e

    def render_frame(
        self,
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        pose: Pose | None = None,
    ) -> Frame:
        """Render a single fluoroscopy frame.

        Args:
            rotation: Euler angles (rx, ry, rz) in radians.
            translation: Translation (tx, ty, tz) in mm.
            pose: Alternative to rotation/translation. If provided, overrides them.

        Returns:
            Rendered Frame object.
        """
        if pose is None:
            pose = Pose(rotation=rotation, translation=translation)
        else:
            rotation = pose.rotation
            translation = pose.translation

        self._check_view_frame(pose)

        start_time = time.perf_counter()

        # Render using Slang DiffDRR
        image = self._renderer.render(rotation, translation)

        # Apply realism if enabled
        if self._config.realism.enabled:
            image = self._apply_realism(image)

        intensity = image if self._config.output.keep_intensity else None
        image = self._to_display(image)

        # Record timing
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._frame_times.append(elapsed_ms)

        # Create frame
        frame = Frame(
            image=image,
            pose=replace(pose),
            frame_idx=0,
            timestamp_ms=elapsed_ms,
            intensity=intensity,
            i0=self._config.physics.i0,
        )

        # Save if configured
        if self._config.output.save_to_disk and self._config.output.output_dir:
            output_dir = Path(self._config.output.output_dir)
            frame.save(output_dir / f"frame_0000.{self._config.output.format}")

        return frame

    def render_cine(
        self,
        poses: Sequence[Pose] | Sequence[dict],
        fps: float = 30.0,
        progress: bool = True,
    ) -> CineSequence:
        """Render a cine (movie) sequence of fluoroscopy frames.

        Args:
            poses: Sequence of Pose objects or dicts with rotation/translation.
            fps: Target frames per second (for metadata).
            progress: If True, print progress.

        Returns:
            CineSequence containing all rendered frames.
        """
        frames = []
        n_frames = len(poses)

        for i, pose_data in enumerate(poses):
            if isinstance(pose_data, Pose):
                pose = pose_data
            else:
                pose = Pose.from_dict(pose_data)

            self._check_view_frame(pose)

            start_time = time.perf_counter()

            # Render frame using Slang DiffDRR
            image = self._renderer.render(pose.rotation, pose.translation)

            # Apply realism
            if self._config.realism.enabled:
                image = self._apply_realism(image, seed_offset=i)

            intensity = image if self._config.output.keep_intensity else None
            image = self._to_display(image)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._frame_times.append(elapsed_ms)

            frames.append(Frame(
                image=image,
                pose=pose,
                frame_idx=i,
                timestamp_ms=elapsed_ms,
                intensity=intensity,
                i0=self._config.physics.i0,
            ))

            if progress and (i + 1) % 10 == 0:
                current_fps = 1000.0 / np.mean(list(self._frame_times)[-10:])
                print(f"[xray_simulator] Frame {i + 1}/{n_frames} ({current_fps:.1f} FPS)")

        cine = CineSequence(frames=frames, fps=fps)

        # Save if configured
        if self._config.output.save_to_disk and self._config.output.output_dir:
            cine.save_all(self._config.output.output_dir, self._config.output.format)

        return cine

    def stream(
        self,
        pose_generator: Iterator[Pose | dict],
        max_frames: int | None = None,
    ) -> Iterator[Frame]:
        """Stream frames in real-time from a pose generator.

        This is useful for interactive applications or RL environments.

        Args:
            pose_generator: Iterator yielding Pose objects or dicts.
            max_frames: Maximum number of frames to generate (None = unlimited).

        Yields:
            Frame objects as they are rendered.

        Example:
            >>> def pose_gen():
            ...     angle = 0.0
            ...     while True:
            ...         yield Pose(rotation=(angle, 0, 0))
            ...         angle += 0.01
            >>>
            >>> for frame in simulator.stream(pose_gen(), max_frames=100):
            ...     process_frame(frame)
        """
        frame_idx = 0

        for pose_data in pose_generator:
            if max_frames is not None and frame_idx >= max_frames:
                break

            if isinstance(pose_data, Pose):
                pose = pose_data
            else:
                pose = Pose.from_dict(pose_data)

            frame = self.render_frame(pose=pose)
            frame.frame_idx = frame_idx

            yield frame
            frame_idx += 1

    def get_metrics(self) -> SimulatorMetrics:
        """Get current performance metrics.

        Returns:
            SimulatorMetrics with FPS, frame times, GPU info.
        """
        frame_times = list(self._frame_times)

        if frame_times:
            avg_time = np.mean(frame_times)
            fps = 1000.0 / avg_time if avg_time > 0 else 0.0
            jitter = np.std(frame_times)
        else:
            fps = 0.0
            jitter = 0.0

        # Try to get GPU metrics
        gpu_memory = None
        gpu_util = None
        try:
            import torch

            if torch.cuda.is_available():
                gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except ImportError:
            pass

        return SimulatorMetrics(
            fps=fps,
            frame_times_ms=frame_times,
            gpu_memory_mb=gpu_memory,
            gpu_utilization=gpu_util,
            jitter_ms=jitter,
        )

    def _check_view_frame(self, pose: Pose) -> None:
        """Warn at most once if an anatomically labeled pose meets an unlabeled volume."""
        if self._frame_warning_issued:
            return

        message = view_frame_warning(pose.view, self._volume.metadata.anatomical_frame)
        if message is None:
            return

        self._frame_warning_issued = True
        warnings.warn(message, UserWarning, stacklevel=3)

    def isocenter_translation(
        self,
        patient_point_xyz_mm: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Return the pose translation that aims the isocenter at a patient-space point.

        Args:
            patient_point_xyz_mm: Target position in world mm, in the same frame as the
                volume origin (e.g. a centerline point from preprocessing).

        Returns:
            Translation to pass to a Pose, in mm.
        """
        offset = np.asarray(patient_point_xyz_mm, dtype=np.float64) - self.volume_center_xyz_mm
        return tuple(float(v) for v in offset)

    @property
    def volume_center_xyz_mm(self) -> np.ndarray:
        """Return the world position the C-arm rotates about at zero translation."""
        return volume_center_xyz_mm(
            self._volume.shape,
            self._volume.spacing_zyx_mm,
            self._volume.metadata.origin_xyz_mm or (0.0, 0.0, 0.0),
        )

    def _apply_realism(
        self,
        image: np.ndarray,
        seed_offset: int = 0,
    ) -> np.ndarray:
        """Apply realism post-processing to a frame, in intensity units."""
        from xray_simulator.rendering.realism import RealismConfig, apply_realism

        cfg = self._config.realism
        base_seed = cfg.seed

        realism_cfg = RealismConfig(
            gain=cfg.gain,
            bias=cfg.bias,
            poisson_photons=cfg.poisson_photons,
            gaussian_sigma=cfg.gaussian_sigma,
            blur_sigma_px=cfg.blur_sigma_px,
            normalize_output=False,
            seed=None if base_seed is None else base_seed + seed_offset,
        )

        return apply_realism(image, realism_cfg)

    def _to_display(self, intensity: np.ndarray) -> np.ndarray:
        """Map rendered intensity to display values using the resolved settings."""
        return apply_display(intensity, self._display, i0=self._config.physics.i0)

    def set_appearance(self, appearance: str | DisplaySettings) -> DisplaySettings:
        """Replace the display settings at runtime, e.g. to switch preset.

        Both modes are the same render mapped differently, so this costs nothing and does not
        touch the GPU. Note that a preset brings its own scaling and window, discarding a
        window fitted by :meth:`calibrate_display`; use :meth:`set_polarity` to flip between
        fluoroscopy and X-ray while keeping calibration.

        Args:
            appearance: Preset name (``"fluoro"``, ``"xray"``, ``"fluoro_contrast"``,
                ``"transmission"``, ``"legacy"``) or explicit settings.

        Returns:
            The display settings now in effect.

        Raises:
            ValueError: If the preset name is unknown.
        """
        self._display = _as_display_settings(appearance)
        return self._display

    def set_polarity(self, polarity: str) -> DisplaySettings:
        """Flip between fluoroscopy and X-ray while keeping scaling and calibration.

        Fluoroscopy and X-ray differ only in polarity, so this is the cheap way to move
        between the two modes: everything else about the mapping, including a window fitted
        by :meth:`calibrate_display`, is preserved.

        Args:
            polarity: ``"fluoro"`` for dense-dark, ``"diagnostic"`` for dense-bright X-ray
                appearance. ``"xray"`` is accepted as a synonym for the latter.

        Returns:
            The display settings now in effect.

        Raises:
            ValueError: If the polarity is not recognised.
        """
        resolved = "diagnostic" if polarity == "xray" else polarity
        self._display = replace(self._display, polarity=resolved)
        return self._display

    def calibrate_display(
        self,
        pose: Pose | None = None,
        percentiles: tuple[float, float] = (1.0, 99.0),
    ) -> DisplaySettings:
        """Fit the log display window to this volume, then keep it fixed for later frames.

        The useful attenuation range depends on patient size and μ scaling, so a fixed
        default window is either flat or clipped for any given study. This renders one frame,
        measures the range, and stores the result, which every subsequent frame then reuses —
        stable brightness without per-frame normalization.

        Args:
            pose: Pose to measure from. Defaults to an AP view, which sees the most tissue.
            percentiles: Line-integral percentiles mapped to the bright and dark ends.

        Returns:
            The display settings now in effect.

        Raises:
            ValueError: If the measured frame has no attenuation range to fit.
        """
        pose = pose or Pose.ap()
        intensity = self._renderer.render(pose.rotation, pose.translation)
        self._display = calibrate_display(
            intensity,
            self._display,
            i0=self._config.physics.i0,
            percentiles=percentiles,
        )
        return self._display

    @property
    def config(self) -> SimulatorConfig:
        """Return current configuration."""
        return self._config

    @property
    def display(self) -> DisplaySettings:
        """Return the display settings in effect, after deprecated flags are folded in."""
        return self._display

    @property
    def volume(self) -> PreprocessedVolume:
        """Return the preprocessed volume."""
        return self._volume

    @property
    def renderer(self):
        """Return the underlying SlangDiffDRRRenderer."""
        return self._renderer

    def __repr__(self) -> str:
        return (
            f"xray_simulator(\n"
            f"  volume_shape={self._volume.shape},\n"
            f"  detector=({self._config.geometry.detector_height_px}×"
            f"{self._config.geometry.detector_width_px})\n"
            f")"
        )
