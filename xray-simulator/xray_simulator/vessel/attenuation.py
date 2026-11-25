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

"""The deforming, contrast-filled vessel's contribution to a projection's line integral.

The other half of the carve: :mod:`~xray_simulator.vessel.carve` takes the contrast lumen out of
the static volume, and this puts it back every frame at wherever a deformation solver has moved
the centerline to and whatever concentration :mod:`~xray_simulator.vessel.contrast` says is
there. Because both describe the same tube, the result at rest reproduces the original scan, and
the difference from rest is exactly the wall motion and the bolus.

Beer-Lambert makes this composable without re-rendering the volume. Transmission is
:math:`\\exp(-\\int \\mu \\, ds)`, so attenuators add in the exponent: the vessel's chord
integral is accumulated separately and multiplied into the volume's transmission, like any other
:class:`~xray_simulator.absorbers.Absorber`. Nothing about the volume is touched per frame.

**Why chords and not samples.** The lumen is 6-30 mm across and the wall moves 0.5-2 mm. Ray
marching at a step fine enough to resolve that costs hundreds of samples per pixel and still
quantises the boundary at the step size, which aliases into exactly the signal being measured.
A frustum is convex, so a ray crosses it once and the entry and exit parameters come from one
quadratic -- the geometry is resolved exactly, at a cost independent of the vessel's size.

**Why frusta and not cylinders.** The segment primitive has to match the one the carve removed,
or the two do not cancel at rest and the residual appears as a static vessel-shaped artifact.
The carve is a sphere sweep, whose radius interpolates linearly between nodes; the matching
analytic primitive is therefore the truncated cone, not the cylinder. On a segmented aorta the
radius changes by a median 0.34 mm and up to 2.14 mm across a single 1.8 mm segment, so holding
it constant would misplace the wall by up to 1.07 mm -- the size of the effect being rendered.
The cost is that a taper this steep (a tangent up to 0.82, so a 39 degree half-angle) puts rays
inside the cone's own half-angle, where the quadratic's leading coefficient goes negative and
containment is the region *outside* the roots rather than between them. That branch is real for
near-axial rays and is handled below rather than clamped away.

Requires ``warp-lang`` (the ``warp`` extra).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from ..absorbers._warp import read_out, torch_stream, warp_rays
from ..absorbers.rays import DetectorRays
from .state import VesselState

#: Segments shorter than this carry no geometry worth a chord, and a degenerate axis makes the
#: frustum's axial frame ill-conditioned. Matches the catheter kernel's threshold.
_MIN_SEGMENT_MM = 1.0e-4


@wp.func
def _clip_halfspace(low: float, high: float, constant: float, rate: float) -> wp.vec2:
    """Narrow ``[low, high]`` to the ray parameters satisfying ``constant + rate * t >= 0``.

    An empty result comes back as a reversed interval, which stays empty under any further
    clipping, so the caller tests once at the end rather than after every plane.
    """
    if rate > 1.0e-12:
        return wp.vec2(wp.max(low, -constant / rate), high)
    if rate < -1.0e-12:
        return wp.vec2(low, wp.min(high, -constant / rate))
    if constant < 0.0:
        return wp.vec2(1.0, -1.0)
    return wp.vec2(low, high)


@wp.func
def _frustum_chord(
    origin: wp.vec3,
    direction: wp.vec3,
    start: wp.vec3,
    end: wp.vec3,
    radius_start: float,
    radius_end: float,
    normal_start: wp.vec3,
    normal_end: wp.vec3,
    span: float,
) -> wp.vec2:
    """Chord length through one truncated cone, and the axial coordinate of its midpoint.

    The midpoint comes back because attenuation varies linearly along the segment, so the mean
    of a linear function over the chord is its value at the chord's centre -- exact, and one
    evaluation instead of a quadrature.

    Returns:
        ``(chord_length, axial_fraction)``, where the fraction is the chord midpoint's position
        along the segment in ``[0, 1]``. A length of zero means the ray misses.
    """
    axis_vector = end - start
    length = wp.length(axis_vector)
    if length <= _MIN_SEGMENT_MM:
        return wp.vec2(0.0, 0.0)

    # Solve relative to the ray's closest approach to the segment rather than the source. The
    # source sits hundreds of millimetres away, and in float32 a squared offset that large
    # swallows a radius of a fraction of a millimetre, so thin lumens vanished. Re-origining
    # makes every term O(segment size); the ray parameter shifts by `shift` and chords, being
    # differences of it, are unchanged.
    shift = wp.dot(0.5 * (start + end) - origin, direction)
    local_origin = origin + direction * shift

    axis = axis_vector / length
    offset = local_origin - start
    axial_offset = wp.dot(offset, axis)
    axial_direction = wp.dot(direction, axis)
    radial_offset = offset - axis * axial_offset
    radial_direction = direction - axis * axial_direction

    taper = (radius_end - radius_start) / length
    radius_at_origin = radius_start + taper * axial_offset
    radius_rate = taper * axial_direction

    # Clip to the joint planes first. Each is shared with the neighbouring segment, so the sign
    # has to be chosen from this segment's own axis to keep its own body: the neighbour makes the
    # opposite choice at the same node and the two half-spaces partition the joint exactly.
    plane_start = normal_start
    if wp.dot(normal_start, axis) < 0.0:
        plane_start = -normal_start
    plane_end = normal_end
    if wp.dot(normal_end, axis) < 0.0:
        plane_end = -normal_end
    bounds = _clip_halfspace(-shift, span - shift, wp.dot(offset, plane_start), wp.dot(direction, plane_start))
    bounds = _clip_halfspace(bounds[0], bounds[1], wp.dot(end - local_origin, plane_end), -wp.dot(direction, plane_end))
    # A mitred plane leans past its node, far enough in a tight turn that the linear radius can
    # cross zero into the mirror cone on the far side of the apex. Clipping to where the radius
    # is still positive removes that without assuming the plane stays near the end of the axis.
    bounds = _clip_halfspace(bounds[0], bounds[1], radius_at_origin, radius_rate)
    low = bounds[0]
    high = bounds[1]
    if high <= low:
        return wp.vec2(0.0, 0.0)

    # Containment is |radial(t)| <= R(t) with R linear in t, which squares to a quadratic.
    quadratic = wp.dot(radial_direction, radial_direction) - radius_rate * radius_rate
    linear = 2.0 * (wp.dot(radial_direction, radial_offset) - radius_at_origin * radius_rate)
    constant = wp.dot(radial_offset, radial_offset) - radius_at_origin * radius_at_origin

    entry = float(0.0)
    exit_ = float(-1.0)
    if wp.abs(quadratic) > 1.0e-12:
        discriminant = linear * linear - 4.0 * quadratic * constant
        if discriminant >= 0.0:
            root = wp.sqrt(discriminant)
            first = (-linear - root) / (2.0 * quadratic)
            second = (-linear + root) / (2.0 * quadratic)
            lower = wp.min(first, second)
            upper = wp.max(first, second)
            if quadratic > 0.0:
                entry = wp.max(low, lower)
                exit_ = wp.min(high, upper)
            else:
                # The ray lies inside the cone's own half-angle, so containment holds outside the
                # roots. A truncated cone is convex, so after the clipping above at most one
                # of the two tails survives; taking whichever is non-empty keeps that exact without
                # assuming which one it is.
                entry = low
                exit_ = wp.min(high, lower)
                if exit_ <= entry:
                    entry = wp.max(low, upper)
                    exit_ = high
        elif quadratic < 0.0:
            # No crossing and containment holds outside the roots, so the whole clipped span is
            # inside: an axial ray threading the cone from end plane to end plane.
            entry = low
            exit_ = high
    elif wp.abs(linear) > 1.0e-12:
        crossing = -constant / linear
        if linear > 0.0:
            entry = low
            exit_ = wp.min(high, crossing)
        else:
            entry = wp.max(low, crossing)
            exit_ = high
    elif constant <= 0.0:
        entry = low
        exit_ = high

    if exit_ <= entry:
        return wp.vec2(0.0, 0.0)
    midpoint_axial = axial_offset + 0.5 * (entry + exit_) * axial_direction
    return wp.vec2(exit_ - entry, wp.clamp(midpoint_axial / length, 0.0, 1.0))


@wp.kernel
def _accumulate_vessel_line_integral(
    source: wp.vec3,
    first_pixel: wp.vec3,
    column_step: wp.vec3,
    row_step: wp.vec3,
    starts: wp.array2d(dtype=wp.vec3),
    ends: wp.array2d(dtype=wp.vec3),
    radius_start: wp.array2d(dtype=float),
    radius_end: wp.array2d(dtype=float),
    normal_start: wp.array2d(dtype=wp.vec3),
    normal_end: wp.array2d(dtype=wp.vec3),
    mu_start: wp.array2d(dtype=float),
    mu_end: wp.array2d(dtype=float),
    line_integral: wp.array3d(dtype=float),
) -> None:
    env, row, column = wp.tid()
    beam = first_pixel + column_step * float(column) + row_step * float(row) - source
    span = wp.length(beam)
    total = float(0.0)
    if span > 1.0e-6:
        direction = beam / span
        for index in range(starts.shape[1]):
            result = _frustum_chord(
                source,
                direction,
                starts[env, index],
                ends[env, index],
                radius_start[env, index],
                radius_end[env, index],
                normal_start[env, index],
                normal_end[env, index],
                span,
            )
            if result[0] > 0.0:
                # Linear in the axial coordinate, so the chord's mean attenuation is its value at
                # the chord midpoint. Contrast concentration varies node to node, and this is what
                # makes a bolus front render as a gradient rather than a staircase.
                mu = mu_start[env, index] + (mu_end[env, index] - mu_start[env, index]) * result[1]
                total += mu * result[0]
    line_integral[env, row, column] = total


class VesselAttenuation:
    """The contrast-filled lumen's contribution to a projection's line integral.

    Shaped like :class:`~xray_simulator.absorbers.CatheterAttenuation`: both are analytic chord
    integrators on the same :class:`~xray_simulator.absorbers.DetectorRays`, so the ray
    construction -- where a registration error against the anatomy would come from -- is shared.
    """

    def __init__(self, *, device: str | None = None) -> None:
        self._device = wp.get_device(device)
        self._line_integral: wp.array | None = None
        self._buffers: dict[str, wp.array] = {}

    def line_integral(self, state: VesselState, rays: DetectorRays, *, as_torch: bool = False) -> Any:
        """Integrate the vessel's attenuation along every detector ray.

        Every environment in ``state`` is integrated along the same ``rays``; for one pose per
        environment, call once per environment.

        Args:
            state: This frame's node positions, radii and contrast attenuation.
            rays: The projection, in the frame ``state`` is expressed in.
            as_torch: Return a torch tensor on this instance's device rather than numpy.

        Returns:
            ``(envs, height, width)`` of :math:`\\int \\mu \\, ds`, ready to be added to any other
            attenuator's integral before exponentiating.
        """
        envs = state.points_mm.shape[0]
        height, width = rays.height, rays.width
        buffer = self._output_buffer(envs=envs, height=height, width=width)
        buffer.zero_()

        segments = _segment_arrays(state)
        starts, ends, radius_start, radius_end, normal_start, normal_end, mu_start, mu_end = segments
        if starts.shape[1] == 0:
            return read_out(buffer, as_torch=as_torch)

        with wp.ScopedStream(torch_stream(self._device, as_torch)):
            wp.launch(
                _accumulate_vessel_line_integral,
                dim=(envs, height, width),
                inputs=[
                    *warp_rays(rays),
                    self._upload("starts", starts, wp.vec3),
                    self._upload("ends", ends, wp.vec3),
                    self._upload("radius_start", radius_start, float),
                    self._upload("radius_end", radius_end, float),
                    self._upload("normal_start", normal_start, wp.vec3),
                    self._upload("normal_end", normal_end, wp.vec3),
                    self._upload("mu_start", mu_start, float),
                    self._upload("mu_end", mu_end, float),
                    buffer,
                ],
                device=self._device,
            )
        return read_out(buffer, as_torch=as_torch)

    def posed(self, state: VesselState) -> PosedVessel:
        """This frame's vessel as an :class:`~xray_simulator.absorbers.Absorber`."""
        return PosedVessel(self, state)

    def _upload(self, name: str, values: np.ndarray, dtype: object) -> wp.array:
        """Reuse one device array per input, so a cine run does not reallocate every frame.

        The catheter integrator allocates fresh arrays per call, which is affordable for a
        hundred segments and is not for a vessel tree evaluated at every control step.
        """
        existing = self._buffers.get(name)
        if existing is None or existing.shape != values.shape[: existing.ndim] or existing.dtype is not dtype:
            self._buffers[name] = wp.array(values, dtype=dtype, device=self._device)
            return self._buffers[name]
        existing.assign(values)
        return existing

    def _output_buffer(self, *, envs: int, height: int, width: int) -> wp.array:
        if self._line_integral is None or self._line_integral.shape != (envs, height, width):
            self._line_integral = wp.zeros((envs, height, width), dtype=float, device=self._device)
        return self._line_integral


@dataclass(frozen=True, slots=True)
class PosedVessel:
    """A vessel at one frame's geometry and contrast, ready to composite."""

    attenuation: VesselAttenuation
    state: VesselState

    def line_integral(self, rays: DetectorRays, *, as_torch: bool = False) -> Any:
        return self.attenuation.line_integral(self.state, rays, as_torch=as_torch)


def _joint_normals(points: np.ndarray, edges: np.ndarray, unit: np.ndarray) -> np.ndarray:
    """One plane direction per node, shared by every segment meeting there.

    Clipping each frustum to a plane perpendicular to its own axis leaves a wedge uncovered on
    the outside of every turn and a double-counted overlap on the inside, because the two planes
    at a shared node are not the same plane. An unsmoothed centerline follows a voxel staircase,
    so on a segmented aorta it turns a mean 32 degrees per node, which on a 7.5 mm radius displaces the joint
    by 3.7 mm either way -- twice the 1.8 mm segment length. Summing the per-segment chords then
    alternately counts the lumen twice and misses it, and the vessel renders as a barcode.
    Mitring the joint the way adjoining pipes are gives both segments one plane, so the chain
    partitions space and the chords really are additive.

    The direction is deliberately unsigned: each segment orients it from its own axis in the
    kernel. Centerline edges are not consistently oriented along flow -- extracted trees have nodes
    that two edges both point into -- so a normal whose sign came from the stored edge direction
    would cancel at exactly those nodes instead of bisecting.

    A node where more than two segments meet has no plane separating all of them. The average is
    still shared, so a parent does not overlap its children, but the children overlap each other
    within a radius of the bifurcation.
    """
    # Any one incident direction, as the reference each node's contributions are aligned to.
    # Which one is arbitrary and the bisector does not depend on it, only its overall sign does.
    reference = np.zeros_like(points)
    for side in (edges[:, 0], edges[:, 1]):
        reference[:, side] = unit

    normals = np.zeros_like(points)
    for side in (edges[:, 0], edges[:, 1]):
        # Exactly perpendicular counts as aligned: np.sign would give 0 and drop this edge's
        # contribution, leaving the joint plane following the other edge alone.
        facing = np.sum(unit * reference[:, side], axis=-1, keepdims=True)
        aligned = unit * np.where(facing < 0.0, -1.0, 1.0)
        np.add.at(normals, (slice(None), side), aligned)

    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    degenerate = length <= 1.0e-6
    return np.where(degenerate, reference, normals / np.where(degenerate, 1.0, length))


def _segment_arrays(state: VesselState) -> tuple[np.ndarray, ...]:
    """Per-segment endpoints, radii and attenuation, with degenerate segments dropped.

    A segment is dropped only when it is degenerate in *every* environment, so every environment
    keeps the same segment count and the launch stays one rectangular grid. Nodes collapse when
    the solver pinches a vessel, and that happens per environment; a segment that is open in any
    environment stays, and the kernel's own length check gives the collapsed copies a zero chord.
    """
    points = state.points_mm
    edges = state.edges
    if edges.size == 0:
        vectors = np.zeros((points.shape[0], 0, 3), dtype=np.float32)
        empty = np.zeros((points.shape[0], 0), dtype=np.float32)
        return (vectors,) * 2 + (empty,) * 2 + (vectors,) * 2 + (empty,) * 2

    left, right = edges[:, 0], edges[:, 1]
    starts = points[:, left]
    ends = points[:, right]
    axis = ends - starts
    length = np.linalg.norm(axis, axis=-1, keepdims=True)
    unit = np.divide(axis, length, out=np.zeros_like(axis), where=length > _MIN_SEGMENT_MM)
    normals = _joint_normals(points, edges, unit)

    keep = (length[:, :, 0] > _MIN_SEGMENT_MM).any(axis=0)
    return (
        np.ascontiguousarray(starts[:, keep]),
        np.ascontiguousarray(ends[:, keep]),
        np.ascontiguousarray(state.radii_mm[:, left][:, keep]),
        np.ascontiguousarray(state.radii_mm[:, right][:, keep]),
        np.ascontiguousarray(normals[:, left][:, keep]),
        np.ascontiguousarray(normals[:, right][:, keep]),
        np.ascontiguousarray(state.mu_per_mm[:, left][:, keep]),
        np.ascontiguousarray(state.mu_per_mm[:, right][:, keep]),
    )


__all__ = ["PosedVessel", "VesselAttenuation", "VesselState"]
