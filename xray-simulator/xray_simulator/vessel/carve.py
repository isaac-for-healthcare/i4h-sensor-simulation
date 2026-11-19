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

"""Removing the contrast lumen from an attenuation volume so it can be composited instead.

A contrast CT carries the lumen baked in at the concentration the scan was taken with. That is
fine while the vessel is static and the contrast never changes, and wrong as soon as either
moves: a bolus that washes in and out cannot be represented by a constant, and a wall a catheter
pushes cannot be represented by voxels fixed to the original pose. So the lumen is carved out
once and composited back every frame as attenuating geometry
(:class:`~xray_simulator.vessel.VesselAttenuation`).

Three things make the carve harder than ``volume[mask] = soft_tissue``:

**There is no single value to fill with.** The aorta's neighbourhood is soft tissue *and* lung
*and* vertebral body *and* fat. Filling with one scalar leaves a vessel-shaped column of uniform
attenuation where lung should sit on one side and bone on the other. The fill is instead a
nearest-donor inpaint: every carved voxel takes the value of the closest *eligible* voxel, and
eligibility is an attenuation band measured from the shell around the vessel, so air and bone
are excluded without hard-coding Hounsfield thresholds for a particular scanner.

**What is carved and what is added back have to be the same shape.** The add-back can only use
the centerline model, and on a typical segmentation only about 83% of the segmented voxels fall
inside that tube. Carving the tube alone keeps the accounting exact -- removed and restored are
then one region. Carving the *union* removes real attenuation that nothing can put back, as a
permanent static error; it is available through ``include_segmentation`` for the case where
leaving unmodelled contrast in the volume is the worse problem.

**The boundary matters at the scale of the effect.** A binary mask on a 1.5 mm grid stair-steps
by half a voxel, against wall motion of 0.5-2 mm. ``supersample`` evaluates the tube at
sub-voxel offsets and carves fractionally.

The static residual this leaves is why the DSA view divides by a contrast-free mask frame
rather than boosting voxels: a static error appears identically in the mask and the live frame
and cancels.

Every function takes the volume as a ``(k, j, i)`` array plus the 4x4 ``ijk_to_world_mm`` that
places voxel centre ``(i, j, k)``, and the :class:`~xray_simulator.vessel.VesselTree` in that
world frame -- the same pair :class:`~xray_simulator.scan_volume.ScanVolume` exposes as
``values_kji`` and ``ijk_to_world``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .centerline import VesselTree

#: Donor band, as quantiles of the attenuation in the shell around the vessel. The shell is
#: whatever the vessel actually borders, so its interquartile range is the local soft tissue
#: without needing to name a Hounsfield window; the tails are the lung and the spine.
DONOR_QUANTILES = (0.25, 0.75)
#: Shell thickness used to measure that band, in voxels.
SHELL_VOXELS = 3
#: Margin around the carve region for the inpaint search, in voxels. Donors are looked for inside
#: this halo, so the fill stays local rather than reaching across the body.
INPAINT_MARGIN_VOXELS = 12


@dataclass(frozen=True, slots=True)
class CarveResult:
    """A carved attenuation volume and the measurements that describe what was taken out."""

    #: Carved volume, ``(k, j, i)``, in the same units and layout as the original.
    mu_volume: np.ndarray
    #: Fraction of each voxel that was carved, ``(k, j, i)`` in ``[0, 1]``. Binary unless supersampled.
    occupancy: np.ndarray
    #: Mean attenuation inside the carved region before the carve.
    lumen_mu_per_mm: float
    #: Mean attenuation of the material filled in.
    fill_mu_per_mm: float
    #: Donor band the inpaint drew from.
    donor_band_mu_per_mm: tuple[float, float]
    #: Carved voxel count and volume.
    carved_voxels: int
    carved_volume_ml: float
    #: Share of the segmentation mask the centerline tube covers, when a mask was supplied.
    tube_covers_mask: float | None

    @property
    def contrast_mu_per_mm(self) -> float:
        """Attenuation the lumen contributed over the tissue it displaced.

        The coefficient the composited vessel carries at the scan's own contrast concentration,
        derived from the volume so the dynamic geometry stays on the same transfer function as
        the anatomy behind it.
        """
        return float(self.lumen_mu_per_mm - self.fill_mu_per_mm)


def _affine(ijk_to_world_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(ijk_to_world_mm, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"ijk_to_world_mm must be 4x4, got {matrix.shape}")
    linear = matrix[:3, :3]
    if abs(np.linalg.det(linear)) <= 1.0e-12:
        raise ValueError("ijk_to_world_mm is singular")
    return linear, matrix[:3, 3]


def _voxel_centres_mm(indices_kji: np.ndarray, ijk_to_world_mm: np.ndarray) -> np.ndarray:
    linear, offset = _affine(ijk_to_world_mm)
    return indices_kji[:, ::-1].astype(np.float64) @ linear.T + offset


def tube_occupancy(
    shape_kji: tuple[int, int, int],
    ijk_to_world_mm: np.ndarray,
    tree: VesselTree,
    *,
    supersample: int = 1,
) -> np.ndarray:
    """Fraction of each voxel inside the centerline tube, ``(k, j, i)`` in ``[0, 1]``.

    The tube is the union of spheres swept along the centerline graph with each node's lumen
    radius, which is the same primitive the per-frame compositing integrates analytically, so
    what is carved and what is added back describe one geometry.

    Args:
        shape_kji: Volume shape.
        ijk_to_world_mm: Voxel-centre index to world millimetres.
        tree: Centerline in that world frame.
        supersample: Sub-voxel samples per axis. ``1`` carves whole voxels; ``2`` or more
            resolves the boundary, at a cost cubic in this value.

    Raises:
        ValueError: If ``supersample`` is below one.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415 - build-time only

    if int(supersample) < 1:
        raise ValueError(f"supersample must be at least 1, got {supersample}")
    linear, offset = _affine(ijk_to_world_mm)
    world_to_index = np.linalg.inv(linear)
    points_mm, radii_mm = tree.points_mm, tree.radii_mm

    shape_kji = tuple(int(extent) for extent in shape_kji)
    occupancy = np.zeros(shape_kji, dtype=np.float32)

    # Only the tube's bounding box can be occupied, and it is a small part of a torso scan. A
    # world sphere is an ellipsoid in index space whose half-extent along each index axis is the
    # radius times that row's norm of the inverse.
    points_ijk = (points_mm - offset) @ world_to_index.T
    reach_ijk = radii_mm[:, None] * np.linalg.norm(world_to_index, axis=1)
    lower = np.maximum(np.floor((points_ijk - reach_ijk).min(axis=0)).astype(np.int64) - 1, 0)
    upper = np.minimum(
        np.ceil((points_ijk + reach_ijk).max(axis=0)).astype(np.int64) + 2,
        np.asarray(shape_kji[::-1], dtype=np.int64),
    )
    if np.any(upper <= lower):
        return occupancy

    grid_i = np.arange(lower[0], upper[0], dtype=np.float64)
    grid_j = np.arange(lower[1], upper[1], dtype=np.float64)
    kd_tree = cKDTree(points_mm)
    steps = (np.arange(supersample, dtype=np.float64) + 0.5) / float(supersample) - 0.5
    weight = 1.0 / float(supersample**3)

    # A k-slab at a time: the whole box supersampled is several hundred million query points,
    # which is a needless allocation when the reduction is per voxel anyway.
    for k_index in range(int(lower[2]), int(upper[2])):
        slab = np.zeros((len(grid_j), len(grid_i)), dtype=np.float32)
        for dk in steps:
            for dj in steps:
                for di in steps:
                    sample_i, sample_j = np.meshgrid(grid_i + di, grid_j + dj, indexing="xy")
                    samples_ijk = np.stack(
                        (sample_i.ravel(), sample_j.ravel(), np.full(sample_i.size, float(k_index) + dk)),
                        axis=-1,
                    )
                    distance, nearest = kd_tree.query(samples_ijk @ linear.T + offset, k=1, workers=-1)
                    inside = distance <= radii_mm[nearest]
                    slab += weight * inside.reshape(sample_i.shape).astype(np.float32)
        occupancy[k_index, lower[1] : upper[1], lower[0] : upper[0]] = slab
    return occupancy


def carve_lumen(
    mu_kji: np.ndarray,
    ijk_to_world_mm: np.ndarray,
    tree: VesselTree,
    *,
    vessel_mask_kji: np.ndarray | None = None,
    supersample: int = 1,
    include_segmentation: bool = False,
    donor_quantiles: tuple[float, float] = DONOR_QUANTILES,
) -> CarveResult:
    """Carve the contrast lumen out of ``mu_kji`` and report what was taken.

    Args:
        mu_kji: Attenuation volume. Not modified.
        ijk_to_world_mm: Voxel-centre index to world millimetres.
        tree: Centerline in that world frame.
        vessel_mask_kji: Optional segmentation of the same grid, used for the coverage report
            and, with ``include_segmentation``, for the carve itself.
        supersample: Sub-voxel samples per axis for the tube boundary.
        include_segmentation: Carve the union of the tube and ``vessel_mask_kji`` rather than the
            tube alone. Off by default: the add-back can only cover the tube, so the union's
            extra voxels are removed and never restored.
        donor_quantiles: Quantiles of the shell attenuation bounding the inpaint's donors.

    Raises:
        ValueError: If the quantiles are not ordered inside ``[0, 1]``, the mask does not match
            the volume, or the carve is empty.
    """
    low, high = (float(value) for value in donor_quantiles)
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"donor_quantiles must satisfy 0 <= low < high <= 1, got {donor_quantiles}")

    mu = np.asarray(mu_kji, dtype=np.float32)
    if mu.ndim != 3:
        raise ValueError(f"mu_kji must be 3-D, got shape {mu.shape}")
    occupancy = tube_occupancy(mu.shape, ijk_to_world_mm, tree, supersample=supersample)
    tube_covers_mask = None
    if vessel_mask_kji is not None:
        segmentation = np.asarray(vessel_mask_kji).astype(bool)
        if segmentation.shape != occupancy.shape:
            raise ValueError(f"vessel_mask shape {segmentation.shape} does not match volume {occupancy.shape}")
        if segmentation.any():
            # Reported whether or not the union is carved, because it is the number that says how
            # much contrast the centerline model cannot represent at all.
            tube_covers_mask = float((occupancy > 0.0)[segmentation].mean())
        if include_segmentation:
            occupancy = np.maximum(occupancy, segmentation.astype(np.float32))

    carved_region = occupancy > 0.0
    if not carved_region.any():
        raise ValueError("the carve region is empty; check the centerline placement against the volume")

    band = _donor_band(mu, carved_region, quantiles=(low, high))
    fill = _inpaint_from_donors(mu, carved_region, band)

    # Fractional at the boundary: a voxel half inside keeps half its original attenuation.
    carved = mu * (1.0 - occupancy) + fill * occupancy

    linear, _ = _affine(ijk_to_world_mm)
    voxel_ml = float(abs(np.linalg.det(linear))) * 1.0e-3
    return CarveResult(
        mu_volume=carved,
        occupancy=occupancy,
        lumen_mu_per_mm=float(mu[carved_region].mean()),
        fill_mu_per_mm=float(fill[carved_region].mean()),
        donor_band_mu_per_mm=band,
        carved_voxels=int(carved_region.sum()),
        carved_volume_ml=float(occupancy.sum()) * voxel_ml,
        tube_covers_mask=tube_covers_mask,
    )


def node_contrast_mu_per_mm(
    mu_kji: np.ndarray,
    ijk_to_world_mm: np.ndarray,
    tree: VesselTree,
    result: CarveResult,
) -> np.ndarray:
    """Attenuation to composite back at each centerline node, measured per node.

    :attr:`CarveResult.contrast_mu_per_mm` is one number for the whole tree, and a real lumen is
    not: its attenuation typically varies by a third across the anatomy, from beam hardening,
    bolus timing during the scan and partial volume at the wall. Putting a uniform value back
    leaves a residual dominated by that spread rather than by the carve's geometry.

    Each carved voxel is attributed to its nearest centerline node, and the node's coefficient is
    what the carve took there: the mean original attenuation minus the mean fill. The rest state
    then reproduces the scan node by node rather than on average.

    Returns:
        ``(nodes,)`` attenuation per millimetre, zero at nodes no carved voxel was nearest to.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415 - build-time only

    points_mm = tree.points_mm
    carved = result.occupancy > 0.5
    indices = np.argwhere(carved)
    if not len(indices):
        return np.zeros(len(points_mm))

    _, nearest = cKDTree(points_mm).query(_voxel_centres_mm(indices, ijk_to_world_mm), k=1, workers=-1)

    original = np.asarray(mu_kji, dtype=np.float64)[carved]
    fill = np.asarray(result.mu_volume, dtype=np.float64)[carved]
    counts = np.bincount(nearest, minlength=len(points_mm))
    removed = np.bincount(nearest, weights=original - fill, minlength=len(points_mm))
    return np.divide(removed, counts, out=np.zeros(len(points_mm)), where=counts > 0)


def carve_residual(
    mu_kji: np.ndarray,
    ijk_to_world_mm: np.ndarray,
    result: CarveResult,
    *,
    axis: int = 0,
) -> dict[str, float]:
    """How much line integral the carve moved, before any deformation is composited.

    Putting the vessel back where it already was has to reproduce the original image. This is that
    check without a renderer: parallel line integrals along one array axis, comparing the original
    volume against the carved one with the tube's attenuation restored at ``contrast_mu_per_mm``.
    The residual is in the same units as the effect being added, where one millimetre of wall
    motion is about ``contrast_mu_per_mm``, so the two compare directly.

    Args:
        mu_kji: The volume the carve was built from.
        ijk_to_world_mm: Its voxel-centre index to world millimetres.
        result: What :func:`carve_lumen` produced.
        axis: Array axis to integrate along, in ``(k, j, i)`` order.

    Returns:
        ``max``, ``mean`` and ``rms`` of the absolute residual, plus the ``signal`` a millimetre of
        wall motion would produce, for scale.
    """
    linear, _ = _affine(ijk_to_world_mm)
    step_mm = float(np.linalg.norm(linear[:, 2 - axis]))
    original = np.asarray(mu_kji, dtype=np.float64).sum(axis=axis) * step_mm
    restored = (result.mu_volume + result.occupancy * result.contrast_mu_per_mm).sum(axis=axis) * step_mm
    residual = np.abs(restored - original)
    return {
        "max": float(residual.max()),
        "mean": float(residual.mean()),
        "rms": float(np.sqrt(np.mean(np.square(residual)))),
        "signal": float(result.contrast_mu_per_mm * 1.0),
    }


def _donor_band(mu: np.ndarray, region: np.ndarray, *, quantiles: tuple[float, float]) -> tuple[float, float]:
    """Attenuation range the inpaint may draw from, measured on the shell around ``region``.

    Taken from the vessel's own surroundings rather than a fixed Hounsfield window, because what
    the vessel borders is a property of the anatomy and the scanner. The interquartile range of
    that shell is the local soft tissue; the tails are lung and bone, the donors to refuse.
    """
    from scipy.ndimage import binary_dilation, generate_binary_structure  # noqa: PLC0415

    shell = binary_dilation(region, structure=generate_binary_structure(3, 1), iterations=SHELL_VOXELS) & ~region
    values = mu[shell] if shell.any() else mu[~region]
    return (float(np.quantile(values, quantiles[0])), float(np.quantile(values, quantiles[1])))


def _inpaint_from_donors(mu: np.ndarray, region: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Fill ``region`` from the nearest voxel whose attenuation sits inside ``band``.

    A Euclidean distance transform over the eligible donors gives, for every voxel at once, the
    index of its closest acceptable neighbour, so the fill is local by construction and costs one
    pass. It runs on a halo around the carve instead of the whole torso.
    """
    from scipy.ndimage import distance_transform_edt  # noqa: PLC0415

    box = _halo(region, INPAINT_MARGIN_VOXELS, mu.shape)
    local_mu = mu[box]
    local_region = region[box]
    eligible = (~local_region) & (local_mu >= band[0]) & (local_mu <= band[1])

    fill = np.array(mu, dtype=np.float32)
    if not eligible.any():
        # Nothing in the halo passes the band, so fall back to its median rather than leaving the
        # lumen in place, which would double-count against the composited tube.
        fill[region] = float(np.median(local_mu[~local_region])) if (~local_region).any() else 0.0
        return fill

    _, indices = distance_transform_edt(~eligible, return_indices=True)
    local_fill = local_mu[tuple(indices)]
    fill[box] = np.where(local_region, local_fill, local_mu)
    return fill


def _halo(region: np.ndarray, margin: int, shape: tuple[int, ...]) -> tuple[slice, ...]:
    """Bounding box of ``region`` grown by ``margin`` voxels and clipped to ``shape``."""
    bounds = []
    for axis, extent in enumerate(shape):
        along = np.any(region, axis=tuple(index for index in range(region.ndim) if index != axis))
        occupied = np.nonzero(along)[0]
        start = max(int(occupied[0]) - margin, 0)
        stop = min(int(occupied[-1]) + margin + 1, extent)
        bounds.append(slice(start, stop))
    return tuple(bounds)


__all__ = [
    "DONOR_QUANTILES",
    "CarveResult",
    "carve_lumen",
    "carve_residual",
    "node_contrast_mu_per_mm",
    "tube_occupancy",
]
