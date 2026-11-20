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

"""Vessel carve, contrast transport, chord integration and the compositor that ties them."""

import numpy as np
import pytest
from xray_simulator.absorbers import DetectorRays
from xray_simulator.config import HuToMuMapping
from xray_simulator.vessel import (
    ContrastAgent,
    ContrastTransport,
    HuToMu,
    Injection,
    VesselTree,
    carve_lumen,
    carve_residual,
    node_contrast_mu_per_mm,
    smooth_centerline,
    tube_occupancy,
)

# A beam down +z with the isocenter at the origin, so a chord can be read off a column by hand.
DETECTOR_PX = 64
PIXEL_MM = 1.0
SID_MM = 500.0
SDD_MM = 1000.0
MAGNIFICATION = SDD_MM / SID_MM
CENTER_COLUMN = 0.5 * DETECTOR_PX - 0.5
CENTER_ROW = DETECTOR_PX // 2

SHAPE_KJI = (24, 24, 24)
SPACING_MM = 1.0
LUMEN_MU = 0.004
TISSUE_MU = 0.001
CONTROL_POINTS = ((-1000.0, 0.0), (-300.0, 0.0), (100.0, 0.0008), (300.0, 0.0028))


def _rays() -> DetectorRays:
    return DetectorRays.from_axes(
        np.eye(3),
        np.zeros(3),
        source_to_detector_mm=SDD_MM,
        source_to_isocenter_mm=SID_MM,
        width=DETECTOR_PX,
        height=DETECTOR_PX,
        pixel_spacing_mm=PIXEL_MM,
    )


def _column_of(world_x_mm: float) -> int:
    return int(round(MAGNIFICATION * world_x_mm + CENTER_COLUMN))


# The detector has an even pixel count, so no pixel centre sits exactly on the axis: the nearest
# row and column are half a pixel off, which at this magnification is a quarter of a millimetre
# at the isocenter. A ray that misses the axis by 0.25 mm cuts a shorter chord, and expecting the
# full diameter would read as a kernel error on a narrow vessel.
_AXIS_OFFSET_MM = 0.5 * PIXEL_MM / MAGNIFICATION


def _chord_through(radius_mm: float) -> float:
    """Chord a ray offset from the axis by half a pixel cuts through a cylinder."""
    return 2.0 * np.sqrt(radius_mm**2 - _AXIS_OFFSET_MM**2)


def _tube(radius_mm: float = 3.0) -> tuple[np.ndarray, np.ndarray, VesselTree, np.ndarray]:
    """A straight contrast-filled tube along i through a block of soft tissue.

    Synthetic so the carve's answers are known: the lumen is exactly a cylinder of one
    attenuation, so what has to come out and what has to go back are both analytic.

    Returns:
        ``(mu_kji, ijk_to_world_mm, tree, vessel_mask_kji)``.
    """
    k, j, _ = np.indices(SHAPE_KJI)
    centre = (SHAPE_KJI[0] - 1) / 2.0
    inside = ((j - centre) ** 2 + (k - centre) ** 2) <= radius_mm**2
    mu = np.full(SHAPE_KJI, TISSUE_MU, dtype=np.float32)
    mu[inside] = LUMEN_MU
    # One node per voxel. The tube is a sweep of per-node spheres, so sparse nodes scallop the
    # surface inward by about h^2 / 8r between them; at this spacing that is well under a tenth
    # of a voxel. Spanning the full extent, so coverage is a property of the radii.
    nodes = SHAPE_KJI[2]
    points = np.zeros((nodes, 3))
    points[:, 0] = np.linspace(0.0, (SHAPE_KJI[2] - 1) * SPACING_MM, nodes)
    points[:, 1:] = centre * SPACING_MM
    edges = np.stack([np.arange(nodes - 1), np.arange(1, nodes)], axis=-1)
    tree = VesselTree(points, edges, np.full(nodes, radius_mm))
    affine = np.diag((SPACING_MM, SPACING_MM, SPACING_MM, 1.0))
    return mu, affine, tree, inside.astype(np.uint8)


def _curve() -> HuToMu:
    return HuToMu.from_mapping(HuToMuMapping(control_points=CONTROL_POINTS))


# ---------------------------------------------------------------- carve


def test_tube_occupancy_matches_the_cylinder_it_describes() -> None:
    mu, affine, tree, _ = _tube(radius_mm=3.0)
    occupancy = tube_occupancy(mu.shape, affine, tree)
    # Away from the capped ends the cross-section is a disc of the node radius.
    assert occupancy[:, :, SHAPE_KJI[2] // 2].sum() == pytest.approx(np.pi * 9.0, rel=0.15)


def test_supersampling_resolves_the_boundary_fractionally() -> None:
    mu, affine, tree, _ = _tube()
    coarse = tube_occupancy(mu.shape, affine, tree, supersample=1)
    fine = tube_occupancy(mu.shape, affine, tree, supersample=3)
    assert set(np.unique(coarse)) <= {0.0, 1.0}
    assert np.any((fine > 0.0) & (fine < 1.0)), "supersampling should produce partial voxels"
    assert fine.sum() == pytest.approx(coarse.sum(), rel=0.2)


def test_occupancy_follows_the_grid_affine_not_the_array_layout() -> None:
    """The same world tube on a rotated, shifted grid carves the same world voxels.

    The second grid swaps the i and j axes and moves the origin, so a carve that indexed the
    array by world position divided by spacing would put the tube somewhere else entirely.
    """
    mu, affine, tree, _ = _tube()
    reference = tube_occupancy(mu.shape, affine, tree)

    swapped = np.array(
        [[0.0, 1.0, 0.0, -5.0], [1.0, 0.0, 0.0, 7.0], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]],
    )
    # Voxel (i, j, k) of the new grid sits where voxel (j, i, k) of the old one did, shifted, so
    # the same anatomy is the old tube shifted and its occupancy is the old one transposed.
    moved = VesselTree(tree.points_mm + [-5.0, 7.0, 3.0], tree.edges, tree.radii_mm)
    occupancy = tube_occupancy(mu.shape, swapped, moved)
    np.testing.assert_array_equal(occupancy, reference.transpose(0, 2, 1))


def test_anisotropic_spacing_scales_the_carved_volume() -> None:
    mu, _, tree, _ = _tube(radius_mm=3.0)
    fine = carve_lumen(mu, np.diag((1.0, 1.0, 1.0, 1.0)), tree)
    # Halving every voxel's k extent halves the volume each carved voxel stands for.
    flat = np.diag((1.0, 1.0, 0.5, 1.0))
    flat_tree = VesselTree(tree.points_mm * [1.0, 1.0, 0.5], tree.edges, tree.radii_mm)
    coarse = carve_lumen(mu, flat, flat_tree)
    assert coarse.carved_volume_ml / coarse.carved_voxels == pytest.approx(
        0.5 * fine.carved_volume_ml / fine.carved_voxels
    )


def test_carve_removes_the_lumen_and_leaves_everything_else() -> None:
    mu, affine, tree, _ = _tube()
    original = mu.copy()
    result = carve_lumen(mu, affine, tree)
    carved = result.occupancy > 0.0

    assert result.lumen_mu_per_mm == pytest.approx(LUMEN_MU, rel=0.05)
    assert result.fill_mu_per_mm == pytest.approx(TISSUE_MU, rel=0.1)
    assert result.contrast_mu_per_mm == pytest.approx(LUMEN_MU - TISSUE_MU, rel=0.1)
    assert np.array_equal(result.mu_volume[~carved], original[~carved])
    assert mu is not result.mu_volume
    np.testing.assert_array_equal(mu, original)


def test_carve_reports_how_much_of_the_segmentation_the_tube_covers() -> None:
    mu, affine, tree, mask = _tube()
    # Reported even when the union is not carved, because it says how much contrast the
    # centerline model cannot represent at all.
    for union in (False, True):
        result = carve_lumen(mu, affine, tree, vessel_mask_kji=mask, include_segmentation=union)
        assert result.tube_covers_mask == pytest.approx(1.0, abs=0.1)


def test_carve_without_a_mask_reports_no_coverage() -> None:
    mu, affine, tree, _ = _tube()
    assert carve_lumen(mu, affine, tree).tube_covers_mask is None


def test_carving_the_union_removes_at_least_the_tube() -> None:
    mu, affine, tree, mask = _tube()
    tube_only = carve_lumen(mu, affine, tree, vessel_mask_kji=mask)
    union = carve_lumen(mu, affine, tree, vessel_mask_kji=mask, include_segmentation=True)
    assert union.carved_voxels >= tube_only.carved_voxels


def test_restoring_the_carve_reproduces_the_volume_it_came_from() -> None:
    mu, affine, tree, _ = _tube()
    residual = carve_residual(mu, affine, carve_lumen(mu, affine, tree))
    # The synthetic lumen is uniform, so putting it back is near-exact; the bar is that the
    # residual sits well under the effect it would otherwise swamp.
    assert residual["rms"] < 0.25 * residual["signal"]
    assert residual["max"] < 2.0 * residual["signal"]


def test_node_coefficients_recover_the_lumen_the_carve_removed() -> None:
    mu, affine, tree, _ = _tube()
    per_node = node_contrast_mu_per_mm(mu, affine, tree, carve_lumen(mu, affine, tree))
    assigned = per_node[per_node > 0.0]
    assert len(assigned) > 0
    assert assigned.mean() == pytest.approx(LUMEN_MU - TISSUE_MU, rel=0.15)


def test_carve_rejects_a_mask_that_does_not_match_the_volume() -> None:
    mu, affine, tree, _ = _tube()
    with pytest.raises(ValueError, match="does not match"):
        carve_lumen(mu, affine, tree, vessel_mask_kji=np.zeros((3, 3, 3)), include_segmentation=True)


def test_carve_rejects_nonsense_donor_quantiles() -> None:
    mu, affine, tree, _ = _tube()
    with pytest.raises(ValueError, match="donor_quantiles"):
        carve_lumen(mu, affine, tree, donor_quantiles=(0.8, 0.2))


def test_carve_rejects_a_singular_affine() -> None:
    mu, _, tree, _ = _tube()
    with pytest.raises(ValueError, match="singular"):
        carve_lumen(mu, np.diag((1.0, 0.0, 1.0, 1.0)), tree)


def test_tree_requires_one_radius_per_node() -> None:
    with pytest.raises(ValueError, match="one radius per node"):
        VesselTree(np.zeros((4, 3)), np.array([[0, 1]]), np.ones(3))
    with pytest.raises(ValueError, match="edges index outside"):
        VesselTree(np.zeros((4, 3)), np.array([[0, 9]]), np.ones(4))


def test_smoothing_straightens_the_voxel_staircase_without_moving_the_vessel() -> None:
    """A node is pulled off the staircase by less than the voxel the staircase came from."""
    staircase = np.array([[float(index // 2), float(index % 2), 0.0] for index in range(12)])
    edges = np.stack([np.arange(11), np.arange(1, 12)], axis=-1)
    smoothed = smooth_centerline(staircase, edges)

    def mean_turn(points: np.ndarray) -> float:
        before = points[1:-1] - points[:-2]
        after = points[2:] - points[1:-1]
        cosine = np.sum(before * after, axis=-1) / (np.linalg.norm(before, axis=-1) * np.linalg.norm(after, axis=-1))
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).mean())

    assert mean_turn(smoothed) < 0.5 * mean_turn(staircase)
    assert np.linalg.norm(smoothed - staircase, axis=-1).max() < 1.0
    np.testing.assert_array_equal(VesselTree(staircase, edges, np.ones(12)).smoothed().points_mm, smoothed)


# ---------------------------------------------------------------- HU curve and transport


def test_curve_reads_a_preset_and_a_metadata_record_alike() -> None:
    from_metadata = HuToMu.from_metadata({"hu_to_mu": {"control_points": [list(p) for p in CONTROL_POINTS]}})
    np.testing.assert_array_equal(_curve().hu, from_metadata.hu)
    np.testing.assert_array_equal(_curve().mu, from_metadata.mu)
    assert HuToMu.from_mapping(HuToMuMapping.preset("interventional")).mu_of(0.0) > 0.0


def test_curve_rejects_a_decreasing_attenuation() -> None:
    with pytest.raises(ValueError, match="not decrease in mu"):
        HuToMu(hu=np.array([0.0, 1.0]), mu=np.array([1.0, 0.5]))


def test_transfer_function_inverts_on_its_strictly_increasing_part() -> None:
    curve = _curve()
    for hu in (100.0, 200.0, 300.0):
        assert float(curve.hu_of(curve.mu_of(hu))) == pytest.approx(hu)


def test_inverting_a_flat_run_keeps_its_upper_end() -> None:
    # Zero attenuation spans -1000..-300 HU. Collapsing the run to its first point instead would
    # stretch the next segment back to -1000 and drag soft tissue hundreds of HU negative.
    curve = _curve()
    assert float(curve.hu_of(0.0)) == pytest.approx(-300.0)
    assert float(curve.hu_of(0.0004)) == pytest.approx(-100.0, abs=1.0)


def test_concentration_is_fitted_from_the_attenuation_it_has_to_reproduce() -> None:
    curve, agent = _curve(), ContrastAgent()
    baseline = 0.001
    concentration = agent.concentration_matching(0.0005, baseline_mu=baseline, curve=curve)
    replayed = agent.mu_excess(np.array([concentration]), baseline_mu=baseline, curve=curve)[0]
    assert replayed == pytest.approx(0.0005, abs=1e-9)


def _tree() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A trunk that bifurcates: 10 nodes down, then two branches of 5."""
    trunk = np.stack([np.zeros(10), np.zeros(10), np.arange(10.0)], axis=-1)
    left = np.stack([np.arange(1.0, 6.0), np.zeros(5), np.full(5, 9.0)], axis=-1)
    right = np.stack([-np.arange(1.0, 6.0), np.zeros(5), np.full(5, 9.0)], axis=-1)
    points = np.concatenate([trunk, left, right])
    edges = [(index, index + 1) for index in range(9)]
    edges += [(9, 10)] + [(10 + index, 11 + index) for index in range(4)]
    edges += [(9, 15)] + [(15 + index, 16 + index) for index in range(4)]
    radii = np.concatenate([np.full(10, 4.0), np.full(5, 3.0), np.full(5, 2.0)])
    return points, np.asarray(edges), radii


def test_flow_splits_by_murrays_law_at_a_bifurcation() -> None:
    points, edges, radii = _tree()
    flow = ContrastTransport(points, edges, radii, inflow=0, flow_ml_per_s=10.0).flow_ml_per_s
    assert flow[0] == pytest.approx(10.0)
    assert flow[10] + flow[15] == pytest.approx(10.0)
    assert flow[10] / flow[15] == pytest.approx(3.0**3 / 2.0**3)


def test_flow_is_conserved_along_an_unbranched_run() -> None:
    points, edges, radii = _tree()
    flow = ContrastTransport(points, edges, radii, inflow=0, flow_ml_per_s=10.0).flow_ml_per_s
    np.testing.assert_allclose(flow[:10], 10.0)


def test_contrast_fills_downstream_and_then_clears() -> None:
    points, edges, radii = _tree()
    transport = ContrastTransport(points, edges, radii, inflow=0, flow_ml_per_s=10.0)
    injection = Injection(node=0, start_s=0.0, duration_s=0.2, flow_ml_per_s=10.0)
    peak = 0.0
    for _ in range(40):
        transport.step(0.05, injection)
        peak = max(peak, float(transport.concentration_mg_per_ml.max()))
    assert peak > 1.0, "the injection should raise concentration somewhere"
    assert float(transport.concentration_mg_per_ml.max()) < 0.01 * peak, "and it should wash out"


def test_the_root_is_flushed_by_contrast_free_blood() -> None:
    # Falling back to the root's own concentration makes it a perfect reservoir with nothing to
    # flush it, and the bolus never clears.
    points, edges, radii = _tree()
    transport = ContrastTransport(points, edges, radii, inflow=0, flow_ml_per_s=10.0)
    transport.step(0.1, Injection(node=0, start_s=0.0, duration_s=0.1, flow_ml_per_s=10.0))
    loaded = float(transport.concentration_mg_per_ml[0])
    transport.step(2.0, None)
    assert float(transport.concentration_mg_per_ml[0]) < 0.01 * loaded


def test_advection_carries_contrast_downstream_only() -> None:
    # With dispersion off the transport is pure advection, which is one-directional: contrast
    # delivered mid-tree fills what is below the tip and cannot reach the root or the sibling.
    points, edges, radii = _tree()
    transport = ContrastTransport(points, edges, radii, inflow=0, flow_ml_per_s=10.0, dispersion_mm2_per_s=0.0)
    transport.step(0.4, Injection(node=12, start_s=0.0, duration_s=0.4, flow_ml_per_s=10.0))
    concentration = transport.concentration_mg_per_ml
    assert concentration[13] > 1.0, "the branch below the tip should fill"
    assert concentration[:10].max() == pytest.approx(0.0), "the trunk above it must not"
    assert concentration[15:].max() == pytest.approx(0.0), "nor the sibling branch"


def test_dispersion_spreads_both_ways_but_advection_dominates() -> None:
    points, edges, radii = _tree()
    transport = ContrastTransport(points, edges, radii, inflow=0, flow_ml_per_s=10.0)
    transport.step(0.4, Injection(node=12, start_s=0.0, duration_s=0.4, flow_ml_per_s=10.0))
    concentration = transport.concentration_mg_per_ml
    assert concentration[:10].max() > 0.0
    assert concentration[:10].max() < 0.25 * concentration[13]


def test_inflow_defaults_to_the_widest_endpoint() -> None:
    points, edges, radii = _tree()
    assert ContrastTransport(points, edges, radii).inflow_node == 0


def test_transport_rejects_a_disconnected_centerline() -> None:
    points, edges, radii = _tree()
    with pytest.raises(ValueError, match="disconnected"):
        ContrastTransport(points, edges[:3], radii, inflow=0)


def test_injection_concentration_is_diluted_by_the_blood_it_enters() -> None:
    points, edges, radii = _tree()
    agent = ContrastAgent(stock_mg_per_ml=300.0)
    fast = ContrastTransport(points, edges, radii, inflow=0, flow_ml_per_s=10.0)
    fast.step(0.3, Injection(node=0, start_s=0.0, duration_s=0.3, flow_ml_per_s=10.0), agent=agent)
    slow = ContrastTransport(points, edges, radii, inflow=0, flow_ml_per_s=10.0)
    slow.step(0.3, Injection(node=0, start_s=0.0, duration_s=0.3, flow_ml_per_s=1.0), agent=agent)
    assert float(fast.concentration_mg_per_ml.max()) == pytest.approx(300.0 * 10.0 / 20.0, rel=0.05)
    assert float(slow.concentration_mg_per_ml.max()) < 0.3 * float(fast.concentration_mg_per_ml.max())


# ---------------------------------------------------------------- chord kernel


def _attenuation():
    pytest.importorskip("warp")
    from xray_simulator.vessel import VesselAttenuation

    return VesselAttenuation(device="cpu")


def _state(points, edges, radii, mus):
    from xray_simulator.vessel import VesselState

    return VesselState(np.asarray(points, dtype=np.float32), np.asarray(edges), np.asarray(radii), np.asarray(mus))


def _chord_image(radii: tuple[float, float], mus: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    attenuation = _attenuation()
    state = _state([[[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]], [[0, 1]], radii, mus)
    return attenuation.line_integral(state, _rays())


def test_a_ray_across_a_cylinder_sees_its_diameter() -> None:
    image = _chord_image((4.0, 4.0))
    assert image[0, CENTER_ROW, _column_of(0.0)] == pytest.approx(_chord_through(4.0), abs=0.02)


def test_a_ray_past_the_end_planes_sees_nothing() -> None:
    # The segment spans x in [-10, 10] and the detector reaches about 15 mm of world x.
    image = _chord_image((4.0, 4.0))
    assert image[0, CENTER_ROW, _column_of(13.0)] == pytest.approx(0.0)


def test_the_chord_follows_a_taper_along_the_segment() -> None:
    # Radius runs 2 -> 10 over x in [-10, 10], so it is 6 + 0.4x, read where the ray crosses.
    image = _chord_image((2.0, 10.0))
    for world_x in (-5.0, 0.0, 5.0):
        axial_x = (_column_of(world_x) - CENTER_COLUMN) / MAGNIFICATION
        expected = _chord_through(6.0 + 0.4 * axial_x)
        assert image[0, CENTER_ROW, _column_of(world_x)] == pytest.approx(expected, abs=0.05)


def test_a_bent_chain_is_not_counted_twice_at_its_joints() -> None:
    """Two segments meeting at an angle must sum to the chord through their union.

    Clipped to planes perpendicular to each segment's own axis, the two shapes overlap on the
    inside of the turn and leave a wedge on the outside, and summing their chords counts the
    overlap twice. On a real centerline that alternates along the vessel and renders as a
    barcode, so the chords are checked against a marched union rather than each other.
    """
    radius = 4.0
    points = np.array([[[-10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [8.0, 6.0, 0.0]]], dtype=np.float32)
    image = _attenuation().line_integral(_state(points, [[0, 1], [1, 2]], np.full(3, radius), np.ones(3)), _rays())[0]

    nodes = points[0].astype(np.float64)
    step = 0.02
    depth = np.arange(-30.0, 30.0, step)
    for world_x in (-3.0, 0.0, 3.0):
        for world_y in (-3.0, 0.0, 3.0):
            column, row = _column_of(world_x), _column_of(world_y)
            position = (np.array([column, row]) - CENTER_COLUMN) / MAGNIFICATION
            along = np.stack([np.full_like(depth, position[0]), np.full_like(depth, position[1]), depth], axis=-1)
            inside = np.zeros(len(depth), dtype=bool)
            for first, second in ((nodes[0], nodes[1]), (nodes[1], nodes[2])):
                axis = second - first
                fraction = np.clip(((along - first) @ axis) / (axis @ axis), 0.0, 1.0)
                nearest = first + fraction[:, None] * axis
                inside |= np.linalg.norm(along - nearest, axis=-1) <= radius
            assert image[row, column] == pytest.approx(inside.sum() * step, abs=0.1)


def test_a_taper_is_not_silently_treated_as_a_cylinder() -> None:
    assert np.abs(_chord_image((2.0, 10.0)) - _chord_image((6.0, 6.0))).max() > 1.0


def test_a_ray_inside_the_cones_own_half_angle_is_still_contained() -> None:
    """The branch where the quadratic's leading coefficient goes negative.

    A steeply tapered segment pointing along the beam puts the ray inside the cone's half-angle,
    where containment is the region outside the roots rather than between them.
    """
    attenuation = _attenuation()
    for radii in ((1.0, 9.0), (9.0, 1.0)):
        state = _state([[[0.0, 0.0, -10.0], [0.0, 0.0, 10.0]]], [[0, 1]], radii, [1.0, 1.0])
        image = attenuation.line_integral(state, _rays())
        # The central ray runs the segment's whole length, end plane to end plane.
        assert image[0, CENTER_ROW, _column_of(0.0)] == pytest.approx(20.0, abs=0.1)
        assert np.isfinite(image).all()
        assert image.min() >= 0.0


def test_attenuation_varying_along_a_segment_is_integrated_at_the_chord_midpoint() -> None:
    # mu runs 0 -> 2 over x in [-10, 10], so the chord at world x carries mu = 1 + x / 10.
    image = _chord_image((4.0, 4.0), mus=(0.0, 2.0))
    chord = _chord_through(4.0)
    for world_x in (-5.0, 0.0, 5.0):
        axial_x = (_column_of(world_x) - CENTER_COLUMN) / MAGNIFICATION
        assert image[0, CENTER_ROW, _column_of(world_x)] == pytest.approx(chord * (1.0 + axial_x / 10.0), abs=0.05)


def test_environments_do_not_bleed_into_each_other() -> None:
    points = np.repeat(np.array([[[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]]), 2, axis=0)
    state = _state(points, [[0, 1]], [[4.0, 4.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]])
    image = _attenuation().line_integral(state, _rays())
    assert image[0, CENTER_ROW, _column_of(0.0)] == pytest.approx(_chord_through(4.0), abs=0.02)
    assert image[1, CENTER_ROW, _column_of(0.0)] == pytest.approx(_chord_through(1.0), abs=0.02)


def test_vessel_state_rejects_mismatched_shapes() -> None:
    pytest.importorskip("warp")
    points = np.zeros((1, 4, 3))
    with pytest.raises(ValueError, match="radii_mm"):
        _state(points, [[0, 1]], np.ones(3), np.ones(4))
    with pytest.raises(ValueError, match="edges index outside"):
        _state(points, [[0, 9]], np.ones(4), np.ones(4))


# ---------------------------------------------------------------- compositor


def _compositor(**kwargs):
    pytest.importorskip("warp")
    from xray_simulator.vessel import VesselContrast

    mu, affine, tree, mask = _tube()
    return VesselContrast(mu, affine, tree, _curve(), vessel_mask_kji=mask, device="cpu", **kwargs), mu


def test_the_mask_volume_is_the_scan_without_its_lumen() -> None:
    vessel, mu = _compositor()
    assert vessel.carve.carved_voxels > 0
    assert vessel.mask_volume.mean() < mu.mean()
    # Coefficients are per node, so on a lumen that really is uniform every node lands on the
    # same excess.
    np.testing.assert_allclose(vessel.rest_state.mu_per_mm, LUMEN_MU - TISSUE_MU, rtol=0.15)


def test_compositing_the_rest_state_restores_the_scan_along_a_ray() -> None:
    """The carve's contract, checked through the compositor's own integrator.

    The tube runs along world x through the block. Beam down +z puts it across the detector, so a
    ray through its axis crosses the lumen's diameter; what the carve took out along that ray has
    to come back from the composited vessel.
    """
    vessel, mu = _compositor()
    centre = (SHAPE_KJI[0] - 1) / 2.0
    rays = DetectorRays.from_axes(
        np.eye(3),
        np.array([centre, centre, centre]),
        source_to_detector_mm=SDD_MM,
        source_to_isocenter_mm=SID_MM,
        width=DETECTOR_PX,
        height=DETECTOR_PX,
        pixel_spacing_mm=PIXEL_MM,
    )
    composited = vessel.absorber().line_integral(rays)[0, CENTER_ROW, CENTER_ROW]
    removed = float((mu - vessel.mask_volume)[:, int(centre), int(centre)].sum())
    assert composited == pytest.approx(removed, rel=0.15)


def test_displacement_moves_the_vessel_and_concentration_sets_its_contrast() -> None:
    vessel, _ = _compositor()
    rest = vessel.rest_state
    shift = np.zeros((2, *rest.points_mm.shape[1:]))
    shift[1, :, 1] = 1.5
    moved = vessel.state(shift, concentration_mg_per_ml=np.full(rest.points_mm.shape[1], 60.0))
    assert moved.points_mm.shape[0] == 2
    np.testing.assert_allclose(moved.points_mm[0], rest.points_mm[0])
    np.testing.assert_allclose(moved.points_mm[1, :, 1], rest.points_mm[0, :, 1] + 1.5)
    np.testing.assert_allclose(moved.mu_per_mm, vessel.mu_excess(np.array([60.0]))[0], rtol=1e-6)
    np.testing.assert_array_equal(moved.radii_mm[1], rest.radii_mm[0])


def test_the_subtracted_window_is_set_by_the_reference_injection() -> None:
    vessel, _ = _compositor()
    # Both below the synthetic curve's top knot, where it saturates.
    low = vessel.subtracted_window(_rays(), reference_mg_per_ml=1.0)
    high = vessel.subtracted_window(_rays(), reference_mg_per_ml=4.0)
    assert low[0] == high[0] == 0.0
    assert high[1] > low[1] > 0.0


def test_subtraction_cancels_everything_static() -> None:
    pytest.importorskip("warp")
    from xray_simulator.vessel import subtract

    mask = np.array([[0.5, 0.25]])
    live = mask * np.array([[1.0, 0.8]])
    np.testing.assert_allclose(subtract(live, mask), [[1.0, 0.8]])
