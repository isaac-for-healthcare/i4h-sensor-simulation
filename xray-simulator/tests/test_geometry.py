# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for C-arm pose geometry and the clinical view presets.

These run on the CPU, with no GPU or Slang required. The reference renderer below
reimplements the shader's cone-beam ray marching in numpy, which lets the tests check that
a fiducial lands where the closed-form projection says it should, and that anatomical
landmarks appear on the expected side of a rendered view.
"""

from __future__ import annotations

import numpy as np
import pytest
from xray_simulator.geometry import (
    ANTERIOR,
    CANONICAL_FRAME,
    INFERIOR,
    LEFT,
    POSTERIOR,
    RIGHT,
    SUPERIOR,
    VIEWS,
    clinical_angles_to_rotation,
    euler_zxy_to_matrix,
    matrix_to_euler_zxy,
    project_point_to_detector,
    rotation_to_clinical_angles,
    view_frame_warning,
    view_matrix,
    view_rotation,
    volume_center_xyz_mm,
)
from xray_simulator.simulator import Pose

# Small detector and coarse steps: these tests care about which side of the image things
# land on, not about image quality.
GEOMETRY = {
    "source_to_detector_mm": 1000.0,
    "source_to_isocenter_mm": 500.0,
    "detector_width_px": 96,
    "detector_height_px": 96,
    "pixel_spacing_mm": 4.0,
}

SHAPE_ZYX = (64, 48, 48)
SPACING_ZYX_MM = (3.0, 3.0, 3.0)

# Offsets from the volume center, in mm, along canonical patient directions. Kept well
# inside the volume so every marker stays in the field of view for all views.
MARKER_OFFSET_MM = 60.0
MARKERS = {
    "head": SUPERIOR,
    "feet": INFERIOR,
    "left": LEFT,
    "right": RIGHT,
    "front": ANTERIOR,
    "back": POSTERIOR,
}


def marker_position(name: str, origin_xyz_mm=(0.0, 0.0, 0.0)) -> np.ndarray:
    """World position of a named anatomical marker."""
    center = volume_center_xyz_mm(SHAPE_ZYX, SPACING_ZYX_MM, origin_xyz_mm)
    return center + MARKER_OFFSET_MM * MARKERS[name]


def marker_volume(name: str, origin_xyz_mm=(0.0, 0.0, 0.0), radius_mm: float = 9.0) -> np.ndarray:
    """Volume holding a single dense marker sphere and nothing else."""
    sz, sy, sx = SPACING_ZYX_MM
    z, y, x = SHAPE_ZYX
    # Voxel center coordinates, world X from axis 2, world Y from axis 1, world Z from axis 0.
    world_z = origin_xyz_mm[2] + (np.arange(z) + 0.5) * sz
    world_y = origin_xyz_mm[1] + (np.arange(y) + 0.5) * sy
    world_x = origin_xyz_mm[0] + (np.arange(x) + 0.5) * sx

    target = marker_position(name, origin_xyz_mm)
    dz = world_z[:, None, None] - target[2]
    dy = world_y[None, :, None] - target[1]
    dx = world_x[None, None, :] - target[0]

    volume = np.zeros(SHAPE_ZYX, dtype=np.float32)
    volume[(dx**2 + dy**2 + dz**2) <= radius_mm**2] = 0.05
    return volume


def render_line_integral(
    mu_volume: np.ndarray,
    rotation: tuple[float, float, float],
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    origin_xyz_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    step_mm: float = 1.5,
) -> np.ndarray:
    """Reference CPU renderer mirroring the shader's cone-beam geometry.

    Returns the raw line integral of mu, so brighter means more attenuating. Sampling is
    nearest-neighbor rather than trilinear, which is enough to locate a marker to within a
    pixel.
    """
    rotation_matrix = euler_zxy_to_matrix(rotation)
    center = volume_center_xyz_mm(mu_volume.shape, SPACING_ZYX_MM, origin_xyz_mm)
    offset = center + np.asarray(translation, dtype=np.float64)

    sid = GEOMETRY["source_to_isocenter_mm"]
    sdd = GEOMETRY["source_to_detector_mm"]
    width = GEOMETRY["detector_width_px"]
    height = GEOMETRY["detector_height_px"]
    pitch = GEOMETRY["pixel_spacing_mm"]

    source = rotation_matrix @ np.array([0.0, 0.0, -sid]) + offset

    columns = (np.arange(width) + 0.5 - 0.5 * width) * pitch
    rows = (np.arange(height) + 0.5 - 0.5 * height) * pitch
    local = np.empty((height, width, 3))
    local[..., 0] = columns[None, :]
    local[..., 1] = rows[:, None]
    local[..., 2] = sdd - sid
    detector = local @ rotation_matrix.T + offset

    directions = detector - source
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    spacing_xyz = np.array(SPACING_ZYX_MM[::-1])
    box_min = np.asarray(origin_xyz_mm, dtype=np.float64)
    box_max = box_min + np.array(mu_volume.shape[::-1]) * spacing_xyz

    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / directions
        t0 = (box_min - source) * inv
        t1 = (box_max - source) * inv
    t_near = np.nanmax(np.minimum(t0, t1), axis=-1)
    t_far = np.nanmin(np.maximum(t0, t1), axis=-1)
    hit = t_far > np.maximum(t_near, 0.0)

    t_start = np.maximum(t_near, 0.0)
    num_steps = int(np.ceil((t_far[hit] - t_start[hit]).max() / step_mm))
    offsets = (np.arange(num_steps) + 0.5) * step_mm

    image = np.zeros((height, width))
    dims_xyz = np.array(mu_volume.shape[::-1])
    for step in offsets:
        t = t_start + step
        active = hit & (t < t_far)
        if not active.any():
            break
        points = source + directions[active] * t[active][:, None]
        voxel = np.floor((points - box_min) / spacing_xyz).astype(int)
        inside = np.all((voxel >= 0) & (voxel < dims_xyz), axis=-1)
        contribution = np.zeros(voxel.shape[0])
        valid = voxel[inside]
        contribution[inside] = mu_volume[valid[:, 2], valid[:, 1], valid[:, 0]]
        image[active] += contribution * step_mm

    return image


def intensity_centroid(image: np.ndarray) -> tuple[float, float]:
    """Intensity-weighted (column, row) centroid of an image."""
    total = image.sum()
    assert total > 0, "expected the marker to be visible in the rendered image"
    rows, columns = np.indices(image.shape)
    return (float((image * columns).sum() / total), float((image * rows).sum() / total))


def project_marker(name: str, pose: Pose, origin_xyz_mm=(0.0, 0.0, 0.0)):
    """Projected (column, row) of a named marker under a pose."""
    return project_point_to_detector(
        marker_position(name, origin_xyz_mm),
        pose.rotation,
        pose.translation,
        isocenter_xyz_mm=volume_center_xyz_mm(SHAPE_ZYX, SPACING_ZYX_MM, origin_xyz_mm),
        **GEOMETRY,
    )


class TestEulerConversion:
    """The Euler helpers must agree with the shader's ZXY convention."""

    def test_identity(self):
        assert np.allclose(euler_zxy_to_matrix((0.0, 0.0, 0.0)), np.eye(3))

    def test_single_axis_rotations_match_textbook_matrices(self):
        angle = 0.3
        cos, sin = np.cos(angle), np.sin(angle)
        assert np.allclose(
            euler_zxy_to_matrix((angle, 0.0, 0.0)),
            [[1, 0, 0], [0, cos, -sin], [0, sin, cos]],
        )
        assert np.allclose(
            euler_zxy_to_matrix((0.0, angle, 0.0)),
            [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]],
        )
        assert np.allclose(
            euler_zxy_to_matrix((0.0, 0.0, angle)),
            [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]],
        )

    def test_composition_order_is_z_then_x_then_y(self):
        rx, ry, rz = 0.2, -0.4, 0.7
        expected = (
            euler_zxy_to_matrix((0.0, 0.0, rz))
            @ euler_zxy_to_matrix((rx, 0.0, 0.0))
            @ euler_zxy_to_matrix((0.0, ry, 0.0))
        )
        assert np.allclose(euler_zxy_to_matrix((rx, ry, rz)), expected)

    def test_roundtrip_recovers_the_matrix(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            angles = rng.uniform(-np.pi / 2 + 0.05, np.pi / 2 - 0.05, size=3)
            matrix = euler_zxy_to_matrix(tuple(angles))
            assert np.allclose(euler_zxy_to_matrix(matrix_to_euler_zxy(matrix)), matrix, atol=1e-9)

    def test_roundtrip_at_gimbal_lock(self):
        # Every view preset sits at rx = -90 degrees, where only rz - ry is determined.
        for sign in (-1.0, 1.0):
            matrix = euler_zxy_to_matrix((sign * np.pi / 2, 0.4, -0.9))
            recovered = matrix_to_euler_zxy(matrix)
            assert recovered[1] == 0.0
            assert np.allclose(euler_zxy_to_matrix(recovered), matrix, atol=1e-9)


class TestViewPresets:
    """Each preset must be a proper rotation aiming the beam along the intended axis."""

    @pytest.mark.parametrize("view", VIEWS)
    def test_is_a_right_handed_rotation(self, view):
        matrix = view_matrix(view)
        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(matrix), 1.0)

    @pytest.mark.parametrize(
        "view,beam,image_column",
        [
            ("ap", POSTERIOR, LEFT),
            ("pa", ANTERIOR, RIGHT),
            ("lateral_left", LEFT, ANTERIOR),
            ("lateral_right", RIGHT, POSTERIOR),
        ],
    )
    def test_beam_and_column_directions(self, view, beam, image_column):
        matrix = view_matrix(view)
        assert np.allclose(matrix[:, 2], beam, atol=1e-12)
        assert np.allclose(matrix[:, 0], image_column, atol=1e-12)

    @pytest.mark.parametrize("view", VIEWS)
    def test_rows_run_toward_the_feet_so_the_head_is_up(self, view):
        assert np.allclose(view_matrix(view)[:, 1], INFERIOR, atol=1e-12)

    @pytest.mark.parametrize(
        "view,expected_deg",
        [
            ("ap", (-90.0, 0.0, 0.0)),
            ("pa", (-90.0, 0.0, 180.0)),
            ("lateral_left", (-90.0, 0.0, -90.0)),
            ("lateral_right", (-90.0, 0.0, 90.0)),
        ],
    )
    def test_euler_angles_are_stable(self, view, expected_deg):
        # Pinned so that a change in the shader convention breaks loudly rather than
        # silently rotating everyone's saved poses.
        actual = np.degrees(view_rotation(view))
        assert np.allclose(np.abs(actual), np.abs(expected_deg), atol=1e-9)

    def test_unknown_view_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown view"):
            view_matrix("oblique")

    def test_identity_pose_looks_along_the_patient_long_axis(self):
        # Documents the legacy default: the identity pose is an axial projection, not a
        # radiographic view. Changing it would silently invalidate saved poses.
        assert np.allclose(euler_zxy_to_matrix((0.0, 0.0, 0.0))[:, 2], SUPERIOR)


class TestClinicalAngles:
    """LAO/RAO and CRAN/CAUD must behave the way a cath-lab operator expects."""

    def test_zero_angles_is_the_pa_view(self):
        assert np.allclose(
            euler_zxy_to_matrix(clinical_angles_to_rotation(0.0, 0.0)),
            view_matrix("pa"),
            atol=1e-12,
        )

    def test_lao_swings_the_detector_toward_the_patient_left(self):
        beam = euler_zxy_to_matrix(clinical_angles_to_rotation(30.0, 0.0))[:, 2]
        assert beam @ LEFT > 0
        assert beam @ ANTERIOR > 0

    def test_rao_swings_the_detector_toward_the_patient_right(self):
        beam = euler_zxy_to_matrix(clinical_angles_to_rotation(-30.0, 0.0))[:, 2]
        assert beam @ RIGHT > 0
        assert beam @ ANTERIOR > 0

    def test_cranial_tilts_the_detector_toward_the_head(self):
        beam = euler_zxy_to_matrix(clinical_angles_to_rotation(0.0, 25.0))[:, 2]
        assert beam @ SUPERIOR > 0

    def test_caudal_tilts_the_detector_toward_the_feet(self):
        beam = euler_zxy_to_matrix(clinical_angles_to_rotation(0.0, -25.0))[:, 2]
        assert beam @ INFERIOR > 0

    def test_lao_90_is_the_left_lateral_view(self):
        assert np.allclose(
            euler_zxy_to_matrix(clinical_angles_to_rotation(90.0, 0.0)),
            view_matrix("lateral_left"),
            atol=1e-12,
        )

    def test_rao_90_is_the_right_lateral_view(self):
        assert np.allclose(
            euler_zxy_to_matrix(clinical_angles_to_rotation(-90.0, 0.0)),
            view_matrix("lateral_right"),
            atol=1e-12,
        )

    @pytest.mark.parametrize(
        "primary,secondary",
        [(0.0, 0.0), (30.0, 0.0), (-30.0, 20.0), (45.0, -25.0), (89.0, 15.0), (-60.0, -40.0)],
    )
    def test_roundtrip(self, primary, secondary):
        reported = rotation_to_clinical_angles(clinical_angles_to_rotation(primary, secondary))
        assert np.allclose(reported, (primary, secondary), atol=1e-6)

    def test_pose_helper_reports_its_own_angles(self):
        pose = Pose.from_clinical_angles(35.0, -15.0)
        assert np.allclose(pose.clinical_angles(), (35.0, -15.0), atol=1e-6)
        assert pose.view == "LAO 35 / CAUD 15"


class TestFiducialProjection:
    """Anatomical markers must land on the clinically expected side of the image."""

    @staticmethod
    def _center() -> tuple[float, float]:
        return (
            0.5 * GEOMETRY["detector_width_px"] - 0.5,
            0.5 * GEOMETRY["detector_height_px"] - 0.5,
        )

    def test_ap_view_has_head_up_and_patient_left_on_the_right(self):
        pose = Pose.ap()
        center_column, center_row = self._center()

        assert project_marker("head", pose)[1] < center_row
        assert project_marker("feet", pose)[1] > center_row
        assert project_marker("left", pose)[0] > center_column
        assert project_marker("right", pose)[0] < center_column

    def test_pa_view_has_head_up_and_mirrors_left_and_right(self):
        pose = Pose.pa()
        center_column, center_row = self._center()

        assert project_marker("head", pose)[1] < center_row
        assert project_marker("left", pose)[0] < center_column
        assert project_marker("right", pose)[0] > center_column

    @pytest.mark.parametrize(
        "side,anterior_side",
        [("left", "right"), ("right", "left")],
    )
    def test_lateral_views_have_head_up_and_the_expected_anterior_side(self, side, anterior_side):
        pose = Pose.lateral(side)
        center_column, center_row = self._center()

        assert project_marker("head", pose)[1] < center_row
        front_column = project_marker("front", pose)[0]
        if anterior_side == "left":
            assert front_column < center_column
        else:
            assert front_column > center_column

    def test_markers_along_the_beam_project_near_the_image_center(self):
        # For an AP view the front and back markers are stacked along the beam, so both
        # project close to the center; a wrong beam axis would push them apart.
        pose = Pose.ap()
        center_column, center_row = self._center()
        for name in ("front", "back"):
            column, row = project_marker(name, pose)
            assert abs(column - center_column) < 1.0
            assert abs(row - center_row) < 1.0

    def test_magnification_grows_toward_the_source(self):
        # Two points the same distance to the patient's left, one nearer the source. In an
        # AP view the source is anterior, so the anterior point is magnified more.
        pose = Pose.ap()
        center_column, _ = self._center()
        center = volume_center_xyz_mm(SHAPE_ZYX, SPACING_ZYX_MM)
        lateral = 40.0 * LEFT
        depth = 50.0 * ANTERIOR

        def column_offset(point):
            column, _ = project_point_to_detector(
                point, pose.rotation, pose.translation, isocenter_xyz_mm=center, **GEOMETRY
            )
            return column - center_column

        near_source = column_offset(center + lateral + depth)
        far_from_source = column_offset(center + lateral - depth)
        assert near_source > far_from_source > 0

    def test_translation_shifts_the_image(self):
        pose = Pose.ap()
        shifted = Pose(rotation=pose.rotation, translation=(0.0, 0.0, 20.0))
        # Moving the isocenter toward the head moves anatomy down in the image.
        assert project_marker("head", shifted)[1] > project_marker("head", pose)[1]

    def test_points_behind_the_source_do_not_project(self):
        pose = Pose.ap()
        center = volume_center_xyz_mm(SHAPE_ZYX, SPACING_ZYX_MM)
        behind = center + POSTERIOR * -2.0 * GEOMETRY["source_to_isocenter_mm"]
        assert (
            project_point_to_detector(
                behind, pose.rotation, pose.translation, isocenter_xyz_mm=center, **GEOMETRY
            )
            is None
        )

    def test_projection_is_independent_of_the_volume_origin(self):
        pose = Pose.ap()
        at_zero = project_marker("left", pose)
        shifted = project_marker("left", pose, origin_xyz_mm=(-123.0, 45.0, -678.0))
        assert np.allclose(at_zero, shifted, atol=1e-6)


class TestFiducialRendering:
    """The closed-form projection must agree with ray-marched images."""

    @pytest.mark.parametrize("view", ["ap", "pa", "lateral_left", "lateral_right"])
    @pytest.mark.parametrize("marker", ["head", "left", "front"])
    def test_rendered_marker_lands_where_the_projection_says(self, view, marker):
        pose = Pose(rotation=view_rotation(view))
        image = render_line_integral(marker_volume(marker), pose.rotation)
        centroid = intensity_centroid(image)
        expected = project_marker(marker, pose)
        # Within a pixel and a half, which is the resolution of a nearest-neighbor march
        # through a marker a few voxels across.
        assert np.allclose(centroid, expected, atol=1.5)

    def test_rendered_ap_view_puts_the_head_above_the_feet(self):
        pose = Pose.ap()
        head_row = intensity_centroid(render_line_integral(marker_volume("head"), pose.rotation))[1]
        feet_row = intensity_centroid(render_line_integral(marker_volume("feet"), pose.rotation))[1]
        assert head_row < feet_row

    def test_rendered_ap_view_puts_patient_left_on_the_right(self):
        pose = Pose.ap()
        left = intensity_centroid(render_line_integral(marker_volume("left"), pose.rotation))[0]
        right = intensity_centroid(render_line_integral(marker_volume("right"), pose.rotation))[0]
        assert left > right

    def test_rendered_view_is_unchanged_by_the_volume_origin(self):
        # Passing the real origin moves the source, detector and volume together, so the
        # image must be identical to rendering at the origin.
        pose = Pose.ap()
        origin = (-123.0, 45.0, -678.0)
        at_zero = render_line_integral(marker_volume("left"), pose.rotation)
        shifted = render_line_integral(
            marker_volume("left", origin), pose.rotation, origin_xyz_mm=origin
        )
        assert np.allclose(at_zero, shifted, atol=1e-6)


class TestVolumeCenter:
    """The isocenter helper is what makes translations expressible in patient space."""

    def test_center_of_a_volume_at_the_world_origin(self):
        assert np.allclose(
            volume_center_xyz_mm((4, 6, 8), (2.0, 3.0, 4.0)),
            (16.0, 9.0, 4.0),
        )

    def test_center_follows_the_origin(self):
        origin = (10.0, -20.0, 30.0)
        assert np.allclose(
            volume_center_xyz_mm((4, 6, 8), (2.0, 3.0, 4.0), origin),
            np.array((16.0, 9.0, 4.0)) + origin,
        )

    def test_translation_to_a_patient_point_recenters_the_projection(self):
        pose = Pose.ap()
        origin = (-50.0, 20.0, -300.0)
        center = volume_center_xyz_mm(SHAPE_ZYX, SPACING_ZYX_MM, origin)
        target = marker_position("left", origin)

        aimed = Pose(rotation=pose.rotation, translation=tuple(target - center))
        column, row = project_point_to_detector(
            target, aimed.rotation, aimed.translation, isocenter_xyz_mm=center, **GEOMETRY
        )
        assert np.isclose(column, 0.5 * GEOMETRY["detector_width_px"] - 0.5)
        assert np.isclose(row, 0.5 * GEOMETRY["detector_height_px"] - 0.5)


class TestViewFrameWarning:
    """Labeled views only mean something on a volume in the canonical patient frame."""

    def test_no_warning_for_a_canonical_volume(self):
        assert view_frame_warning("AP", CANONICAL_FRAME) is None

    def test_no_warning_for_an_unlabeled_pose(self):
        assert view_frame_warning(None, None) is None

    def test_warns_when_the_frame_is_unknown(self):
        message = view_frame_warning("AP", None)
        assert message is not None
        assert "does not record an anatomical frame" in message

    def test_warns_when_the_frame_is_something_else(self):
        message = view_frame_warning("PA", "RAS")
        assert message is not None
        assert "'RAS'" in message


class TestPosePresets:
    """The presets are the supported way to reach a radiographic view."""

    def test_labels(self):
        assert Pose.ap().view == "AP"
        assert Pose.pa().view == "PA"
        assert Pose.lateral("right").view == "Right lateral"
        assert Pose.from_clinical_angles(20.0, 10.0).view == "LAO 20 / CRAN 10"

    def test_raw_poses_carry_no_label(self):
        assert Pose().view is None
        assert Pose(rotation=(0.1, 0.2, 0.3)).view is None

    def test_translation_is_preserved(self):
        translation = (1.0, 2.0, 3.0)
        for pose in (
            Pose.ap(translation),
            Pose.pa(translation),
            Pose.lateral("left", translation),
            Pose.from_clinical_angles(10.0, 5.0, translation),
        ):
            assert pose.translation == translation

    def test_bad_lateral_side_is_rejected(self):
        with pytest.raises(ValueError, match="left"):
            Pose.lateral("lateral")

    def test_dict_roundtrip_keeps_the_label(self):
        pose = Pose.lateral("left", (1.0, 0.0, -2.0))
        restored = Pose.from_dict(pose.to_dict())
        assert restored.view == pose.view
        assert np.allclose(restored.rotation, pose.rotation)
        assert restored.translation == pose.translation
