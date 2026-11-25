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

"""Detector rays, the absorber protocol and the catheter integrator."""

import numpy as np
import pytest
from xray_simulator.absorbers import Absorber, DetectorRays, composite
from xray_simulator.config import CarmGeometry
from xray_simulator.geometry import project_point_to_detector


def test_pose_rays_hit_the_pixels_the_renderer_projects_onto() -> None:
    """Every ray must end on the pixel the shader's own projection assigns to points along it.

    An oblique pose, a shifted principal point and non-square pixels together, so a sign or
    axis-order slip in any of them moves the ray off its pixel.
    """
    carm = CarmGeometry(
        source_to_detector_mm=1000.0,
        source_to_isocenter_mm=600.0,
        detector_width_px=48,
        detector_height_px=32,
        pixel_spacing_mm=0.8,
        pixel_spacing_y_mm=1.1,
        principal_point_px=(30.0, 12.0),
    )
    rotation, translation, isocenter = (0.3, -0.4, 1.1), (5.0, -3.0, 12.0), (100.0, 80.0, -40.0)
    rays = DetectorRays.for_pose(
        rotation,
        translation,
        isocenter,
        source_to_detector_mm=carm.source_to_detector_mm,
        source_to_isocenter_mm=carm.source_to_isocenter_mm,
        width=carm.detector_width_px,
        height=carm.detector_height_px,
        pixel_spacing_mm=carm.pixel_spacing_mm,
        pixel_spacing_y_mm=carm.pixel_spacing_y_mm,
        detector_offset_xy_mm=carm.detector_offset_xy_mm,
    )
    centers = rays.pixel_centers_mm()
    for row, column in ((0, 0), (5, 40), (31, 47), (12, 30)):
        point = rays.source_mm + 0.6 * (centers[row, column] - rays.source_mm)
        projected = project_point_to_detector(
            point,
            rotation,
            translation,
            carm.source_to_detector_mm,
            carm.source_to_isocenter_mm,
            carm.detector_width_px,
            carm.detector_height_px,
            carm.pixel_spacing_mm,
            isocenter,
            pixel_spacing_y_mm=carm.pixel_spacing_y_mm,
            principal_point_px=carm.principal_point_px,
        )
        assert projected == pytest.approx((column, row), abs=1e-6)


def test_the_source_sits_sid_from_the_isocenter_and_sdd_from_the_detector() -> None:
    rays = DetectorRays.for_pose(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        source_to_detector_mm=1000.0,
        source_to_isocenter_mm=500.0,
        width=2,
        height=2,
        pixel_spacing_mm=1.0,
    )
    np.testing.assert_allclose(rays.source_mm, [0.0, 0.0, -500.0])
    np.testing.assert_allclose(rays.pixel_centers_mm().mean(axis=(0, 1)), [0.0, 0.0, 500.0])


def test_rays_transform_points_and_directions_differently() -> None:
    rays = DetectorRays.from_axes(
        np.eye(3),
        np.zeros(3),
        source_to_detector_mm=10.0,
        source_to_isocenter_mm=5.0,
        width=3,
        height=3,
        pixel_spacing_mm=1.0,
    )
    moved = rays.transformed(2.0 * np.eye(3), np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(moved.source_mm, 2.0 * rays.source_mm + [1.0, 0.0, 0.0])
    np.testing.assert_allclose(moved.column_step_mm, 2.0 * rays.column_step_mm)


def test_rays_reject_malformed_geometry() -> None:
    with pytest.raises(ValueError, match="detector_axes"):
        DetectorRays.from_axes(
            np.eye(2), np.zeros(3), source_to_detector_mm=1, source_to_isocenter_mm=1, width=1, height=1,
            pixel_spacing_mm=1,
        )
    with pytest.raises(ValueError, match="positive size"):
        DetectorRays(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), width=0, height=1)


class _Slab:
    """A uniform slab: every ray crosses the same line integral."""

    def __init__(self, value: float) -> None:
        self.value = value

    def line_integral(self, rays: DetectorRays, *, as_torch: bool = False) -> np.ndarray:
        return np.full((rays.height, rays.width), self.value)


def test_absorbers_multiply_into_the_transmission() -> None:
    rays = DetectorRays(np.zeros(3), np.ones(3), np.eye(3)[0], np.eye(3)[1], width=4, height=3)
    transmission = np.full((2, 3, 4), 0.5)
    assert isinstance(_Slab(0.0), Absorber)
    result = composite(transmission, [_Slab(0.25), _Slab(0.5)], rays)
    np.testing.assert_allclose(result, 0.5 * np.exp(-0.75))
    assert composite(transmission, [], rays) is transmission


def _beam_down_z() -> DetectorRays:
    return DetectorRays.from_axes(
        np.eye(3),
        np.zeros(3),
        source_to_detector_mm=1000.0,
        source_to_isocenter_mm=500.0,
        width=64,
        height=64,
        pixel_spacing_mm=1.0,
    )


def test_a_posed_catheter_is_an_absorber() -> None:
    pytest.importorskip("warp")
    from xray_simulator.absorbers import CatheterAttenuation

    catheter = CatheterAttenuation(device="cpu")
    points = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    posed = catheter.posed(points, 0.5)
    assert isinstance(posed, Absorber)
    np.testing.assert_array_equal(posed.line_integral(_beam_down_z()), catheter.line_integral(points, 0.5, _beam_down_z()))
