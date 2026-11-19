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

"""Tests for the radial probe Python API."""

import numpy as np
import pytest
import ultrasound_simulator
import ultrasound_simulator.cuda as rs


def identity_pose():
    return rs.Pose(
        position=np.zeros(3, dtype=np.float32),
        rotation=np.zeros(3, dtype=np.float32),
    )


def make_probe(**overrides):
    arguments = {
        "pose": identity_pose(),
        "num_scanlines": 8,
        "start_angle": 0.0,
        "dead_zone_radius": 0.75,
        "rotation_period": 0.04,
        "frequency": 20.0,
        "elevational_height": 0.5,
        "num_el_samples": 1,
        "f_num": 1.0,
        "speed_of_sound": 1.54,
        "pulse_duration": 2.0,
        "transducer_offset_radius": 0.0,
        "beam_tilt": 0.0,
        "rotation_direction": rs.RadialRotationDirection.POSITIVE,
    }
    arguments.update(overrides)
    return rs.RadialProbe(**arguments)


def test_constructor_and_radial_getters_expose_the_python_api():
    probe = make_probe()

    assert ultrasound_simulator.RadialProbe is rs.RadialProbe
    assert ultrasound_simulator.RadialRotationDirection is rs.RadialRotationDirection
    assert probe.get_num_scanlines() == 8
    assert probe.get_num_elements_x() == 8
    assert probe.get_start_angle() == pytest.approx(0.0)
    assert probe.get_dead_zone_radius() == pytest.approx(0.75)
    assert probe.get_rotation_period() == pytest.approx(0.04)
    assert probe.get_transducer_offset_radius() == pytest.approx(0.0)
    assert probe.get_beam_tilt() == pytest.approx(0.0)
    assert probe.get_rotation_direction() == rs.RadialRotationDirection.POSITIVE
    assert probe.get_frequency() == pytest.approx(20.0)
    assert probe.get_speed_of_sound() == pytest.approx(1.54)
    assert probe.get_pulse_duration() == pytest.approx(2.0)


def test_cardinal_scanlines_share_one_origin_and_point_radially():
    probe = make_probe(num_scanlines=4)
    expected_angles = np.array([0.0, 90.0, 180.0, 270.0])
    expected_directions = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
        ]
    )

    angles = np.array([probe.get_a_line_angle(index) for index in range(4)])
    origins = np.stack([probe.get_element_position(index) for index in range(4)])
    directions = np.stack([probe.get_element_direction(index) for index in range(4)])

    np.testing.assert_allclose(angles, expected_angles, atol=1e-6)
    np.testing.assert_allclose(origins, np.zeros((4, 3)), atol=1e-6)
    np.testing.assert_allclose(directions, expected_directions, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(directions[:, 1], 0.0, atol=1e-6)


def test_scanline_angles_cover_one_half_open_revolution_without_a_duplicate_seam():
    num_scanlines = 512
    probe = make_probe(num_scanlines=num_scanlines)
    angles = np.array(
        [probe.get_a_line_angle(index) for index in range(num_scanlines)]
    )
    angular_pitch = 360.0 / num_scanlines

    assert angles[0] == pytest.approx(0.0)
    assert angles[-1] == pytest.approx(360.0 - angular_pitch)
    assert angles[-1] < 360.0
    np.testing.assert_allclose(np.diff(angles), angular_pitch, atol=1e-6)

    first_direction = probe.get_element_direction(0)
    last_direction = probe.get_element_direction(num_scanlines - 1)
    assert not np.allclose(first_direction, last_direction, atol=1e-6)


def test_start_angle_is_in_degrees_and_rotates_scanline_directions():
    probe = make_probe(num_scanlines=4, start_angle=45.0)
    expected_angles = np.array([45.0, 135.0, 225.0, 315.0])
    root_half = np.sqrt(0.5)
    expected_directions = np.array(
        [
            [root_half, 0.0, root_half],
            [root_half, 0.0, -root_half],
            [-root_half, 0.0, -root_half],
            [-root_half, 0.0, root_half],
        ]
    )

    angles = np.array([probe.get_a_line_angle(index) for index in range(4)])
    directions = np.stack([probe.get_element_direction(index) for index in range(4)])

    assert probe.get_start_angle() == pytest.approx(45.0)
    np.testing.assert_allclose(angles, expected_angles, atol=1e-6)
    np.testing.assert_allclose(directions, expected_directions, atol=1e-6)


def test_intrinsic_offset_tilt_and_negative_rotation_define_a_conical_orbit():
    probe = make_probe(
        num_scanlines=4,
        transducer_offset_radius=0.25,
        beam_tilt=30.0,
        rotation_direction=rs.RadialRotationDirection.NEGATIVE,
    )
    expected_angles = np.array([0.0, -90.0, -180.0, -270.0])
    expected_origins = np.array(
        [
            [0.0, 0.0, 0.25],
            [-0.25, 0.0, 0.0],
            [0.0, 0.0, -0.25],
            [0.25, 0.0, 0.0],
        ]
    )
    root_three_quarters = np.sqrt(0.75)
    expected_directions = np.array(
        [
            [0.0, 0.5, root_three_quarters],
            [-root_three_quarters, 0.5, 0.0],
            [0.0, 0.5, -root_three_quarters],
            [root_three_quarters, 0.5, 0.0],
        ]
    )

    angles = np.array([probe.get_a_line_angle(index) for index in range(4)])
    origins = np.stack([probe.get_element_position(index) for index in range(4)])
    directions = np.stack([probe.get_element_direction(index) for index in range(4)])

    np.testing.assert_allclose(angles, expected_angles, atol=1e-6)
    np.testing.assert_allclose(origins, expected_origins, atol=1e-6)
    np.testing.assert_allclose(directions, expected_directions, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(origins[:, [0, 2]], axis=1), 0.25, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(directions[:, 1], 0.5, atol=1e-6)


def test_intrinsic_geometry_setters_update_the_ray_pose():
    probe = make_probe(num_scanlines=4)

    probe.set_transducer_offset_radius(0.2)
    probe.set_beam_tilt(-10.0)
    probe.set_rotation_direction(rs.RadialRotationDirection.NEGATIVE)

    assert probe.get_transducer_offset_radius() == pytest.approx(0.2)
    assert probe.get_beam_tilt() == pytest.approx(-10.0)
    assert probe.get_rotation_direction() == rs.RadialRotationDirection.NEGATIVE
    assert probe.get_a_line_angle(1) == pytest.approx(-90.0)
    np.testing.assert_allclose(probe.get_element_position(0), [0.0, 0.0, 0.2], atol=1e-6)
    assert probe.get_element_direction(0)[1] == pytest.approx(np.sin(np.deg2rad(-10.0)))


@pytest.mark.parametrize(
    "rotation_direction",
    [rs.RadialRotationDirection.POSITIVE, rs.RadialRotationDirection.NEGATIVE],
)
def test_a_line_timestamps_are_monotonic_half_open_and_have_both_aliases(rotation_direction):
    num_scanlines = 8
    rotation_period = 0.04
    probe = make_probe(
        num_scanlines=num_scanlines,
        rotation_period=rotation_period,
        rotation_direction=rotation_direction,
    )
    expected = np.arange(num_scanlines) * rotation_period / num_scanlines

    individual = np.array(
        [probe.get_a_line_timestamp(index) for index in range(num_scanlines)]
    )
    a_line_timestamps = np.asarray(probe.get_a_line_timestamps())
    scanline_timestamps = np.asarray(probe.get_scanline_timestamps())

    assert a_line_timestamps.shape == (num_scanlines,)
    assert scanline_timestamps.shape == (num_scanlines,)
    np.testing.assert_allclose(individual, expected, atol=1e-9)
    np.testing.assert_allclose(a_line_timestamps, expected, atol=1e-9)
    np.testing.assert_allclose(scanline_timestamps, expected, atol=1e-9)
    assert np.all(np.diff(individual) > 0.0)
    assert individual[-1] < rotation_period


def test_pose_transforms_the_common_origin_and_cardinal_directions():
    position = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    rotation = np.array([0.0, np.pi / 2.0, 0.0], dtype=np.float32)
    pose = rs.Pose(position=position, rotation=rotation)
    probe = make_probe(pose=pose, num_scanlines=4)
    expected_directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    origins = np.stack([probe.get_element_position(index) for index in range(4)])
    directions = np.stack([probe.get_element_direction(index) for index in range(4)])

    np.testing.assert_allclose(origins, np.tile(position, (4, 1)), atol=1e-6)
    np.testing.assert_allclose(directions, expected_directions, atol=1e-6)


def test_pose_transforms_an_offset_emitter_and_tilted_beam_together():
    position = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    rotation = np.array([0.0, np.pi / 2.0, 0.0], dtype=np.float32)
    probe = make_probe(
        pose=rs.Pose(position=position, rotation=rotation),
        num_scanlines=4,
        transducer_offset_radius=0.25,
        beam_tilt=30.0,
    )

    np.testing.assert_allclose(probe.get_element_position(0), [1.25, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(
        probe.get_element_direction(0), [np.sqrt(0.75), 0.5, 0.0], atol=1e-6
    )


@pytest.mark.parametrize(
    ("argument", "invalid_value"),
    [
        ("num_scanlines", 0),
        ("dead_zone_radius", -0.01),
        ("rotation_period", 0.0),
        ("rotation_period", -0.01),
        ("transducer_offset_radius", -0.01),
        ("transducer_offset_radius", np.inf),
        ("transducer_offset_radius", np.nan),
        ("beam_tilt", -90.0),
        ("beam_tilt", 90.0),
        ("beam_tilt", np.inf),
        ("beam_tilt", np.nan),
    ],
)
def test_radial_specific_constructor_arguments_are_validated(argument, invalid_value):
    with pytest.raises(ValueError):
        make_probe(**{argument: invalid_value})


@pytest.mark.parametrize(
    ("method_name", "invalid_value"),
    [
        ("set_transducer_offset_radius", -0.01),
        ("set_transducer_offset_radius", np.inf),
        ("set_beam_tilt", -90.0),
        ("set_beam_tilt", 90.0),
        ("set_beam_tilt", np.nan),
    ],
)
def test_intrinsic_geometry_setters_reject_invalid_values(method_name, invalid_value):
    probe = make_probe()

    with pytest.raises(ValueError):
        getattr(probe, method_name)(invalid_value)


@pytest.mark.parametrize("method_name", ["get_a_line_angle", "get_a_line_timestamp"])
def test_a_line_access_rejects_an_index_past_the_last_scanline(method_name):
    probe = make_probe(num_scanlines=4)

    with pytest.raises(IndexError):
        getattr(probe, method_name)(4)


def make_simulator(**sim_overrides):
    materials = rs.Materials()
    world = rs.World("water")
    world.add(rs.Sphere(np.array([0.0, 0.0, 5.0], dtype=np.float32), 1.0, materials.get_index("bone")))
    sim_params = rs.SimParams()
    sim_params.t_far = 10.0
    sim_params.buffer_size = 4096
    sim_params.max_depth = 2
    sim_params.use_scattering = False
    sim_params.conv_psf = False
    sim_params.median_clip_filter = False
    sim_params.enable_cuda_timing = False
    for name, value in sim_overrides.items():
        setattr(sim_params, name, value)
    # The simulator borrows the world and materials, so hand them back to keep them alive.
    return rs.RaytracingUltrasoundSimulator(world, materials), sim_params, (world, materials)


@pytest.mark.gpu
@pytest.mark.parametrize("b_mode_size", [(101, 101), (201, 101), (101, 201)])
def test_image_bounds_match_the_scan_converted_pixel_grid(b_mode_size):
    """The circle is fit to the shorter side, so the longer side extends past +-t_far."""
    simulator, sim_params, _scene = make_simulator(b_mode_size=b_mode_size)
    simulator.simulate_with_metadata(make_probe(), sim_params)

    width, height = b_mode_size
    spacing = 2.0 * sim_params.t_far / (min(width, height) - 1)
    assert simulator.get_max_x() == pytest.approx(0.5 * spacing * (width - 1), rel=1e-5)
    assert simulator.get_min_x() == pytest.approx(-0.5 * spacing * (width - 1), rel=1e-5)
    assert simulator.get_max_z() == pytest.approx(0.5 * spacing * (height - 1), rel=1e-5)
    assert simulator.get_min_z() == pytest.approx(-0.5 * spacing * (height - 1), rel=1e-5)


@pytest.mark.gpu
def test_b_mode_size_beyond_32_bit_indexing_is_rejected_before_allocation():
    simulator, sim_params, _scene = make_simulator(b_mode_size=(65537, 65537))
    with pytest.raises(Exception, match="2\\^32"):
        simulator.simulate_with_metadata(make_probe(), sim_params)
