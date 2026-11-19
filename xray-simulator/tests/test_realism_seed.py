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

"""How render_frame(), render_cine(), and stream() seed realism noise, CPU only."""

from itertools import combinations

import numpy as np
import pytest
from xray_simulator import Pose, SimulatorConfig, VolumePreprocessor
from xray_simulator.simulator import xray_simulator

SHAPE = (12, 16)


@pytest.fixture
def make_simulator(monkeypatch, sample_hu_volume, sample_spacing):
    """Build simulators with realism noise on and GPU ray marching replaced."""

    def initialize(simulator):
        class Renderer:
            def render(self, rotation, translation):
                # Noise-free and a function of the pose alone, so frames at one pose differ only by noise.
                gradient = np.linspace(0.2, 0.8, SHAPE[0] * SHAPE[1], dtype=np.float32).reshape(SHAPE)
                return simulator.config.physics.i0 * gradient * (1.0 - 0.1 * rotation[1])

        simulator._renderer = Renderer()

    monkeypatch.setattr(xray_simulator, "_init_renderer", initialize)
    volume = VolumePreprocessor.from_numpy(sample_hu_volume, spacing_zyx_mm=sample_spacing).preprocess()

    def make(seed):
        config = SimulatorConfig().with_realism(enabled=True, poisson_photons=500.0, gaussian_sigma=0.01, seed=seed)
        return xray_simulator(volume, config)

    return make


def poses(count):
    return [Pose(rotation=(0.0, 0.1 * i, 0.0)) for i in range(count)]


def stream_images(simulator, pose_list):
    return [frame.image for frame in simulator.stream(iter(pose_list))]


def test_consecutive_stream_frames_draw_independent_noise_under_a_fixed_seed(make_simulator):
    images = stream_images(make_simulator(seed=7), [Pose()] * 3)

    for a, b in combinations(images, 2):
        assert not np.array_equal(a, b)


def test_stream_matches_render_cine_frame_for_frame(make_simulator):
    simulator = make_simulator(seed=7)

    streamed = stream_images(simulator, poses(4))
    cine = simulator.render_cine(poses(4), progress=False)

    assert len(streamed) == len(cine)
    for image, frame in zip(streamed, cine):
        np.testing.assert_array_equal(image, frame.image)


def test_stream_frame_i_draws_its_noise_with_seed_plus_i(make_simulator):
    streamed = stream_images(make_simulator(seed=7), [Pose()] * 3)

    for i, image in enumerate(streamed):
        np.testing.assert_array_equal(image, make_simulator(seed=7 + i).render_frame(pose=Pose()).image)


def test_a_fixed_seed_reproduces_the_stream_across_calls_and_simulators(make_simulator):
    simulator = make_simulator(seed=7)
    first = stream_images(simulator, poses(3))

    again = stream_images(simulator, poses(3))
    elsewhere = stream_images(make_simulator(seed=7), poses(3))
    other_seed = stream_images(make_simulator(seed=1000), poses(3))

    for i, image in enumerate(first):
        np.testing.assert_array_equal(again[i], image)
        np.testing.assert_array_equal(elsewhere[i], image)
        assert not np.array_equal(other_seed[i], image)


def test_render_frame_repeats_its_noise_under_a_fixed_seed(make_simulator):
    simulator = make_simulator(seed=7)
    first = simulator.render_frame(pose=Pose())

    list(simulator.stream(iter([Pose()] * 2)))
    again = simulator.render_frame(pose=Pose())

    np.testing.assert_array_equal(again.image, first.image)


def test_a_null_seed_draws_fresh_noise_for_every_frame_and_call(make_simulator):
    simulator = make_simulator(seed=None)

    first = stream_images(simulator, [Pose()] * 2)
    second = stream_images(simulator, [Pose()] * 2)

    assert not np.array_equal(first[0], first[1])
    assert not np.array_equal(first[0], second[0])
    assert not np.array_equal(simulator.render_frame(pose=Pose()).image, simulator.render_frame(pose=Pose()).image)


@pytest.mark.parametrize("render", [
    lambda simulator: simulator.render_frame(pose=Pose.ap()),
    lambda simulator: list(simulator.stream(iter([Pose.ap()]))),
], ids=["render_frame", "stream"])
def test_the_view_frame_warning_points_at_the_caller(make_simulator, render):
    simulator = make_simulator(seed=7)

    with pytest.warns(UserWarning, match="AP") as record:
        render(simulator)

    assert [warning.filename for warning in record] == [__file__]
