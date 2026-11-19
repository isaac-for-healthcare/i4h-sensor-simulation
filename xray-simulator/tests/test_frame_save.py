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

"""Frame and CineSequence file output, and the simulator's automatic frame writer, CPU only."""

import sys

import numpy as np
import pytest
from xray_simulator import CineSequence, Frame, OutputSettings, Pose, SimulatorConfig, VolumePreprocessor
from xray_simulator.simulator import xray_simulator

FORMATS = ["png", "npy", "npz"]


@pytest.fixture
def frame():
    image = np.linspace(0.0, 1.0, 12 * 16, dtype=np.float32).reshape(12, 16)
    return Frame(image=image, pose=Pose.ap())


@pytest.fixture
def make_simulator(monkeypatch, sample_hu_volume, sample_spacing):
    """Build simulators that run the production pipeline with GPU ray marching replaced."""

    def initialize(simulator):
        class Renderer:
            calls = 0

            def render(self, rotation, translation):
                # A different image on every call, so a test can tell which render a file holds.
                Renderer.calls += 1
                gradient = np.linspace(0.05, 0.95, 12 * 16, dtype=np.float32).reshape(12, 16)
                return simulator.config.physics.i0 * gradient**Renderer.calls

        simulator._renderer = Renderer()

    monkeypatch.setattr(xray_simulator, "_init_renderer", initialize)
    volume = VolumePreprocessor.from_numpy(
        sample_hu_volume, spacing_zyx_mm=sample_spacing, anatomical_frame="LPS"
    ).preprocess()

    def make(output_dir, fmt):
        if fmt == "png":
            pytest.importorskip("PIL")
        config = SimulatorConfig().with_output(save_to_disk=True, output_dir=str(output_dir), format=fmt)
        return xray_simulator(volume, config)

    return make


def load_saved(path):
    """Read back the array a frame file holds (8-bit for PNG)."""
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        with np.load(path) as archive:
            return archive["image"]
    from PIL import Image

    with Image.open(path) as png:
        return np.asarray(png)


def as_saved(image, fmt):
    """Return what a frame file in this format holds for the image."""
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8) if fmt == "png" else image


def poses(count):
    return [Pose(rotation=(0.0, 0.1 * i, 0.0)) for i in range(count)]


def test_png_is_8_bit_grayscale_with_the_rounded_pixel_values(tmp_path, frame):
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "frame.png"

    frame.save(path)

    with Image.open(path) as png:
        assert png.mode == "L"
        pixels = np.asarray(png)
    np.testing.assert_array_equal(pixels, (frame.image * 255.0 + 0.5).astype(np.uint8))
    assert list(tmp_path.iterdir()) == [path]


def test_npy_keeps_full_precision(tmp_path, frame):
    frame.save(tmp_path / "frame.npy")
    np.testing.assert_array_equal(np.load(tmp_path / "frame.npy"), frame.image)


def test_npz_is_compressed_with_the_image_key(tmp_path, frame):
    frame.save(tmp_path / "frame.npz")
    with np.load(tmp_path / "frame.npz") as archive:
        assert archive.files == ["image"]
        np.testing.assert_array_equal(archive["image"], frame.image)


@pytest.mark.parametrize("name", ["frame.jpg", "frame.pgm", "frame"])
def test_unsupported_suffix_is_rejected_without_writing(tmp_path, frame, name):
    with pytest.raises(ValueError, match="unsupported frame format"):
        frame.save(tmp_path / "out" / name)
    assert not (tmp_path / "out").exists()


def test_png_without_pillow_raises_instead_of_writing_another_format(tmp_path, frame, monkeypatch):
    monkeypatch.setitem(sys.modules, "PIL", None)

    with pytest.raises(ImportError, match="Pillow"):
        frame.save(tmp_path / "frame.png")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("frames", [0, 1])
def test_save_all_rejects_an_unsupported_format_before_creating_the_directory(tmp_path, frame, frames):
    cine = CineSequence(frames=[frame] * frames)

    with pytest.raises(ValueError, match="unsupported frame format"):
        cine.save_all(tmp_path / "out", format="jpg")
    assert not (tmp_path / "out").exists()


def test_output_settings_rejects_an_unsupported_format():
    with pytest.raises(ValueError, match="format must be"):
        OutputSettings(format="jpg")


@pytest.mark.parametrize("fmt", ["png", "npy", "npz"])
def test_save_all_writes_every_documented_format(tmp_path, frame, fmt):
    if fmt == "png":
        pytest.importorskip("PIL")
    cine = CineSequence(frames=[frame, Frame(image=frame.image, pose=frame.pose, frame_idx=1)])

    paths = cine.save_all(tmp_path, format=fmt)

    assert paths == [tmp_path / f"frame_0000.{fmt}", tmp_path / f"frame_0001.{fmt}"]
    assert sorted(tmp_path.iterdir()) == paths


@pytest.mark.parametrize("fmt", FORMATS)
def test_stream_autosaves_one_file_per_yielded_frame(tmp_path, make_simulator, fmt):
    simulator = make_simulator(tmp_path, fmt)

    frames = list(simulator.stream(iter(poses(3)), max_frames=3))

    assert [f.frame_idx for f in frames] == [0, 1, 2]
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"frame_{i:04d}.{fmt}" for i in range(3)]
    for f in frames:
        np.testing.assert_array_equal(load_saved(tmp_path / f"frame_{f.frame_idx:04d}.{fmt}"), as_saved(f.image, fmt))
    assert not np.array_equal(frames[0].image, frames[-1].image)


def test_stream_saves_each_frame_before_yielding_it(tmp_path, make_simulator):
    stream = make_simulator(tmp_path, "npy").stream(iter(poses(5)))

    first = next(stream)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["frame_0000.npy"]
    np.testing.assert_array_equal(np.load(tmp_path / "frame_0000.npy"), first.image)
    stream.close()

    assert sorted(p.name for p in tmp_path.iterdir()) == ["frame_0000.npy"]


def test_stream_stops_writing_at_max_frames(tmp_path, make_simulator):
    frames = list(make_simulator(tmp_path, "npy").stream(iter(poses(5)), max_frames=2))

    assert len(frames) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["frame_0000.npy", "frame_0001.npy"]


def test_each_stream_call_numbers_its_files_from_zero(tmp_path, make_simulator):
    simulator = make_simulator(tmp_path, "npy")
    list(simulator.stream(iter(poses(3))))

    second = list(simulator.stream(iter(poses(2))))

    assert [f.frame_idx for f in second] == [0, 1]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["frame_0000.npy", "frame_0001.npy", "frame_0002.npy"]
    for f in second:
        np.testing.assert_array_equal(np.load(tmp_path / f"frame_{f.frame_idx:04d}.npy"), f.image)


@pytest.mark.parametrize("fmt", FORMATS)
def test_render_cine_autosaves_one_file_per_frame(tmp_path, make_simulator, fmt):
    cine = make_simulator(tmp_path, fmt).render_cine(poses(3), progress=False)

    assert sorted(p.name for p in tmp_path.iterdir()) == [f"frame_{i:04d}.{fmt}" for i in range(3)]
    for f in cine:
        np.testing.assert_array_equal(load_saved(tmp_path / f"frame_{f.frame_idx:04d}.{fmt}"), as_saved(f.image, fmt))


@pytest.mark.parametrize("fmt", FORMATS)
def test_render_frame_autosave_overwrites_frame_0000(tmp_path, make_simulator, fmt):
    simulator = make_simulator(tmp_path, fmt)
    simulator.render_frame(pose=Pose.ap())

    last = simulator.render_frame(pose=Pose.pa())

    assert last.frame_idx == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"frame_0000.{fmt}"]
    np.testing.assert_array_equal(load_saved(tmp_path / f"frame_0000.{fmt}"), as_saved(last.image, fmt))


@pytest.mark.parametrize("seed", [0, 29])
def test_stream_realism_advances_per_frame_and_matches_cine(tmp_path, make_simulator, monkeypatch, seed):
    template = make_simulator(tmp_path, "npy")
    config = template.config.with_realism(enabled=True, seed=seed, gaussian_sigma=0.01).with_output(
        save_to_disk=False, keep_intensity=True
    )
    simulator = xray_simulator(template.volume, config)
    monkeypatch.setattr(
        simulator._renderer, "render", lambda *_: np.full((12, 16), 0.5 * config.physics.i0, dtype=np.float32)
    )
    stationary = [Pose.ap() for _ in range(3)]

    stream = list(simulator.stream(iter(stationary)))
    repeated = list(simulator.stream(iter(stationary)))
    cine = simulator.render_cine(stationary, progress=False)

    assert not np.array_equal(stream[0].intensity, stream[1].intensity)
    assert not np.array_equal(stream[1].intensity, stream[2].intensity)
    for streamed, again, filmed in zip(stream, repeated, cine):
        np.testing.assert_array_equal(streamed.intensity, again.intensity)
        np.testing.assert_array_equal(streamed.intensity, filmed.intensity)
        np.testing.assert_array_equal(streamed.image, filmed.image)
    for _ in range(2):
        np.testing.assert_array_equal(simulator.render_frame(pose=Pose.ap()).intensity, stream[0].intensity)


def test_render_frame_after_stream_replaces_only_the_first_file(tmp_path, make_simulator):
    simulator = make_simulator(tmp_path, "npy")
    list(simulator.stream(iter(poses(3))))
    remaining = {p: p.read_bytes() for p in (tmp_path / "frame_0001.npy", tmp_path / "frame_0002.npy")}

    frame = simulator.render_frame(pose=Pose.ap())

    np.testing.assert_array_equal(np.load(tmp_path / "frame_0000.npy"), frame.image)
    assert all(p.read_bytes() == content for p, content in remaining.items())
