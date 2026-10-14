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

"""Frame and CineSequence file output, CPU only."""

import sys

import numpy as np
import pytest
from xray_simulator import CineSequence, Frame, Pose


@pytest.fixture
def frame():
    image = np.linspace(0.0, 1.0, 12 * 16, dtype=np.float32).reshape(12, 16)
    return Frame(image=image, pose=Pose.ap())


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


@pytest.mark.parametrize("fmt", ["png", "npy", "npz"])
def test_save_all_writes_every_documented_format(tmp_path, frame, fmt):
    if fmt == "png":
        pytest.importorskip("PIL")
    cine = CineSequence(frames=[frame, Frame(image=frame.image, pose=frame.pose, frame_idx=1)])

    paths = cine.save_all(tmp_path, format=fmt)

    assert paths == [tmp_path / f"frame_0000.{fmt}", tmp_path / f"frame_0001.{fmt}"]
    assert sorted(tmp_path.iterdir()) == paths
