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

"""DICOM/artifact parity and physical rendering invariance on native grids."""

import os

import numpy as np
import pytest
from xray_simulator import PreprocessingSettings, VolumePreprocessor
from xray_simulator import scan_volume as scan
from xray_simulator import volume_center_xyz_mm

from .test_scan_volume import write_dicom


@pytest.mark.parametrize("preset", ["linear", "interventional"])
@pytest.mark.parametrize("resample", [False, True])
def test_dicom_artifact_mu_parity(tmp_path, preset, resample):
    write_dicom(tmp_path / "dicom", angle=0.4)
    conversion = scan.Conversion(
        array_axes="ijk", origin="source_center", spacing_ijk_mm=(0.8, 0.8, 1.0) if resample else None
    )
    recipe = scan.export_ct(tmp_path / "dicom", tmp_path / "artifact", options=conversion)
    settings = PreprocessingSettings.preset(preset)
    direct = VolumePreprocessor.from_dicom(tmp_path / "dicom", conversion=conversion, settings=settings).preprocess()
    saved = VolumePreprocessor.from_artifact(recipe, settings=settings).preprocess()
    np.testing.assert_array_equal(direct.mu_volume, saved.mu_volume)
    np.testing.assert_array_equal(direct.metadata.voxel_to_lps_mm, saved.metadata.voxel_to_lps_mm)
    # Public C-arm centering must use the same affine as the GPU renderer.
    affine = np.asarray(direct.metadata.voxel_to_lps_mm)
    center = affine @ np.r_[(np.array(direct.shape[::-1]) - 1) / 2, 1]
    np.testing.assert_allclose(
        volume_center_xyz_mm(direct.shape, direct.spacing_zyx_mm, voxel_to_world_mm=affine), center[:3]
    )
    if os.environ.get("I4H_TEST_GPU") == "1":
        np.testing.assert_array_equal(
            render(direct.mu_volume, direct.metadata.voxel_to_lps_mm),
            render(saved.mu_volume, saved.metadata.voxel_to_lps_mm),
        )


def make_renderer(data, affine, det_px=48, step_mm=0.2, device="vulkan"):
    from xray_simulator.rendering.diffdrr_slang_renderer import SlangDiffDRRConfig, SlangDiffDRRRenderer

    affine = np.asarray(affine)
    cfg = SlangDiffDRRConfig(
        det_height_px=det_px,
        det_width_px=det_px,
        pixel_spacing_mm=1.0,
        device_type=device,
        normalize=False,
        invert=False,
        step_mm=step_mm,
    )
    return SlangDiffDRRRenderer(
        data, tuple(np.linalg.norm(affine[:3, :3], axis=0)[::-1]), cfg=cfg, voxel_to_world_mm=affine
    )


def render(data, affine):
    return make_renderer(data, affine).render((0.12, 0.21, -0.15), (0.0, 0.0, 0.0))


GPU = pytest.mark.skipif(os.environ.get("I4H_TEST_GPU") != "1", reason="set I4H_TEST_GPU=1 for Vulkan rendering")


def oblique_affine():
    ax, ay = 0.5, 0.3
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    affine = np.eye(4)
    affine[:3, :3] = rx @ ry @ np.diag([1.2, 1.7, 2.1])
    affine[:3, 3] = [30.0, -25.0, 40.0]
    return affine


@GPU
def test_uniform_volume_centre_pixel_matches_closed_form_chord():
    """exp(-mu * chord) with the chord from an independent numpy ray/parallelepiped intersection.

    At zero pose the centre ray runs along world +z through the centre of the volume's bounding
    box, so the expected line integral depends only on the stated affine convention, not on the
    shader. A transposed rotation or a half-voxel offset changes the chord.
    """
    shape, mu, step = (30, 34, 38), 0.02, 0.1
    affine = oblique_affine()
    centre = (affine @ np.r_[(np.array(shape[::-1]) - 1) / 2, 1])[:3]
    inverse = np.linalg.inv(affine)
    origin_index = (inverse @ np.r_[centre, 1])[:3]
    direction_index = inverse[:3, :3] @ [0.0, 0.0, 1.0]
    lo, hi = -0.5, np.array(shape[::-1]) - 0.5
    t = np.sort(np.stack([(lo - origin_index) / direction_index, (hi - origin_index) / direction_index]), axis=0)
    chord = t[1].min() - t[0].max()
    assert chord > 20.0
    image = make_renderer(np.full(shape, mu, np.float32), affine, det_px=49, step_mm=step).render()
    assert -np.log(image[24, 24]) == pytest.approx(mu * chord, abs=3 * mu * step)


@GPU
def test_pose_gradients_match_finite_differences_on_oblique_volume():
    shape = (17, 19, 23)
    z, y, x = np.indices(shape)
    values = np.zeros(shape, np.float32)
    values[(x - 11) ** 2 + (y - 9) ** 2 + (z - 8) ** 2 < 30] = 0.02
    values[3:9, 4:8, 14:19] = 0.03
    try:  # the differentiable backward kernel does not compile for Vulkan
        renderer = make_renderer(values, oblique_affine(), det_px=32, step_mm=0.3, device="cuda")
    except Exception as exc:
        pytest.skip(f"CUDA device unavailable: {exc}")
    rotation, translation = np.array([0.12, 0.21, -0.15]), np.array([2.0, -1.5, 1.0])
    _, grads = renderer.render_with_gradients(rotation, translation)
    for name, base, h in [("rotation", rotation, 2e-3), ("translation", translation, 0.2)]:
        for axis in range(3):
            plus, minus = base.copy(), base.copy()
            plus[axis] += h
            minus[axis] -= h
            args = (lambda v: (v, translation)) if name == "rotation" else (lambda v: (rotation, v))
            numeric = (renderer.render(*args(plus)).sum() - renderer.render(*args(minus)).sum()) / (2 * h)
            assert grads[name][axis] == pytest.approx(numeric, rel=0.1, abs=2e-3), (name, axis)


@GPU
def test_projection_invariant_to_array_reindexing():
    shape = (17, 19, 23)
    z, y, x = np.indices(shape)
    values = np.zeros(shape, np.float32)
    values[(x - 8) ** 2 + (y - 11) ** 2 + (z - 7) ** 2 < 25] = 0.018
    values[3:9, 4:8, 14:19] = 0.03
    angle = 0.3
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([1.2, 1.7, 2.1])
    affine[:3, 3] = [30.0, -25.0, 40.0]
    reference = render(values, affine)
    flip = np.eye(4)
    flip[0, 0] = -1
    flip[0, 3] = shape[2] - 1
    perm = np.eye(4)
    perm[:3, :3] = np.eye(3)[:, [2, 1, 0]]
    for data, transformed in [(values[:, :, ::-1], affine @ flip), (values.transpose(2, 1, 0), affine @ perm)]:
        np.testing.assert_allclose(render(data, transformed), reference, rtol=2e-5, atol=2e-6)
