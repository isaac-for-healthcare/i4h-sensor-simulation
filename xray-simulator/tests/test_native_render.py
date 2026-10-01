# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DICOM/artifact parity and physical rendering invariance on native grids."""

import os

import numpy as np
import pytest
from xray_simulator import HuToMuMapping, PreprocessingSettings, VolumePreprocessor
from xray_simulator import scan_volume as scan

from .test_scan_volume import write_dicom


@pytest.mark.parametrize("preset", ["linear", "interventional"])
@pytest.mark.parametrize("resample", [False, True])
def test_dicom_artifact_mu_parity(tmp_path, preset, resample):
    write_dicom(tmp_path / "dicom", angle=0.4)
    conversion = scan.Conversion(
        array_axes="ijk", origin="source_center", spacing_ijk_mm=(0.8, 0.8, 1.0) if resample else None
    )
    recipe = scan.export_ct(tmp_path / "dicom", tmp_path / "artifact", options=conversion)
    settings = PreprocessingSettings(hu_to_mu=HuToMuMapping.preset(preset))
    direct = VolumePreprocessor.from_dicom(tmp_path / "dicom", conversion=conversion, settings=settings).preprocess()
    saved = VolumePreprocessor.from_artifact(recipe, settings=settings).preprocess()
    np.testing.assert_array_equal(direct.mu_volume, saved.mu_volume)
    np.testing.assert_array_equal(direct.metadata.voxel_to_lps_mm, saved.metadata.voxel_to_lps_mm)
    # Public C-arm centering must use the same affine as the GPU renderer.
    from xray_simulator import xray_simulator

    simulator = object.__new__(xray_simulator)
    simulator._volume = direct
    center = np.asarray(direct.metadata.voxel_to_lps_mm) @ np.r_[(np.array(direct.shape[::-1]) - 1) / 2, 1]
    np.testing.assert_allclose(simulator.volume_center_xyz_mm, center[:3])
    if os.environ.get("I4H_TEST_GPU") == "1":
        np.testing.assert_array_equal(
            render(direct.mu_volume, direct.metadata.voxel_to_lps_mm),
            render(saved.mu_volume, saved.metadata.voxel_to_lps_mm),
        )


def render(data, affine):
    from xray_simulator.rendering.diffdrr_slang_renderer import SlangDiffDRRConfig, SlangDiffDRRRenderer

    affine = np.asarray(affine)
    cfg = SlangDiffDRRConfig(
        det_height_px=48,
        det_width_px=48,
        pixel_spacing_mm=1.0,
        device_type="vulkan",
        normalize=False,
        invert=False,
        step_mm=0.2,
    )
    renderer = SlangDiffDRRRenderer(
        data, tuple(np.linalg.norm(affine[:3, :3], axis=0)[::-1]), cfg=cfg, voxel_to_world_mm=affine
    )
    return renderer.render((0.12, 0.21, -0.15), (0.0, 0.0, 0.0))


@pytest.mark.skipif(os.environ.get("I4H_TEST_GPU") != "1", reason="set I4H_TEST_GPU=1 for Vulkan rendering")
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
