# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native grids, reproducible artifacts, and DICOM geometry validation."""

import itertools

import numpy as np
import pytest
import yaml
from xray_simulator import scan_volume as scan


def test_array_axes_and_affine_roundtrip(tmp_path):
    values = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    a = np.array([[0.0, -2.0, 0.0, 20.0], [1.0, 0.0, 0.0, -10.0], [0.0, 0.0, 3.0, 40.0], [0.0, 0.0, 0.0, 1.0]])
    for axes in map("".join, itertools.permutations("ijk")):
        perm = ["ijk".index(c) for c in axes]
        p = np.eye(4)
        p[:3, :3] = np.eye(3)[:, perm]
        volume = scan.from_array(values.transpose(perm), a @ p, array_axes=axes)
        np.testing.assert_array_equal(volume.values_kji, values.transpose(2, 1, 0))
        np.testing.assert_allclose(volume.ijk_to_world, a)
        artifact = volume.save(tmp_path / axes)
        loaded = scan.load_artifact(artifact)
        np.testing.assert_array_equal(loaded.values, volume.values)
        np.testing.assert_allclose(loaded.ijk_to_world, a)
    path = tmp_path / "ijk" / "volume.npy"
    changed = np.load(path)
    changed.flat[0] += 1
    np.save(path, changed)
    with pytest.raises(ValueError, match="hash"):
        scan.load_artifact(path.with_suffix(".yaml"))


def test_nifti_preserves_oblique_affine_and_native_values(tmp_path):
    nib = pytest.importorskip("nibabel")
    values = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    a = np.array([[0.8, -0.6, 0.0, 15.0], [0.6, 0.8, 0.0, -20.0], [0.0, 0.0, 2.0, 30.0], [0.0, 0.0, 0.0, 1.0]])
    path = tmp_path / "scan.nii.gz"
    image = nib.Nifti1Image(values, a)
    image.header.set_xyzt_units("mm")
    nib.save(image, path)
    volume = scan.from_nifti(path)
    np.testing.assert_array_equal(volume.values, values)
    np.testing.assert_allclose(volume.ijk_to_world, nib.load(path).affine)
    assert volume.array_axes == "ijk"
    artifact = volume.save(tmp_path / "saved")
    replayed = scan.replay(artifact, path, tmp_path / "replayed")
    assert yaml.safe_load(artifact.read_text()) == yaml.safe_load(replayed.read_text())


def write_dicom(directory, *, angle=0.0, irregular=False):
    sitk = pytest.importorskip("SimpleITK")
    directory.mkdir()
    z, y, x = np.indices((7, 8, 9))
    values = np.where((x - 4) ** 2 + (y - 3) ** 2 + (z - 3) ** 2 < 8, 700, -1000).astype(np.int16)
    image = sitk.GetImageFromArray(values)
    image.SetSpacing((0.7, 1.1, 2.0))
    image.SetOrigin((23.0, -19.0, 41.0))
    d = np.array([[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]])
    image.SetDirection(tuple(d.ravel()))
    series = "2.25.315758790148123413413412341"
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    for k in range(values.shape[0]):
        plane = image[:, :, k]
        position = np.array(image.TransformIndexToPhysicalPoint((0, 0, k)))
        if irregular and k == 3:
            position += d[:, 2] * 0.2
        tags = {
            "0008|0060": "CT",
            "0020|000e": series,
            "0020|000d": "2.25.315758790148123413413412342",
            "0020|0013": str(k + 1),
            "0020|0032": "\\".join(map(str, position)),
            "0020|0037": "\\".join(map(str, np.r_[d[:, 0], d[:, 1]])),
            "0028|1052": "-1024",
            "0028|1053": "1",
        }
        for key, value in tags.items():
            plane.SetMetaData(key, value)
        writer.SetFileName(str(directory / f"{k:03}.dcm"))
        writer.Execute(plane)
    return values, image


@pytest.mark.parametrize("angle", [0.0, 0.4])
def test_dicom_recipe_replays_and_preserves_obliquity(tmp_path, angle):
    values, image = write_dicom(tmp_path / "dicom", angle=angle)
    volume = scan.from_dicom(tmp_path / "dicom")
    np.testing.assert_array_equal(volume.values_kji, values)
    expected = np.diag([-0.001, -0.001, 0.001, 1.0]) @ scan.affine(image)
    np.testing.assert_allclose(volume.ijk_to_world, expected, atol=1e-8)
    for settings in [
        scan.Conversion(),
        scan.Conversion(origin="source_center", array_axes="ijk", spacing_ijk_mm=(1.0, 1.0, 1.0)),
    ]:
        suffix = "resampled" if settings.spacing_ijk_mm else "native"
        artifact = scan.export_ct(tmp_path / "dicom", tmp_path / suffix, options=settings)
        replayed = scan.replay(artifact, tmp_path / "dicom", tmp_path / (suffix + "-replay"))
        assert yaml.safe_load(artifact.read_text()) == yaml.safe_load(replayed.read_text())


def test_irregular_dicom_rejected(tmp_path):
    write_dicom(tmp_path / "dicom", irregular=True)
    with pytest.raises(ValueError, match="Irregular"):
        scan.from_dicom(tmp_path / "dicom")


def test_multiple_series_requires_selection(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    write_dicom(tmp_path / "dicom")
    p = tmp_path / "dicom" / "000.dcm"
    image = sitk.ReadImage(str(p))
    image.SetMetaData("0020|000e", "2.25.999888777")
    w = sitk.ImageFileWriter()
    w.KeepOriginalImageUIDOn()
    w.SetFileName(str(tmp_path / "dicom" / "other.dcm"))
    w.Execute(image)
    with pytest.raises(ValueError, match="Select one series"):
        scan.from_dicom(tmp_path / "dicom")
