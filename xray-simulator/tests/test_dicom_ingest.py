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

"""DICOM series loading applies RescaleSlope and RescaleIntercept exactly once."""

import numpy as np
import pytest
from xray_simulator import VolumePreprocessor
from xray_simulator.ct.dicom_ingest import load_dicom_series_hu

pydicom = pytest.importorskip("pydicom")
pytest.importorskip("SimpleITK")


def _write(path, dataset):
    try:
        pydicom.dcmwrite(path, dataset, enforce_file_format=True)
    except TypeError:  # pydicom < 3
        dataset.is_little_endian = True
        dataset.is_implicit_VR = False
        pydicom.dcmwrite(path, dataset, write_like_original=False)


def _write_ct_series(directory, stored_zyx, slope, intercept):
    """Write one CT slice per file with the given stored pixel values and rescale tags."""
    uid = pydicom.uid
    study_uid, series_uid = uid.generate_uid(), uid.generate_uid()
    for z, stored in enumerate(stored_zyx):
        meta = pydicom.dataset.FileMetaDataset()
        meta.MediaStorageSOPClassUID = uid.CTImageStorage
        meta.MediaStorageSOPInstanceUID = uid.generate_uid()
        meta.TransferSyntaxUID = uid.ExplicitVRLittleEndian

        ds = pydicom.Dataset()
        ds.file_meta = meta
        ds.SOPClassUID = uid.CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.Modality = "CT"
        ds.PatientID = "synthetic"
        ds.InstanceNumber = z + 1
        ds.ImagePositionPatient = [0.0, 0.0, float(z) * 2.0]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.PixelSpacing = [0.5, 0.5]
        ds.SliceThickness = 2.0
        ds.Rows, ds.Columns = stored.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleSlope = slope
        ds.RescaleIntercept = intercept
        ds.PixelData = stored.astype("<i2").tobytes()
        _write(directory / f"slice_{z:03d}.dcm", ds)


@pytest.mark.parametrize("slope, intercept", [(1, -1024), (2, -1000), (1, 0)])
def test_series_hu_is_stored_value_rescaled_once(tmp_path, slope, intercept):
    rng = np.random.default_rng(0)
    stored = rng.integers(0, 4096, size=(4, 8, 6)).astype(np.int16)
    stored[0, 0, 0], stored[-1, -1, -1] = 0, 4095
    _write_ct_series(tmp_path, stored, slope, intercept)

    ct = load_dicom_series_hu(tmp_path)

    expected = stored.astype(np.float32) * slope + intercept
    np.testing.assert_allclose(ct.hu_zyx, expected, atol=0.5)


def test_preprocessor_hu_range_matches_the_rescaled_input(tmp_path):
    stored = np.zeros((3, 8, 8), dtype=np.int16)
    stored[1, 2:6, 2:6] = 4095
    _write_ct_series(tmp_path, stored, 1, -1024)

    preprocessor = VolumePreprocessor.from_dicom(tmp_path)

    assert preprocessor.hu_range == pytest.approx((-1024.0, 3071.0), abs=0.5)
