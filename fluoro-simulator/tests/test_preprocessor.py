# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for VolumePreprocessor functionality."""

import json

import numpy as np
import pytest
from fluorosim import HuToMuMapping, PreprocessedVolume, PreprocessingSettings, VolumePreprocessor


class TestVolumePreprocessor:
    """Test suite for VolumePreprocessor class."""

    def test_from_numpy_creates_preprocessor(self, sample_hu_volume, sample_spacing):
        """Test that from_numpy creates a valid preprocessor."""
        preprocessor = VolumePreprocessor.from_numpy(
            sample_hu_volume, spacing_zyx_mm=sample_spacing
        )
        assert preprocessor is not None

    def test_preprocess_returns_volume(self, sample_hu_volume, sample_spacing):
        """Test that preprocess() returns a PreprocessedVolume."""
        preprocessor = VolumePreprocessor.from_numpy(
            sample_hu_volume, spacing_zyx_mm=sample_spacing
        )
        volume = preprocessor.preprocess()

        assert isinstance(volume, PreprocessedVolume)
        assert volume.mu_volume is not None
        assert volume.mu_volume.shape == sample_hu_volume.shape

    def test_preprocess_converts_hu_to_mu(self, sample_hu_volume, sample_spacing):
        """Test that HU values are converted to attenuation coefficients."""
        preprocessor = VolumePreprocessor.from_numpy(
            sample_hu_volume, spacing_zyx_mm=sample_spacing
        )
        volume = preprocessor.preprocess()

        # Air (-1000 HU) should have ~0 attenuation
        # Bone (+1000 HU) should have higher attenuation
        # Our volume has -900 (near air) and +800 (bone-like)

        # Check that we have a range of values
        assert volume.mu_volume.min() < volume.mu_volume.max()

        # Check dtype is float32
        assert volume.mu_volume.dtype == np.float32

        water = 0.020590
        bone = 0.3148 * 1.92 / 10
        assert float(volume.mu_volume[0, 0, 0]) == pytest.approx(water * 0.1, rel=1e-4)
        assert float(volume.mu_volume[8, 16, 16]) == pytest.approx(
            water + 800.0 * (bone - water) / 1500.0,
            rel=1e-4,
        )

    def test_metadata_preserved(self, sample_hu_volume, sample_spacing):
        """Test that spacing metadata is preserved."""
        preprocessor = VolumePreprocessor.from_numpy(
            sample_hu_volume, spacing_zyx_mm=sample_spacing
        )
        volume = preprocessor.preprocess()

        assert volume.metadata.spacing_zyx_mm == sample_spacing
        assert volume.metadata.calibration is not None
        assert volume.metadata.calibration["scheme"] == "two_anchor_piecewise_linear_v1"

    def test_legacy_mapping_records_provenance(self, sample_hu_volume, sample_spacing):
        """Explicit legacy processing records the old mapping parameters."""
        with pytest.warns(FutureWarning):
            volume = VolumePreprocessor.from_numpy(
                sample_hu_volume,
                spacing_zyx_mm=sample_spacing,
                settings=PreprocessingSettings(hu_to_mu=HuToMuMapping()),
            ).preprocess()
        assert volume.metadata.calibration is not None
        assert volume.metadata.calibration["scheme"] == "legacy_minmax"


class TestPreprocessedVolume:
    """Test suite for PreprocessedVolume class."""

    def test_save_and_load_roundtrip(
        self, sample_hu_volume, sample_spacing, temp_cache_dir
    ):
        """Test that save/load preserves volume data."""
        preprocessor = VolumePreprocessor.from_numpy(
            sample_hu_volume, spacing_zyx_mm=sample_spacing
        )
        original = preprocessor.preprocess()

        # Save
        cache_path = temp_cache_dir / "test_volume"
        original.save(cache_path)

        # Load
        loaded = PreprocessedVolume.load(cache_path)

        # Verify
        assert loaded.mu_volume.shape == original.mu_volume.shape
        np.testing.assert_array_almost_equal(loaded.mu_volume, original.mu_volume)
        assert loaded.metadata.spacing_zyx_mm == original.metadata.spacing_zyx_mm
        assert loaded.metadata.calibration == original.metadata.calibration

    def test_load_nonexistent_raises(self, temp_cache_dir):
        """Test that loading a nonexistent volume raises an error."""
        with pytest.raises((FileNotFoundError, ValueError)):
            PreprocessedVolume.load(temp_cache_dir / "nonexistent")

    def test_legacy_metadata_without_calibration_requires_reprocessing(self, temp_cache_dir):
        """Caches without calibration provenance are invalidated."""
        legacy = {
            "shape_zyx": [1, 2, 3],
            "spacing_zyx_mm": [1.0, 1.0, 1.0],
            "origin_xyz_mm": None,
            "hu_range": [-1000.0, 1000.0],
            "mu_range": [0.0, 0.02],
            "source": "legacy",
        }
        cache_path = temp_cache_dir / "legacy_volume"
        cache_path.mkdir()
        np.save(cache_path / "mu_volume.npy", np.zeros((1, 2, 3), dtype=np.float32))
        (cache_path / "metadata.json").write_text(
            json.dumps(legacy), encoding="utf-8"
        )

        with pytest.raises(ValueError, match="calibration provenance.*reprocess"):
            PreprocessedVolume.load(cache_path)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_2d_volume_rejected(self):
        """Test that 2D arrays are rejected."""
        arr_2d = np.full((8, 8), 100, dtype=np.float32)
        with pytest.raises(ValueError, match="Expected 3D volume"):
            VolumePreprocessor.from_numpy(arr_2d, spacing_zyx_mm=(1.0, 1.0, 1.0))

    def test_1d_volume_rejected(self):
        """Test that 1D arrays are rejected."""
        arr_1d = np.full((8,), 100, dtype=np.float32)
        with pytest.raises(ValueError, match="Expected 3D volume"):
            VolumePreprocessor.from_numpy(arr_1d, spacing_zyx_mm=(1.0, 1.0, 1.0))

    def test_wrong_dtype_converted(self, sample_spacing):
        """Test that non-float32 arrays are handled."""
        hu_int = np.full((8, 8, 8), 100, dtype=np.int16)
        preprocessor = VolumePreprocessor.from_numpy(
            hu_int, spacing_zyx_mm=sample_spacing
        )
        # Should work - conversion should happen internally
        volume = preprocessor.preprocess()
        assert volume.mu_volume.dtype == np.float32

    def test_extreme_hu_values(self, sample_spacing):
        """Test handling of extreme HU values."""
        hu = np.full((8, 8, 8), -3000.0, dtype=np.float32)  # Very low HU
        preprocessor = VolumePreprocessor.from_numpy(hu, spacing_zyx_mm=sample_spacing)
        volume = preprocessor.preprocess()
        # Should complete without error
        assert volume.mu_volume is not None
