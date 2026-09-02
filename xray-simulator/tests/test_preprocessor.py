# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import pytest
from xray_simulator import HuToMuMapping, PreprocessedVolume, PreprocessingSettings, VolumePreprocessor


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

    def test_preprocess_applies_configured_ramp(self, sample_hu_volume, sample_spacing):
        """Test that μ values follow the configured piecewise ramp."""
        settings = PreprocessingSettings(
            hu_to_mu=HuToMuMapping(hu_min=-1000.0, hu_max=1000.0, mu_min=0.0, mu_max=0.02)
        )
        preprocessor = VolumePreprocessor.from_numpy(
            sample_hu_volume, spacing_zyx_mm=sample_spacing, settings=settings
        )
        volume = preprocessor.preprocess()

        # Background is -900 HU, bone cube is +800 HU on a -1000..1000 → 0..0.02 ramp.
        assert volume.mu_volume.min() == pytest.approx(0.001, rel=1e-5)
        assert volume.mu_volume.max() == pytest.approx(0.018, rel=1e-5)

    def test_preprocess_clamps_outside_ramp(self, sample_spacing):
        """Test that HU beyond the ramp saturates instead of extrapolating."""
        hu = np.array([[[-4000.0, 4000.0]]], dtype=np.float32)
        settings = PreprocessingSettings(
            clip_hu=False,
            hu_to_mu=HuToMuMapping(hu_min=-1000.0, hu_max=1000.0, mu_min=0.0, mu_max=0.02),
        )
        preprocessor = VolumePreprocessor.from_numpy(
            hu, spacing_zyx_mm=sample_spacing, settings=settings
        )
        mu = preprocessor.preprocess().mu_volume

        np.testing.assert_allclose(mu.ravel(), [0.0, 0.02], rtol=1e-5)

    def test_with_hu_to_mu_rebuilds_without_reloading(self, sample_hu_volume, sample_spacing):
        """Test that window/level can be swapped on an already-loaded volume."""
        base = VolumePreprocessor.from_numpy(
            sample_hu_volume, spacing_zyx_mm=sample_spacing
        )
        narrow = base.with_hu_to_mu(
            HuToMuMapping.from_window_level(window_center=0.0, window_width=400.0)
        )

        assert base.settings.hu_to_mu != narrow.settings.hu_to_mu
        assert narrow.shape == base.shape
        # A narrower window saturates the bone cube, widening the μ spread.
        base_mu = base.preprocess().mu_volume
        narrow_mu = narrow.preprocess().mu_volume
        assert np.ptp(narrow_mu) > np.ptp(base_mu)

    def test_metadata_records_mapping(self, sample_hu_volume, sample_spacing):
        """Test that the mapping used for conversion is recorded in metadata."""
        mapping = HuToMuMapping.from_window_level(window_center=100.0, window_width=800.0)
        preprocessor = VolumePreprocessor.from_numpy(
            sample_hu_volume,
            spacing_zyx_mm=sample_spacing,
            settings=PreprocessingSettings(hu_to_mu=mapping),
        )
        volume = preprocessor.preprocess()

        assert volume.metadata.hu_to_mu == mapping.to_dict()

    def test_metadata_preserved(self, sample_hu_volume, sample_spacing):
        """Test that spacing metadata is preserved."""
        preprocessor = VolumePreprocessor.from_numpy(
            sample_hu_volume, spacing_zyx_mm=sample_spacing
        )
        volume = preprocessor.preprocess()

        assert volume.metadata.spacing_zyx_mm == sample_spacing


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
        assert loaded.metadata.hu_to_mu == original.metadata.hu_to_mu

    def test_load_nonexistent_raises(self, temp_cache_dir):
        """Test that loading a nonexistent volume raises an error."""
        with pytest.raises((FileNotFoundError, ValueError)):
            PreprocessedVolume.load(temp_cache_dir / "nonexistent")


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
