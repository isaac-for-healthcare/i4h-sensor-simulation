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

"""Non-GPU physics and structural tests for HU-to-μ calibration."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from fluorosim import HuToMuMapping, PreprocessingSettings, VolumePreprocessor
from fluorosim.calibration import HuToMuCalibration, hu_to_mu


def test_water_anchor() -> None:
    """Water at 0 HU resolves to the NIST 60 keV anchor."""
    result = hu_to_mu(np.array([0.0]), HuToMuCalibration())
    assert float(result[0]) == pytest.approx(0.020590, rel=1e-6)


def test_bone_anchor() -> None:
    """The bone CT anchor resolves to the unrounded NIST-derived value."""
    result = hu_to_mu(np.array([1500.0]), HuToMuCalibration())
    assert float(result[0]) == pytest.approx(0.3148 * 1.92 / 10, rel=1e-6)


def test_air() -> None:
    """Air maps exactly to zero attenuation."""
    assert float(hu_to_mu(np.array([-1000.0]), HuToMuCalibration())[0]) == 0.0


def test_subair_clamped() -> None:
    """Below-air scanner padding is clamped rather than becoming negative μ."""
    result = hu_to_mu(np.linspace(-1024.0, 3071.0, 2048), HuToMuCalibration())
    assert result[0] == 0.0
    assert np.all(result >= 0.0)


def test_blood_out_of_sample() -> None:
    """Validate the two-anchor approximation against independent NIST blood data."""
    # Blood is deliberately not an anchor; this validates the approximation.
    result = hu_to_mu(np.array([50.0]), HuToMuCalibration())
    assert float(result[0]) == pytest.approx(0.021804, rel=0.02)


@pytest.mark.parametrize("energy", [40.0, 50.0, 60.0, 80.0, 100.0])
def test_mapping_is_monotone(energy: float) -> None:
    """Each tabulated monochromatic mapping is non-decreasing in HU."""
    result = hu_to_mu(np.linspace(-1024.0, 3071.0, 10000), HuToMuCalibration(energy))
    assert np.all(np.diff(result) >= 0.0)


def test_continuity_at_water() -> None:
    """The production mapping has no jump at water."""
    calibration = HuToMuCalibration()
    epsilon = 1e-6
    result = hu_to_mu(np.array([-epsilon, epsilon]), calibration)
    assert abs(float(result[0]) - float(result[1])) < 1e-9


def test_above_anchor_extrapolates_linearly() -> None:
    """Dense materials do not plateau at the cortical-bone anchor."""
    calibration = HuToMuCalibration()
    expected = calibration.mu_water_mm_inv + 3000.0 * (
        calibration.mu_bone_mm_inv - calibration.mu_water_mm_inv
    ) / calibration.hu_bone_anchor
    assert float(hu_to_mu(np.array([3000.0]), calibration)[0]) == pytest.approx(expected)


def test_validation_and_overrides() -> None:
    """Table validation rejects unknown energies while paired overrides bypass it."""
    with pytest.raises(ValueError, match=r"40\.0, 50\.0, 60\.0, 80\.0, 100\.0"):
        HuToMuCalibration(65.0)
    with pytest.raises(ValueError, match="greater than zero"):
        HuToMuCalibration(hu_bone_anchor=0.0)
    assert HuToMuCalibration(65.0, mu_water_override_mm_inv=0.02, mu_bone_override_mm_inv=0.06)


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_non_finite_or_non_positive_inputs_are_rejected(value: float) -> None:
    """Physical inputs must be finite and positive."""
    with pytest.raises(ValueError, match="reference_energy_kev"):
        HuToMuCalibration(
            reference_energy_kev=value,
            mu_water_override_mm_inv=0.02,
            mu_bone_override_mm_inv=0.06,
        )
    with pytest.raises(ValueError, match="hu_bone_anchor"):
        HuToMuCalibration(hu_bone_anchor=value)
    with pytest.raises(ValueError, match="mu_water_override_mm_inv"):
        HuToMuCalibration(mu_water_override_mm_inv=value)
    with pytest.raises(ValueError, match="mu_bone_override_mm_inv"):
        HuToMuCalibration(mu_bone_override_mm_inv=value)


def test_serialization_roundtrip() -> None:
    """Calibration fields and resolved anchors round-trip through provenance data."""
    calibration = HuToMuCalibration(80.0, 1200.0)
    assert HuToMuCalibration.from_dict(calibration.to_dict()) == calibration


def test_serialization_warns_on_informational_anchor_mismatch() -> None:
    """Defining fields win when stored resolved values are stale or edited."""
    calibration = HuToMuCalibration()
    payload = calibration.to_dict()
    for stale_value in (1.0, "not-a-number"):
        payload["mu_water_mm_inv"] = stale_value
        with pytest.warns(RuntimeWarning, match="disagrees with the resolved value"):
            restored = HuToMuCalibration.from_dict(payload)
        assert restored == calibration


def test_units_oracle_300_mm_water() -> None:
    """The default full chain produces the closed-form water transmission."""
    hu = np.zeros((1, 300, 1), dtype=np.float32)
    volume = VolumePreprocessor.from_numpy(
        hu, spacing_zyx_mm=(1.0, 1.0, 1.0), settings=PreprocessingSettings()
    ).preprocess()
    transmission = math.exp(-float(np.sum(volume.mu_volume[0, :, 0])))
    assert transmission == pytest.approx(2.077e-3, rel=1e-3)


def test_legacy_mapping_is_bit_identical_and_warns() -> None:
    """An explicit legacy request keeps the original formula and signals migration."""
    hu = np.array([[[-1024.0, -1000.0, 0.0, 3000.0, 3071.0]]], dtype=np.float32)
    mapping = HuToMuMapping()
    with pytest.warns(FutureWarning, match="HuToMuCalibration"):
        result = VolumePreprocessor.from_numpy(
            hu, settings=PreprocessingSettings(hu_to_mu=mapping)
        ).preprocess().mu_volume
    clipped = np.clip(np.clip(hu, -1024.0, 3071.0), mapping.hu_min, mapping.hu_max)
    expected = mapping.mu_min + (clipped - mapping.hu_min) / (
        mapping.hu_max - mapping.hu_min + 1e-12
    ) * (mapping.mu_max - mapping.mu_min)
    np.testing.assert_array_equal(result, expected.astype(np.float32))


def test_default_is_calibrated_and_warning_free() -> None:
    """The public preprocessing default selects the physical, warning-free path."""
    assert isinstance(PreprocessingSettings().hu_to_mu, HuToMuCalibration)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        VolumePreprocessor.from_numpy(np.zeros((1, 1, 1), dtype=np.float32)).preprocess()
