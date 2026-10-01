# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from xray_simulator import HU_TO_MU_PRESETS as PRESETS
from xray_simulator import INTERVENTIONAL, LINEAR, HuToMuMapping, PreprocessingSettings, VolumePreprocessor, hu_to_mu

BONE_HU = 900.0
IMPLANT_HU = 6000.0
MUSCLE_HU = 40.0


def _mu(mapping, hu: float | np.ndarray) -> np.ndarray:
    return hu_to_mu(np.asarray(hu, dtype=np.float32), mapping)


def _previous_ramp(hu: np.ndarray) -> np.ndarray:
    """The hardcoded conversion patient twins used before named curves existed."""
    return (np.clip(hu, -1000.0, 3000.0) + 1000.0) / 4000.0 * 0.02


def test_linear_preset_reproduces_the_previous_ramp() -> None:
    hu = np.linspace(-2000.0, 8000.0, 501, dtype=np.float32)

    np.testing.assert_allclose(_mu(LINEAR, hu), _previous_ramp(hu), atol=1e-9)


def test_both_presets_share_the_same_attenuation_ceiling() -> None:
    assert _mu(INTERVENTIONAL, [3000.0])[0] == pytest.approx(_mu(LINEAR, [3000.0])[0])


def test_interventional_separates_implant_density_from_bone() -> None:
    """The single ramp saturates above bone, so an implant and cortical bone look alike."""
    linear_ratio = _mu(LINEAR, [IMPLANT_HU])[0] / _mu(LINEAR, [BONE_HU])[0]
    interventional_ratio = _mu(INTERVENTIONAL, [IMPLANT_HU])[0] / _mu(INTERVENTIONAL, [BONE_HU])[0]

    assert linear_ratio < 2.2
    assert interventional_ratio > 3.5


def test_interventional_suppresses_soft_tissue() -> None:
    assert _mu(INTERVENTIONAL, [MUSCLE_HU])[0] < 0.4 * _mu(LINEAR, [MUSCLE_HU])[0]


def test_air_and_lung_do_not_attenuate() -> None:
    np.testing.assert_allclose(_mu(INTERVENTIONAL, [-1000.0, -800.0, -400.0]), 0.0)


def test_every_preset_is_monotonic_and_non_negative() -> None:
    hu = np.linspace(-2000.0, 10000.0, 2001, dtype=np.float32)

    for name, mapping in PRESETS.items():
        mu = hu_to_mu(hu, mapping)
        assert np.all(np.diff(mu) >= -1e-9), f"{name} is not monotonic"
        assert np.all(mu >= 0.0), f"{name} produced negative attenuation"


def test_values_outside_the_knots_clamp_to_the_end_points() -> None:
    mu = _mu(INTERVENTIONAL, [-5000.0, 20000.0])

    assert mu[0] == pytest.approx(INTERVENTIONAL.mu_knots[0])
    assert mu[1] == pytest.approx(INTERVENTIONAL.mu_knots[-1])


def test_unknown_preset_is_rejected() -> None:
    with pytest.raises(KeyError, match="clinical"):
        PRESETS["clinical"]


def test_default_mapping_is_linear_without_preclipping():
    settings = PreprocessingSettings()
    assert settings.hu_to_mu == LINEAR
    assert settings.clip_hu is False
    assert HuToMuMapping.preset() == LINEAR


def test_interventional_preprocessing_preserves_high_hu():
    hu = np.array([-1500, -300, 100, 300, 900, 3000, 6000, 9000], dtype=np.float32).reshape(2, 2, 2)
    settings = PreprocessingSettings(hu_to_mu=HuToMuMapping.preset("interventional"))
    volume = VolumePreprocessor.from_numpy(hu, settings=settings).preprocess()
    np.testing.assert_array_equal(volume.mu_volume.ravel(), np.array(
        [0, 0, 0.0008, 0.0028, 0.009, 0.02, 0.0344, 0.044], dtype=np.float32))
    assert volume.metadata.hu_to_mu == INTERVENTIONAL.to_dict()
    clipped = VolumePreprocessor.from_numpy(hu, settings=PreprocessingSettings(
        hu_to_mu=INTERVENTIONAL, clip_hu=True)).preprocess()
    assert clipped.mu_volume.max() < volume.mu_volume.max()


def test_named_preset_rejects_typo():
    with pytest.raises(ValueError, match="Unknown HU-to-mu preset"):
        HuToMuMapping.preset("typo")
