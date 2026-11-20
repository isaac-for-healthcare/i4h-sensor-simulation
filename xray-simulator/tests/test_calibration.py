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

"""Physics and structural tests for the NIST-anchored ``physical`` HU-to-μ preset."""

from __future__ import annotations

import math

import numpy as np
import pytest
from xray_simulator import (
    HU_TO_MU_PRESETS,
    CarmGeometry,
    DisplaySettings,
    HuToMuMapping,
    PreprocessingSettings,
    RealismSettings,
    SimulatorConfig,
    VolumePreprocessor,
    XrayPhysics,
    hu_to_mu,
    xray_simulator,
)
from xray_simulator.calibration import physical_control_points
from xray_simulator.config import LINEAR, PHYSICAL

MU_WATER_60KEV = 0.020590
MU_BONE_60KEV = 0.3148 * 1.92 / 10


def _mu(hu, mapping: HuToMuMapping = PHYSICAL) -> np.ndarray:
    return hu_to_mu(np.asarray(hu, dtype=np.float32), mapping)


def test_physical_is_a_named_opt_in_preset_and_linear_stays_the_default() -> None:
    assert HU_TO_MU_PRESETS["physical"] is PHYSICAL
    assert HuToMuMapping.preset("physical") is PHYSICAL
    assert PreprocessingSettings().hu_to_mu == LINEAR
    assert PreprocessingSettings.preset("physical").clip_hu is True


def test_water_anchor() -> None:
    """Water at 0 HU resolves to the NIST 60 keV anchor."""
    assert float(_mu([0.0])[0]) == pytest.approx(MU_WATER_60KEV, rel=1e-6)


def test_bone_anchor() -> None:
    """The bone CT anchor resolves to the unrounded NIST-derived value."""
    assert float(_mu([1500.0])[0]) == pytest.approx(MU_BONE_60KEV, rel=1e-6)


def test_air_and_sub_air_are_zero() -> None:
    """Air, and scanner padding below it, never becomes negative attenuation."""
    result = _mu(np.linspace(-2000.0, -1000.0, 50))
    assert np.all(result == 0.0)


def test_soft_tissue_segment_inverts_the_hu_definition() -> None:
    hu = np.linspace(-1000.0, 0.0, 101)
    np.testing.assert_allclose(_mu(hu), MU_WATER_60KEV * (1.0 + hu / 1000.0), rtol=1e-6, atol=1e-9)


def test_blood_out_of_sample() -> None:
    """Blood is deliberately not an anchor; it validates the two-anchor approximation."""
    assert float(_mu([50.0])[0]) == pytest.approx(0.021804, rel=0.02)


@pytest.mark.parametrize("energy", [40.0, 50.0, 60.0, 80.0, 100.0])
def test_mapping_is_monotone(energy: float) -> None:
    mapping = HuToMuMapping(control_points=physical_control_points(energy))
    assert np.all(np.diff(_mu(np.linspace(-1024.0, 3071.0, 10000), mapping)) >= 0.0)


def test_continuity_at_water() -> None:
    result = _mu([-1e-3, 1e-3])
    assert abs(float(result[0]) - float(result[1])) < 1e-4 * MU_WATER_60KEV


def test_above_the_bone_anchor_the_slope_continues_up_to_the_last_knot() -> None:
    slope = (MU_BONE_60KEV - MU_WATER_60KEV) / 1500.0
    assert float(_mu([3000.0])[0]) == pytest.approx(MU_WATER_60KEV + 3000.0 * slope, rel=1e-6)
    assert float(_mu([3071.0])[0]) == float(_mu([8000.0])[0])  # clamped at the last knot


def test_validation_and_overrides() -> None:
    with pytest.raises(ValueError, match=r"40\.0, 50\.0, 60\.0, 80\.0, 100\.0"):
        physical_control_points(65.0)
    with pytest.raises(ValueError, match="hu_bone_anchor"):
        physical_control_points(hu_bone_anchor=0.0)
    with pytest.raises(ValueError, match="hu_max"):
        physical_control_points(hu_bone_anchor=1500.0, hu_max=1500.0)
    points = physical_control_points(65.0, mu_water_mm_inv=0.02, mu_bone_mm_inv=0.06)
    assert points[1] == (0.0, 0.02) and points[2] == (1500.0, 0.06)


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_non_finite_or_non_positive_inputs_are_rejected(value: float) -> None:
    both = {"mu_water_mm_inv": 0.02, "mu_bone_mm_inv": 0.06}
    with pytest.raises(ValueError, match="reference_energy_kev"):
        physical_control_points(value, **both)
    with pytest.raises(ValueError, match="hu_bone_anchor"):
        physical_control_points(hu_bone_anchor=value)
    with pytest.raises(ValueError, match="mu_water_mm_inv"):
        physical_control_points(mu_water_mm_inv=value)
    with pytest.raises(ValueError, match="mu_bone_mm_inv"):
        physical_control_points(mu_bone_mm_inv=value)


def test_mapping_serialisation_round_trip_records_the_anchors() -> None:
    restored = HuToMuMapping.from_dict(PHYSICAL.to_dict())
    assert restored == PHYSICAL
    assert dict(restored.control_points)[0.0] == pytest.approx(MU_WATER_60KEV)


def test_units_oracle_300_mm_water() -> None:
    """The full preprocessing chain gives the closed-form 300 mm water transmission."""
    hu = np.zeros((1, 300, 1), dtype=np.float32)
    volume = VolumePreprocessor.from_numpy(
        hu, spacing_zyx_mm=(1.0, 1.0, 1.0), settings=PreprocessingSettings.preset("physical")
    ).preprocess()
    assert math.exp(-float(np.sum(volume.mu_volume[0, :, 0]))) == pytest.approx(2.077e-3, rel=1e-3)
    assert volume.metadata.hu_to_mu == PHYSICAL.to_dict()


@pytest.mark.gpu
def test_physical_water_slab_renders_absolute_transmission() -> None:
    """Render 300 physical millimetres of water without display remapping."""
    slangpy = pytest.importorskip("slangpy")
    try:
        device = slangpy.create_device(slangpy.DeviceType.cuda)
    except RuntimeError as exc:
        pytest.skip(f"Slang CUDA device unavailable: {exc}")
    del device

    # The default pose looks along +Z. Anisotropic spacing catches voxel/mm mix-ups.
    hu = np.zeros((150, 8, 8), dtype=np.float32)
    volume = VolumePreprocessor.from_numpy(
        hu, spacing_zyx_mm=(2.0, 1.0, 1.0), settings=PreprocessingSettings.preset("physical")
    ).preprocess()
    config = SimulatorConfig(
        geometry=CarmGeometry(detector_width_px=3, detector_height_px=3),
        physics=XrayPhysics(step_mm=0.25, i0=2.0),
        display=DisplaySettings.preset("transmission"),
        realism=RealismSettings(enabled=False),
    )

    frame = xray_simulator(volume, config).render_frame()

    # Odd detector dimensions put [1, 1] on the normal ray. Allow one march step
    # of endpoint error without accepting a whole-millimetre path length error.
    assert frame.image[1, 1] == pytest.approx(2.077e-3, rel=0.006)


def test_linear_default_still_maps_water_to_the_legacy_value() -> None:
    """Documents the difference the preset fixes: the default maps water to 0.005 mm^-1."""
    assert float(_mu([0.0], LINEAR)[0]) == pytest.approx(0.005)
