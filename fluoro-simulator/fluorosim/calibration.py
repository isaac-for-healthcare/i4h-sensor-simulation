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

"""Physically anchored HU-to-linear-attenuation calibration."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import isclose, isfinite
from typing import Any

import numpy as np

# NIST XCOM, Hubbell and Seltzer mass attenuation tables, retrieved 2026-08-12:
# https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html
# https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/bone.html
_MU_RHO_WATER_CM2_G: dict[float, float] = {
    40.0: 0.2683,
    50.0: 0.2269,
    60.0: 0.2059,
    80.0: 0.1837,
    100.0: 0.1707,
}
_MU_RHO_CORTICAL_BONE_CM2_G: dict[float, float] = {
    40.0: 0.6655,
    50.0: 0.4242,
    60.0: 0.3148,
    80.0: 0.2229,
    100.0: 0.1855,
}
_RHO_WATER_G_CM3: float = 1.0
_RHO_CORTICAL_BONE_G_CM3: float = 1.92
_SCHEME = "two_anchor_piecewise_linear_v1"
_PERMITTED_ENERGIES = frozenset(
    _MU_RHO_WATER_CM2_G.keys() & _MU_RHO_CORTICAL_BONE_CM2_G.keys()
)


@dataclass(frozen=True)
class HuToMuCalibration:
    """Physically anchored piecewise-linear HU-to-μ calibration.

    For HU <= 0, μ = μ_water (1 + HU / 1000).  For HU > 0,
    μ = μ_water + HU (μ_bone - μ_water) / hu_bone_anchor, with linear
    extrapolation above the bone anchor.

    The sub-water expression is the algebraic inversion of HU = 1000
    (μ - μ_water) / (μ_water - μ_air) when μ_air is approximated as zero.
    At 60 keV μ_air is approximately 2.5e-5 mm^-1 and is negligible here, so
    this segment is exact under that stated approximation rather than a fitted
    approximation.  The segments meet continuously at water, with a slope knot.
    Preprocessing runs in NumPy outside the Slang autodiff path, so the knot has
    no gradient implication today; it would matter if this mapping moved into an
    autograd graph.

    ``hu_bone_anchor`` encodes an assumption about the *input CT* calibration,
    including scanner spectrum and reconstruction kernel.  It is not a universal
    material constant.

    Anchors use the Hubbell & Seltzer NIST XCOM mass attenuation tables and
    ICRU-44 water and cortical-bone compositions:
    https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html and
    https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/bone.html.

    Attributes:
        reference_energy_kev: Tabulated monochromatic reference energy in keV.
        hu_bone_anchor: CT HU assigned to the cortical-bone anchor.
        mu_water_override_mm_inv: Optional water attenuation override in mm^-1.
        mu_bone_override_mm_inv: Optional cortical-bone attenuation override in
            mm^-1.
    """

    reference_energy_kev: float = 60.0
    hu_bone_anchor: float = 1500.0
    mu_water_override_mm_inv: float | None = None
    mu_bone_override_mm_inv: float | None = None

    def __post_init__(self) -> None:
        """Validate tabulated-energy and anchor inputs."""
        overrides_supplied = (
            self.mu_water_override_mm_inv is not None
            and self.mu_bone_override_mm_inv is not None
        )
        if not isfinite(self.reference_energy_kev) or self.reference_energy_kev <= 0:
            raise ValueError("reference_energy_kev must be finite and positive")
        if not overrides_supplied and self.reference_energy_kev not in _PERMITTED_ENERGIES:
            permitted = ", ".join(str(energy) for energy in sorted(_PERMITTED_ENERGIES))
            raise ValueError(
                "reference_energy_kev must be one of "
                f"{permitted} keV unless both attenuation overrides are supplied"
            )
        if not isfinite(self.hu_bone_anchor) or self.hu_bone_anchor <= 0:
            raise ValueError("hu_bone_anchor must be finite and greater than zero")
        for name, value in (
            ("mu_water_override_mm_inv", self.mu_water_override_mm_inv),
            ("mu_bone_override_mm_inv", self.mu_bone_override_mm_inv),
        ):
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive when supplied")

    @property
    def mu_water_mm_inv(self) -> float:
        """Return the resolved water linear attenuation coefficient in mm^-1."""
        if self.mu_water_override_mm_inv is not None:
            return self.mu_water_override_mm_inv
        return _MU_RHO_WATER_CM2_G[self.reference_energy_kev] * _RHO_WATER_G_CM3 / 10

    @property
    def mu_bone_mm_inv(self) -> float:
        """Return the resolved cortical-bone attenuation coefficient in mm^-1."""
        if self.mu_bone_override_mm_inv is not None:
            return self.mu_bone_override_mm_inv
        return (
            _MU_RHO_CORTICAL_BONE_CM2_G[self.reference_energy_kev]
            * _RHO_CORTICAL_BONE_G_CM3
            / 10
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize fields and resolved anchors for volume provenance.

        Returns:
            Calibration fields, scheme discriminator, and resolved anchors.
        """
        return {
            "scheme": _SCHEME,
            "reference_energy_kev": self.reference_energy_kev,
            "hu_bone_anchor": self.hu_bone_anchor,
            "mu_water_override_mm_inv": self.mu_water_override_mm_inv,
            "mu_bone_override_mm_inv": self.mu_bone_override_mm_inv,
            "mu_water_mm_inv": self.mu_water_mm_inv,
            "mu_bone_mm_inv": self.mu_bone_mm_inv,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HuToMuCalibration":
        """Deserialize a calibration and check its informational anchors.

        Args:
            d: Serialised calibration mapping.

        Returns:
            Calibration reconstructed from its defining fields.

        Raises:
            ValueError: If the scheme or a defining field is invalid.

        Warns:
            RuntimeWarning: If a stored resolved anchor disagrees with the value
                reconstructed from the defining fields.
        """
        if d.get("scheme", _SCHEME) != _SCHEME:
            raise ValueError(f"Unsupported HU-to-μ calibration scheme: {d.get('scheme')!r}")
        calibration = cls(
            reference_energy_kev=float(d["reference_energy_kev"]),
            hu_bone_anchor=float(d["hu_bone_anchor"]),
            mu_water_override_mm_inv=(
                float(d["mu_water_override_mm_inv"])
                if d.get("mu_water_override_mm_inv") is not None
                else None
            ),
            mu_bone_override_mm_inv=(
                float(d["mu_bone_override_mm_inv"])
                if d.get("mu_bone_override_mm_inv") is not None
                else None
            ),
        )
        for key, resolved in (
            ("mu_water_mm_inv", calibration.mu_water_mm_inv),
            ("mu_bone_mm_inv", calibration.mu_bone_mm_inv),
        ):
            stored = d.get(key)
            if stored is None:
                continue
            try:
                matches = isclose(float(stored), resolved, rel_tol=1e-9)
            except (OverflowError, TypeError, ValueError):
                matches = False
            if not matches:
                warnings.warn(
                    f"Stored {key}={stored} disagrees with the resolved value {resolved}; "
                    "using the calibration fields.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return calibration


def hu_to_mu(hu: np.ndarray, calibration: HuToMuCalibration) -> np.ndarray:
    """Convert arbitrary-shaped HU data to non-negative float32 attenuation.

    For HU <= 0 this uses μ = μ_water (1 + HU / 1000), the algebraic inversion
    of HU = 1000 (μ - μ_water) / (μ_water - μ_air) under μ_air ~= 0.  At 60 keV
    μ_air is approximately 2.5e-5 mm^-1 and negligible, making this segment exact
    under that approximation.  For HU > 0 it interpolates from water to cortical
    bone and linearly extrapolates above the bone anchor.  The mapping is C0 at
    water but has a slope knot.  It runs in NumPy outside the Slang autodiff path,
    so that knot has no current gradient implication.

    Args:
        hu: Arbitrary-shaped Hounsfield Unit array.
        calibration: Physical calibration defining water and bone anchors.

    Returns:
        Non-negative linear attenuation coefficients in mm^-1 as float32.
    """
    values = np.asarray(hu, dtype=np.float32)
    water = np.float32(calibration.mu_water_mm_inv)
    bone = np.float32(calibration.mu_bone_mm_inv)
    bone_slope = (bone - water) / np.float32(calibration.hu_bone_anchor)
    result = np.empty_like(values, dtype=np.float32)
    sub_water = values <= np.float32(0.0)
    above_water = np.logical_not(sub_water)

    np.multiply(values, water / np.float32(1000.0), out=result, where=sub_water)
    np.add(result, water, out=result, where=sub_water)
    np.multiply(values, bone_slope, out=result, where=above_water)
    np.add(result, water, out=result, where=above_water)
    np.maximum(result, np.float32(0.0), out=result)
    return result
