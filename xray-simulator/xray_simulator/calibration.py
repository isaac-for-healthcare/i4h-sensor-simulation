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

"""Physically anchored HU-to-attenuation curve, expressed as ``HuToMuMapping`` control points."""

from __future__ import annotations

from math import isfinite

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
_RHO_WATER_G_CM3 = 1.0
_RHO_CORTICAL_BONE_G_CM3 = 1.92
TABULATED_ENERGIES_KEV = tuple(sorted(_MU_RHO_WATER_CM2_G.keys() & _MU_RHO_CORTICAL_BONE_CM2_G.keys()))


def physical_control_points(
    reference_energy_kev: float = 60.0,
    hu_bone_anchor: float = 1500.0,
    hu_max: float = 3071.0,
    *,
    mu_water_mm_inv: float | None = None,
    mu_bone_mm_inv: float | None = None,
) -> tuple[tuple[float, float], ...]:
    """Control points of the two-anchor physical HU-to-μ curve, in mm⁻¹.

    For HU <= 0, ``μ = μ_water (1 + HU / 1000)``: the algebraic inversion of the HU
    definition with air attenuation approximated as zero (at 60 keV μ_air is about
    2.5e-5 mm⁻¹). For HU > 0 the curve interpolates linearly from water to cortical
    bone at ``hu_bone_anchor`` and continues with the same slope up to ``hu_max``,
    above which ``HuToMuMapping`` clamps. The curve is continuous at water but has a
    slope knot there.

    ``hu_bone_anchor`` encodes an assumption about the *input CT* calibration,
    including scanner spectrum and reconstruction kernel; it is not a material constant.

    Anchors come from the NIST XCOM tables at the monochromatic ``reference_energy_kev``
    (40, 50, 60, 80 or 100 keV) with ICRU-44 water (1.00 g/cm³) and cortical bone
    (1.92 g/cm³). Supplying both ``mu_water_mm_inv`` and ``mu_bone_mm_inv`` replaces the
    table lookup and allows any energy.

    Args:
        reference_energy_kev: Tabulated monochromatic energy in keV.
        hu_bone_anchor: CT value assigned to the cortical-bone anchor.
        hu_max: Last control point; the curve is clamped above it.
        mu_water_mm_inv: Optional water attenuation override in mm⁻¹.
        mu_bone_mm_inv: Optional cortical-bone attenuation override in mm⁻¹.

    Returns:
        ``((HU, μ), ...)`` with strictly increasing HU, ready for ``HuToMuMapping``.

    Raises:
        ValueError: If an input is not finite and positive, the energy is not tabulated
            (without both overrides), or ``hu_max`` does not exceed the bone anchor.
    """
    overrides = mu_water_mm_inv is not None and mu_bone_mm_inv is not None
    for name, value in (
        ("reference_energy_kev", reference_energy_kev),
        ("hu_bone_anchor", hu_bone_anchor),
        ("mu_water_mm_inv", mu_water_mm_inv),
        ("mu_bone_mm_inv", mu_bone_mm_inv),
    ):
        if value is not None and (not isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and greater than zero, got {value!r}")
    if not overrides and reference_energy_kev not in _MU_RHO_WATER_CM2_G:
        energies = ", ".join(str(energy) for energy in TABULATED_ENERGIES_KEV)
        raise ValueError(
            f"reference_energy_kev must be one of {energies} keV unless both attenuation "
            "overrides are supplied"
        )
    if not isfinite(hu_max) or hu_max <= hu_bone_anchor:
        raise ValueError(f"hu_max must be finite and exceed hu_bone_anchor ({hu_bone_anchor}), got {hu_max!r}")

    water = (
        mu_water_mm_inv
        if mu_water_mm_inv is not None
        else _MU_RHO_WATER_CM2_G[reference_energy_kev] * _RHO_WATER_G_CM3 / 10
    )
    bone = (
        mu_bone_mm_inv
        if mu_bone_mm_inv is not None
        else _MU_RHO_CORTICAL_BONE_CM2_G[reference_energy_kev] * _RHO_CORTICAL_BONE_G_CM3 / 10
    )
    slope = (bone - water) / hu_bone_anchor
    return (
        (-1000.0, 0.0),
        (0.0, water),
        (float(hu_bone_anchor), bone),
        (float(hu_max), water + slope * hu_max),
    )
