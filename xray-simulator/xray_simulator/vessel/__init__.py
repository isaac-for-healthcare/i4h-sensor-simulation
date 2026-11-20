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

"""Optional vessel contrast: carve the lumen out of a CT and composite it back per frame.

Off unless a caller builds a :class:`VesselContrast`; the volume renderer itself is unchanged.
The carve, centerline and contrast transport need only numpy and scipy. The chord integrator and
:class:`VesselContrast` need ``warp-lang`` and are imported on first use.
"""

from typing import Any

from .carve import DONOR_QUANTILES, CarveResult, carve_lumen, carve_residual, node_contrast_mu_per_mm, tube_occupancy
from .centerline import SMOOTHING_PASSES, VesselTree, smooth_centerline
from .contrast import (
    DISPERSION_MM2_PER_S,
    HU_PER_MG_IODINE_PER_ML,
    INLET_VELOCITY_MM_PER_S,
    STOCK_CONCENTRATION_MG_PER_ML,
    ContrastAgent,
    ContrastTransport,
    HuToMu,
    Injection,
)

_WARP_EXPORTS = {
    "DSA_REFERENCE_MG_PER_ML": "compositor",
    "PosedVessel": "attenuation",
    "VesselAttenuation": "attenuation",
    "VesselContrast": "compositor",
    "VesselState": "attenuation",
    "subtract": "compositor",
}


def __getattr__(name: str) -> Any:
    module = _WARP_EXPORTS.get(name)
    if module is not None:
        from importlib import import_module

        return getattr(import_module(f".{module}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DISPERSION_MM2_PER_S",
    "DONOR_QUANTILES",
    "DSA_REFERENCE_MG_PER_ML",
    "HU_PER_MG_IODINE_PER_ML",
    "INLET_VELOCITY_MM_PER_S",
    "SMOOTHING_PASSES",
    "STOCK_CONCENTRATION_MG_PER_ML",
    "CarveResult",
    "ContrastAgent",
    "ContrastTransport",
    "HuToMu",
    "Injection",
    "PosedVessel",
    "VesselAttenuation",
    "VesselContrast",
    "VesselState",
    "VesselTree",
    "carve_lumen",
    "carve_residual",
    "node_contrast_mu_per_mm",
    "smooth_centerline",
    "subtract",
    "tube_occupancy",
]
