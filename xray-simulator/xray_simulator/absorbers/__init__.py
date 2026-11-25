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

"""Extra absorbers composited onto a rendered frame along the renderer's own rays.

The interface (:class:`DetectorRays`, :class:`Absorber`, :func:`composite`) needs only numpy.
The catheter integrator needs ``warp-lang`` and is imported on first use.
"""

from typing import Any

from .composite import Absorber, composite
from .rays import DetectorRays

_WARP_EXPORTS = {"CatheterAttenuation", "CatheterMaterial", "PosedCatheter"}


def __getattr__(name: str) -> Any:
    if name in _WARP_EXPORTS:
        from . import catheter

        return getattr(catheter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Absorber", "CatheterAttenuation", "CatheterMaterial", "DetectorRays", "PosedCatheter", "composite"]
