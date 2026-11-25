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

"""Absorbers that are not in the volume, composited onto a rendered frame.

Transmission is :math:`\\exp(-\\int \\mu \\, ds)`, so attenuators add in the exponent. Anything
that moves every frame -- a device, a deforming vessel -- can therefore be integrated on its own
and multiplied into the transmission the volume already produced, instead of being voxelised
into the volume and re-rendered.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .rays import DetectorRays


@runtime_checkable
class Absorber(Protocol):
    """Anything that can report its own line integral along a projection's rays."""

    def line_integral(self, rays: DetectorRays, *, as_torch: bool = False) -> Any:
        """Return :math:`\\int \\mu \\, ds` per pixel, ``(..., height, width)``, dimensionless.

        Leading dimensions, if any, are batch dimensions and broadcast against the transmission
        being composited onto. With ``as_torch`` the result is a torch tensor on the absorber's
        device; otherwise a numpy array.
        """
        ...


def composite(transmission: Any, absorbers: Iterable[Absorber], rays: DetectorRays) -> Any:
    """Attenuate ``transmission`` by every absorber along ``rays``.

    Args:
        transmission: Rendered transmission, ``(..., height, width)``, numpy or torch. The
            absorbers' results are produced in the same kind and on the same device.
        absorbers: Extra attenuators, integrated independently and summed in the exponent.
        rays: The projection ``transmission`` was rendered with.

    Returns:
        ``transmission * exp(-sum(line integrals))``, or ``transmission`` itself when there are
        no absorbers.
    """
    as_torch = not isinstance(transmission, np.ndarray)
    total = None
    for absorber in absorbers:
        integral = absorber.line_integral(rays, as_torch=as_torch)
        if as_torch:
            integral = integral.to(transmission.device)
        total = integral if total is None else total + integral
    if total is None:
        return transmission
    if as_torch:
        return transmission * (-total).exp()
    return transmission * np.exp(-total)


__all__ = ["Absorber", "composite"]
