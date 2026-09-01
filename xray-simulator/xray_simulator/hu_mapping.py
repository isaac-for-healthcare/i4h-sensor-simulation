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

"""Evaluation of the piecewise-linear HU → μ transfer function.

The curve itself is described by :class:`~xray_simulator.config.HuToMuMapping`; this
module applies it to arrays and samples it for plotting.
"""

from __future__ import annotations

import numpy as np

from .config import HuToMuMapping


def hu_to_mu(hu: np.ndarray, mapping: HuToMuMapping | None = None) -> np.ndarray:
    """Map Hounsfield Units to linear attenuation coefficients (μ, mm⁻¹).

    Applies the piecewise-linear curve through ``mapping.points``, holding the first and
    last control values constant outside the knot range.

    Args:
        hu: Array of HU values, any shape.
        mapping: Transfer function to apply. Defaults to the standard two-point ramp.

    Returns:
        μ values as float32 with the same shape as ``hu``.
    """
    mapping = mapping or HuToMuMapping()
    mu = np.interp(hu, mapping.hu_knots, mapping.mu_knots)
    return np.asarray(mu, dtype=np.float32)


def hu_to_mu_curve(
    mapping: HuToMuMapping | None = None,
    hu_range: tuple[float, float] | None = None,
    num: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the transfer function for plotting or inspection.

    Args:
        mapping: Transfer function to sample. Defaults to the standard two-point ramp.
        hu_range: HU interval to sample over. Defaults to the ramp padded by 20% of its
            width on each side, so the clamped tails are visible.
        num: Number of samples.

    Returns:
        Tuple of (HU samples, μ samples).
    """
    mapping = mapping or HuToMuMapping()
    if hu_range is None:
        pad = 0.2 * mapping.window_width
        hu_range = (mapping.hu_min - pad, mapping.hu_max + pad)

    hu = np.linspace(hu_range[0], hu_range[1], num, dtype=np.float32)
    return hu, hu_to_mu(hu, mapping)
