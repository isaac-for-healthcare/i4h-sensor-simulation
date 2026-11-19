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

"""Mapping rendered X-ray intensity to a displayable image.

The renderer produces intensity ``I = i0 · exp(-∫μ ds)``, so dense anatomy carries *less*
signal than air. Turning that into pixels involves two independent choices that are easy to
conflate:

* **Polarity** — whether dense structures are drawn dark (``"fluoro"``, the raw physical
  ordering, matching a live fluoroscopy monitor) or bright (``"diagnostic"``, the inverted
  film-radiograph look).
* **Scaling** — how intensity maps to the ``[0, 1]`` display range. ``"log"`` (the default)
  is linear in the line integral ``∫μ ds``, mirroring a logarithmic detector chain; a torso
  transmits only a few percent of the beam, so mapping transmission directly leaves the
  image crushed against black. ``"transmission"`` is that literal ``I / i0``.
  ``"window"`` stretches a chosen transmission interval for contrast. ``"per_frame"``
  rescales each frame by its own min and max, which maximises contrast but makes brightness
  depend on what happens to be in the field of view, so a cine sequence flickers.

Keeping the two separate means a preset can say "fluoroscopy polarity with stable physical
scaling" without implying anything about contrast stretching.
"""

from __future__ import annotations

import numpy as np

from .config import DisplaySettings

_EPS = 1e-8


def transmission(intensity: np.ndarray, i0: float) -> np.ndarray:
    """Convert rendered intensity to transmission ``I / i0`` in ``[0, 1]``.

    Args:
        intensity: Rendered intensity image, non-negative.
        i0: Unattenuated beam intensity used for the render.

    Returns:
        Fraction of the beam reaching each detector pixel, 1.0 through air and approaching
        0.0 through fully attenuating material.

    Raises:
        ValueError: If i0 is not positive.
    """
    if i0 <= 0.0:
        raise ValueError(f"i0 must be positive, got {i0}")
    return np.clip(np.asarray(intensity, dtype=np.float32) / float(i0), 0.0, 1.0)


def _line_integral(transmitted: np.ndarray) -> np.ndarray:
    """Recover ``∫μ ds`` from transmission, floored so fully blocked rays stay finite."""
    return -np.log(np.clip(transmitted, _EPS, 1.0))


def calibrate_display(
    intensity: np.ndarray,
    settings: DisplaySettings | None = None,
    i0: float = 1.0,
    percentiles: tuple[float, float] = (1.0, 99.0),
) -> DisplaySettings:
    """Fit the log window to a representative frame, once, for reuse on every frame.

    A fixed window has to suit the patient size and μ scaling at hand: too wide and the image
    is flat, too narrow and anatomy clips to black. Measuring it from one frame and then
    holding it fixed keeps the mapping frame-independent, so brightness stays stable while
    the C-arm moves — unlike per-frame normalization, which re-fits on every frame.

    Args:
        intensity: Representative rendered intensity image, e.g. the first frame of a run.
        settings: Settings to update. Defaults to the fluoroscopy preset.
        i0: Unattenuated beam intensity used for the render.
        percentiles: Low and high percentiles of the line integral to map to the bright and
            dark ends. Trimming the extremes keeps a small dense object, such as a catheter
            tip, from stretching the whole window.

    Returns:
        A copy of ``settings`` with ``scaling="log"`` and a fitted ``log_window``.

    Raises:
        ValueError: If the percentiles are not increasing and within [0, 100], or the frame
            has no attenuation range to fit.
    """
    low_pct, high_pct = percentiles
    if not 0.0 <= low_pct < high_pct <= 100.0:
        raise ValueError(f"percentiles must satisfy 0 <= low < high <= 100, got {percentiles}")

    settings = settings or DisplaySettings()
    line_integral = _line_integral(transmission(intensity, i0))
    low, high = (float(v) for v in np.percentile(line_integral, [low_pct, high_pct]))

    if high - low < 1e-6:
        raise ValueError(
            "cannot calibrate: the frame has no attenuation range between the requested "
            f"percentiles (both {low:.6f}). Is the volume empty or the pose off-target?"
        )

    return DisplaySettings(
        polarity=settings.polarity,
        scaling="log",
        log_window=(max(0.0, low), high),
        window=settings.window,
        gamma=settings.gamma,
    )


def apply_display(
    intensity: np.ndarray,
    settings: DisplaySettings | None = None,
    i0: float = 1.0,
) -> np.ndarray:
    """Map a rendered intensity image to display values in ``[0, 1]``.

    Args:
        intensity: 2D rendered intensity image (``i0 · exp(-∫μ ds)``).
        settings: Polarity, scaling and gamma to apply. Defaults to the fluoroscopy preset.
        i0: Unattenuated beam intensity used for the render.

    Returns:
        float32 image in ``[0, 1]``, ready to save or stream.

    Raises:
        ValueError: If the image is not 2D or i0 is not positive.
    """
    if intensity.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {intensity.shape}")

    settings = settings or DisplaySettings()
    scaled = _scale(transmission(intensity, i0), settings)

    if settings.polarity == "diagnostic":
        scaled = 1.0 - scaled

    if settings.gamma != 1.0:
        scaled = np.power(np.clip(scaled, 0.0, 1.0), 1.0 / settings.gamma)

    return np.clip(scaled, 0.0, 1.0).astype(np.float32, copy=False)


def _scale(values: np.ndarray, settings: DisplaySettings) -> np.ndarray:
    """Map transmission onto the display range according to the scaling mode.

    Returns a signal where 1.0 is an unattenuated ray, so polarity can flip it afterwards.
    """
    if settings.scaling == "log":
        low, high = settings.log_window
        line_integral = _line_integral(values)
        return 1.0 - np.clip((line_integral - low) / (high - low), 0.0, 1.0)

    if settings.scaling == "transmission":
        return values

    if settings.scaling == "window":
        lo, hi = settings.window
        return np.clip((values - lo) / (hi - lo), 0.0, 1.0)

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    return (values - vmin) / (vmax - vmin + _EPS)
