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

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RealismConfig:
    """Phase-1 realism knobs for fluoroscopy intensity images.

    This is deliberately minimal and CPU-side (NumPy) so it can be applied to frames after rendering,
    regardless of whether rendering ran on CPU or GPU.

    Conventions:
    - Input is expected to be a non-negative float image (intensity-like), typically I = I0 * exp(-∫μ ds).
    - Output is float32. If normalize_output=True, output is scaled to [0,1] for visualization/saving.
    """

    # Linear intensity scaling (applied before noise)
    gain: float = 1.0
    bias: float = 0.0

    # Noise models (applied in this order)
    poisson_photons: float = 0.0  # if >0, apply Poisson noise assuming this many photons at intensity=1
    gaussian_sigma: float = 0.0  # additive Gaussian noise sigma (in intensity units)

    # Optional blur (requires scipy). sigma in pixels.
    blur_sigma_px: float = 0.0

    # Output normalization
    normalize_output: bool = True
    eps: float = 1e-8

    # Randomness
    seed: int | None = 0


def apply_realism(img: np.ndarray, cfg: RealismConfig = RealismConfig()) -> np.ndarray:
    """Apply Phase-1 realism (scaling/noise/blur) to a single 2D frame."""
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image; got shape={img.shape}")

    out = img.astype(np.float32, copy=True)
    out = out * float(cfg.gain) + float(cfg.bias)
    out = np.clip(out, 0.0, None)

    rng = np.random.default_rng(cfg.seed) if cfg.seed is not None else np.random.default_rng()

    if float(cfg.poisson_photons) > 0.0:
        lam = np.clip(out, 0.0, None) * float(cfg.poisson_photons)
        # Poisson on photon counts, then convert back to intensity units
        out = rng.poisson(lam=lam).astype(np.float32) / float(cfg.poisson_photons)

    if float(cfg.gaussian_sigma) > 0.0:
        out = out + rng.normal(loc=0.0, scale=float(cfg.gaussian_sigma), size=out.shape).astype(np.float32)
        out = np.clip(out, 0.0, None)

    if float(cfg.blur_sigma_px) > 0.0:
        try:
            import scipy.ndimage  # type: ignore

            out = scipy.ndimage.gaussian_filter(out, sigma=float(cfg.blur_sigma_px)).astype(np.float32, copy=False)
        except Exception:
            # Keep it as a soft dependency; blur is optional in Phase-1.
            pass

    if cfg.normalize_output:
        vmin = float(np.min(out))
        vmax = float(np.max(out))
        out = (out - vmin) / (vmax - vmin + float(cfg.eps))

    return out.astype(np.float32, copy=False)
