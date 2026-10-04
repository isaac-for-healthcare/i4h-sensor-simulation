# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Masked metrics with explicit signal domains and reproducible SSIM/MI scales.

Arrays must already represent the same subject and projection on the same grid.
This module does not register, resize, invert, or normalize input images.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, sobel
from scipy.stats import wasserstein_distance

DOMAINS = ("display", "attenuation", "attenuation_proxy")


def _image(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2 or values.dtype.kind not in "buif" or not np.isfinite(values).all():
        raise ValueError("Images must be finite, real, numeric 2D arrays")
    return values.astype(np.float64)


def attenuation_from_intensity(
    intensity: np.ndarray, i0: float, *, transmission_floor: float = 1e-8
) -> np.ndarray:
    """Compute dimensionless A = -ln(I/I0) from pre-display intensity.

Zero transmission is floored explicitly. Values above I0 remain negative in A
(possible with noise/scatter); they are not silently clipped. Do not apply this
conversion to windowed PNGs or vendor-processed DICOM values of unknown scale.
"""
    image = _image(intensity)
    if not np.isfinite(i0) or i0 <= 0 or np.any(image < 0):
        raise ValueError("i0 must be positive and finite; intensity must be nonnegative")
    if not np.isfinite(transmission_floor) or not 0 < transmission_floor < 1:
        raise ValueError("transmission_floor must lie strictly between 0 and 1")
    transmission = image / i0
    if not np.isfinite(transmission).all():
        raise ValueError("Intensity / i0 must remain finite")
    return -np.log(np.maximum(transmission, transmission_floor))


def _ncc(first: np.ndarray, second: np.ndarray) -> float | None:
    first, second = first - first.mean(), second - second.mean()
    norm = np.linalg.norm(first) * np.linalg.norm(second)
    if norm == 0:
        return None
    return float(np.clip(np.dot(first, second) / norm, -1.0, 1.0))


def _gradient(image: np.ndarray) -> np.ndarray:
    return np.hypot(sobel(image, axis=0, mode="reflect"), sobel(image, axis=1, mode="reflect"))


def _scores(reference, rendered, mask, gradient_mask, ssim_mask, data_range, histogram_range, bins):
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise ImportError("Install xray-simulator with the [validation] extra for SSIM") from exc

    a, b = reference[mask], rendered[mask]
    low, high = histogram_range
    # Preserve outlier mass in the boundary bins; report how much was clipped.
    counts, _, _ = np.histogram2d(
        np.clip(a, low, high), np.clip(b, low, high), bins=bins, range=[histogram_range] * 2
    )
    joint = counts / counts.sum()
    independent = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]
    nz = joint > 0
    mi = float(np.sum(joint[nz] * np.log(joint[nz] / independent[nz])))
    _, ssim_map = structural_similarity(
        reference, rendered, data_range=data_range, gaussian_weights=True,
        sigma=1.5, use_sample_covariance=False, full=True,
    )
    return {
        "ncc": _ncc(a, b),
        "gradient_ncc": _ncc(_gradient(reference)[gradient_mask], _gradient(rendered)[gradient_mask]),
        "mi_nats": mi,
        "ssim": float(ssim_map[ssim_mask].mean()),
        "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
        "wasserstein": float(wasserstein_distance(a, b)),
        "histogram_clipped_reference_fraction": float(np.mean((a < low) | (a > high))),
        "histogram_clipped_rendered_fraction": float(np.mean((b < low) | (b > high))),
    }


def _radial_profiles(reference, rendered, fitted, mask, bins=16):
    rows, columns = np.indices(reference.shape)
    radius = np.hypot(rows - (reference.shape[0] - 1) / 2, columns - (reference.shape[1] - 1) / 2)
    edges = np.linspace(0, float(radius.max()) + np.finfo(float).eps * max(reference.shape), bins + 1)
    result = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = mask & (radius >= low) & (radius < high)
        count = int(selected.sum())
        result.append({
            "radius_px": float((low + high) / 2), "pixels": count,
            "reference": float(reference[selected].mean()) if count else None,
            "rendered": float(rendered[selected].mean()) if count else None,
            "shape_only": float(fitted[selected].mean()) if count and fitted is not None else None,
        })
    return result


def evaluate_pair(
    reference: np.ndarray,
    rendered: np.ndarray,
    mask: np.ndarray,
    *,
    domain: str,
    data_range: float,
    histogram_range: tuple[float, float],
    bins: int = 64,
) -> dict:
    """Compare an aligned pair, preserving its input scale and a fixed anatomy mask.

    ``display`` inputs must be in [0, 1] with data_range=1. The attenuation
    domains additionally fit rendered A to reference A with a nonnegative gain
    and free bias. A proxy domain has arbitrary units, not calibrated attenuation.
    SSIM uses an 11x11 Gaussian window (sigma=1.5); only centers with full support
    inside the mask are averaged. Sobel magnitude uses full 3x3 mask support.

    Constant-signal NCC is None. An unidentifiable constant-render affine fit is
    skipped. The caller must establish subject, pose, geometry, polarity and units.
    """
    reference, rendered = _image(reference), _image(rendered)
    mask = np.asarray(mask)
    if reference.shape != rendered.shape or mask.shape != reference.shape:
        raise ValueError("Reference, rendered image and mask must have identical shapes; no automatic resizing")
    if mask.dtype.kind not in "buif" or not np.isin(mask, [0, 1]).all():
        raise ValueError("Mask must contain only boolean or numeric 0/1 values")
    mask = mask.astype(bool)
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS}")
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be positive and finite")
    if len(histogram_range) != 2 or not np.isfinite(histogram_range).all() or histogram_range[0] >= histogram_range[1]:
        raise ValueError("histogram_range must be a finite (low, high) pair with low < high")
    if isinstance(bins, bool) or not isinstance(bins, int) or not 2 <= bins <= 1024:
        raise ValueError("bins must be an integer between 2 and 1024")
    if domain == "display":
        if data_range != 1 or histogram_range[0] != 0 or histogram_range[1] != 1:
            raise ValueError("Display comparisons require data_range=1 and histogram_range=[0, 1]")
        if any(np.any((image < 0) | (image > 1)) for image in (reference, rendered)):
            raise ValueError("Display inputs must already be scaled to [0, 1]")
    # Square footprints are essential: a cross-shaped erosion would leak excluded
    # pixels into the diagonals of Sobel and Gaussian windows.
    gradient_mask = binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    ssim_mask = binary_erosion(mask, structure=np.ones((11, 11)), border_value=0)
    if min(reference.shape) < 11 or mask.sum() < 2 or gradient_mask.sum() < 2 or not ssim_mask.any():
        raise ValueError("Mask must contain at least one complete 11x11 SSIM window and two gradient centers")

    args = (mask, gradient_mask, ssim_mask, data_range, histogram_range, bins)
    raw = _scores(reference, rendered, *args)
    fitted, fit, shape_scores = None, None, None
    warnings = []
    if domain != "display":
        x, y = rendered[mask], reference[mask]
        xc, yc = x - x.mean(), y - y.mean()
        variance = float(np.dot(xc, xc))
        if variance == 0:
            warnings.append("Constant rendered attenuation: affine gain is not identifiable; shape-only fit skipped")
        else:
            gain = max(0.0, float(np.dot(xc, yc) / variance))
            bias = float(y.mean() - gain * x.mean())
            fitted = gain * rendered + bias
            fit = {"gain": gain, "bias": bias}
            shape_scores = _scores(reference, fitted, *args)
            if gain == 0:
                warnings.append("Affine gain reached zero; inspect polarity and pairing")
    if raw["ncc"] is None or raw["gradient_ncc"] is None:
        warnings.append("Constant image or gradient: the corresponding NCC is undefined (null)")
    return {
        "domain": domain,
        "units": {"display": "normalized grayscale", "attenuation": "dimensionless line integral",
                  "attenuation_proxy": "arbitrary proxy units"}[domain],
        "settings": {"data_range": float(data_range), "histogram_range": list(histogram_range), "bins": bins,
                     "ssim": "Gaussian 11x11, sigma=1.5, population covariance",
                     "gradient": "Sobel magnitude, 3x3 support"},
        "mask_pixels": int(mask.sum()), "gradient_mask_pixels": int(gradient_mask.sum()),
        "ssim_mask_pixels": int(ssim_mask.sum()),
        "as_configured": raw, "shape_only": shape_scores, "affine_fit": fit,
        "radial_profiles": _radial_profiles(reference, rendered, fitted, mask) if domain != "display" else None,
        "warnings": warnings,
    }
