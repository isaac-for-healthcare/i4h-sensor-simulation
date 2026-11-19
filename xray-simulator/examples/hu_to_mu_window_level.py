#!/usr/bin/env python3
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

"""Window/level tuning of the HU → μ transfer function.

Sweeps window/level settings on one CT volume and, for each setting, plots the
piecewise-linear HU → μ curve next to the resulting attenuation image. The preview
image is a parallel-beam line integral through the μ volume (Beer-Lambert), which is
enough to judge contrast and brightness without a GPU; the cone-beam Slang renderer
is what produces the final frames.

Run with:
    cd xray-simulator
    python examples/hu_to_mu_window_level.py --synthetic
    python examples/hu_to_mu_window_level.py --nifti ~/ct_data/volume.nii.gz
    python examples/hu_to_mu_window_level.py --nifti ~/ct_data/volume.nii.gz \
        --window-center 100 --window-width 800
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from xray_simulator import HuToMuMapping, VolumePreprocessor, hu_to_mu_curve

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.environ.get("XRAY_SIMULATOR_OUTPUT_DIR", str(SCRIPT_DIR / "output"))).expanduser()

# Starting points for visual tuning against reference fluoroscopy, not spectral calibrations.
DEFAULT_SWEEP: tuple[tuple[str, float, float], ...] = (
    ("Whole HU range", 1000.0, 4000.0),
    ("Soft tissue + vessels", 100.0, 800.0),
    ("Bone emphasis", 800.0, 2000.0),
)


def synthetic_hu_volume() -> tuple[np.ndarray, tuple[float, float, float]]:
    """Build a phantom with air, soft tissue, a contrasted vessel and bone."""
    shape = (96, 192, 192)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    center = np.array(shape) / 2.0
    radial = np.sqrt((y - center[1]) ** 2 + (x - center[2]) ** 2)

    in_body_slab = (z >= 10) & (z < shape[0] - 10)
    spine = (np.abs(y - center[1] - 45) < 12) & (np.abs(x - center[2]) < 12)

    hu = np.full(shape, -1000.0, dtype=np.float32)
    hu[(radial < 70) & in_body_slab] = 40.0  # soft tissue body
    hu[spine & in_body_slab] = 900.0  # bone
    hu[(radial < 8) & in_body_slab] = 350.0  # contrasted vessel
    return hu, (1.0, 0.5, 0.5)


def parallel_beam_preview(mu_volume: np.ndarray, spacing_mm: float) -> np.ndarray:
    """Return a Beer-Lambert transmission image along the volume's first axis."""
    path_length = mu_volume.sum(axis=0) * spacing_mm
    return np.exp(-path_length)


def build_sweep(args: argparse.Namespace) -> list[tuple[str, HuToMuMapping]]:
    """Return the (label, mapping) pairs to compare."""
    if args.window_center is not None or args.window_width is not None:
        defaults = HuToMuMapping()
        center = args.window_center if args.window_center is not None else defaults.window_center
        width = args.window_width if args.window_width is not None else defaults.window_width
        mapping = HuToMuMapping.from_window_level(center, width, mu_max=args.mu_max)
        return [(f"C={center:.0f} W={width:.0f}", mapping)]

    return [
        (f"{label}\nC={center:.0f} W={width:.0f}", HuToMuMapping.from_window_level(center, width, mu_max=args.mu_max))
        for label, center, width in DEFAULT_SWEEP
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare HU → μ window/level settings")
    parser.add_argument("--dicom", type=Path, help="Path to DICOM series directory")
    parser.add_argument("--nifti", type=Path, help="Path to NIfTI file (.nii or .nii.gz)")
    parser.add_argument("--synthetic", action="store_true", help="Use a synthetic phantom")
    parser.add_argument("--window-center", type=float, help="Single level to inspect (HU)")
    parser.add_argument("--window-width", type=float, help="Single window to inspect (HU)")
    parser.add_argument("--mu-max", type=float, default=0.02, help="μ at the top of the ramp (mm⁻¹)")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "hu_to_mu_window_level.png")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.dicom:
        preprocessor = VolumePreprocessor.from_dicom(args.dicom)
    elif args.nifti:
        preprocessor = VolumePreprocessor.from_nifti(args.nifti)
    else:
        hu, spacing = synthetic_hu_volume()
        preprocessor = VolumePreprocessor.from_numpy(hu, spacing_zyx_mm=spacing)

    print(f"Volume shape (Z, Y, X): {preprocessor.shape}")
    print(f"HU range:               [{preprocessor.hu_range[0]:.0f}, {preprocessor.hu_range[1]:.0f}]")

    sweep = build_sweep(args)
    fig, axes = plt.subplots(len(sweep), 2, figsize=(9, 3.6 * len(sweep)), squeeze=False)

    for row, (label, mapping) in enumerate(sweep):
        volume = preprocessor.with_hu_to_mu(mapping).preprocess()
        preview = parallel_beam_preview(volume.mu_volume, volume.spacing_zyx_mm[0])

        hu_samples, mu_samples = hu_to_mu_curve(mapping)
        curve_ax = axes[row][0]
        curve_ax.plot(hu_samples, mu_samples, color="tab:blue")
        curve_ax.plot(mapping.hu_knots, mapping.mu_knots, "o", color="tab:red")
        curve_ax.set_title(label, fontsize=10)
        curve_ax.set_xlabel("HU")
        curve_ax.set_ylabel("μ (mm⁻¹)")
        curve_ax.grid(alpha=0.3)

        image_ax = axes[row][1]
        image_ax.imshow(preview, cmap="gray", vmin=0.0, vmax=1.0)
        image_ax.set_title(
            f"parallel-beam transmission (dense = dark)\nμ ≤ {volume.mu_volume.max():.4f} mm⁻¹",
            fontsize=10,
        )
        image_ax.axis("off")

        print(f"{label.splitlines()[0]:<24} μ range [{volume.mu_volume.min():.6f}, {volume.mu_volume.max():.6f}]")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)
    print(f"\nSaved comparison to: {args.output}")


if __name__ == "__main__":
    main()
