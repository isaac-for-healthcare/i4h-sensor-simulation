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

"""CT Preprocessing Example.

This script demonstrates how to preprocess CT volumes for the fluoroscopy simulator.
It shows three input methods: DICOM, NIfTI, and synthetic data.

The preprocessing pipeline converts Hounsfield Units (HU) to linear attenuation
coefficients (μ) which are required for X-ray simulation.

Run with:
    cd fluoro-simulator
    python examples/preprocess_ct.py

    # Or with a specific input:
    python examples/preprocess_ct.py --dicom ~/ct_data/dicom_folder
    python examples/preprocess_ct.py --nifti ~/ct_data/volume.nii.gz
    python examples/preprocess_ct.py --synthetic
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from fluorosim import PreprocessedVolume, VolumePreprocessor

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.environ.get("FLUOROSIM_OUTPUT_DIR", str(SCRIPT_DIR / "output"))).expanduser()
DEFAULT_CACHE = Path(os.environ.get("FLUOROSIM_CACHE_DIR", str(OUTPUT_DIR / "cache"))).expanduser()


def preprocess_dicom(dicom_path: Path, output_dir: Path) -> PreprocessedVolume:
    """Preprocess a DICOM series."""
    print(f"\n[DICOM] Loading from: {dicom_path}")

    preprocessor = VolumePreprocessor.from_dicom(dicom_path)

    print(f"  Volume shape (Z, Y, X): {preprocessor.shape}")
    print(f"  Voxel spacing (mm):     {preprocessor.spacing_zyx_mm}")
    print(f"  HU range:               [{preprocessor.hu_range[0]:.1f}, {preprocessor.hu_range[1]:.1f}]")

    print("\n[DICOM] Converting HU → μ...")
    volume = preprocessor.preprocess(output_dir=output_dir)

    return volume


def preprocess_nifti(nifti_path: Path, output_dir: Path) -> PreprocessedVolume:
    """Preprocess a NIfTI file."""
    print(f"\n[NIfTI] Loading from: {nifti_path}")

    preprocessor = VolumePreprocessor.from_nifti(nifti_path)

    print(f"  Volume shape (Z, Y, X): {preprocessor.shape}")
    print(f"  Voxel spacing (mm):     {preprocessor.spacing_zyx_mm}")
    print(f"  HU range:               [{preprocessor.hu_range[0]:.1f}, {preprocessor.hu_range[1]:.1f}]")

    print("\n[NIfTI] Converting HU → μ...")
    volume = preprocessor.preprocess(output_dir=output_dir)

    return volume


def preprocess_synthetic(output_dir: Path) -> PreprocessedVolume:
    """Create and preprocess a synthetic test volume."""
    import numpy as np

    print("\n[Synthetic] Creating test volume...")

    # Create a volume with a sphere (bone) in air
    shape = (128, 256, 256)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center = np.array(shape) / 2
    dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)

    # HU values: air=-1000, soft tissue=40, bone=1000
    hu_volume = np.full(shape, -900.0, dtype=np.float32)  # Air background
    hu_volume[dist < 60] = 40.0   # Soft tissue shell
    hu_volume[dist < 40] = 800.0  # Bone core

    print(f"  Volume shape (Z, Y, X): {shape}")
    print("  Voxel spacing (mm):     (1.0, 0.5, 0.5)")
    print(f"  HU range:               [{hu_volume.min():.1f}, {hu_volume.max():.1f}]")

    preprocessor = VolumePreprocessor.from_numpy(
        hu_volume,
        spacing_zyx_mm=(1.0, 0.5, 0.5)
    )

    print("\n[Synthetic] Converting HU → μ...")
    volume = preprocessor.preprocess(output_dir=output_dir)

    return volume


def verify_volume(volume: PreprocessedVolume) -> None:
    """Verify the preprocessed volume."""
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    mu = volume.mu_volume

    print(f"  Output shape:    {mu.shape}")
    print(f"  Output dtype:    {mu.dtype}")
    print(f"  μ range:         [{mu.min():.6f}, {mu.max():.6f}] mm⁻¹")
    print(f"  Spacing (mm):    {volume.spacing_zyx_mm}")

    # Sanity checks
    checks_passed = True

    if mu.min() < 0:
        print("  ⚠ Warning: Negative μ values detected")
        checks_passed = False

    if mu.max() > 0.1:
        print("  ⚠ Warning: μ values seem high (>0.1 mm⁻¹)")
        checks_passed = False

    if mu.max() < 0.001:
        print("  ⚠ Warning: μ values seem low (<0.001 mm⁻¹)")
        checks_passed = False

    if checks_passed:
        print("  ✓ All checks passed!")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="CT Preprocessing Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dicom",
        type=Path,
        help="Path to DICOM series directory",
    )
    parser.add_argument(
        "--nifti",
        type=Path,
        help="Path to NIfTI file (.nii or .nii.gz)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic test volume",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"Output directory for cached volume (default: {DEFAULT_CACHE})",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CT Preprocessing Pipeline")
    print("=" * 60)

    # Determine input source
    if args.dicom:
        if not args.dicom.exists():
            print(f"Error: DICOM path not found: {args.dicom}")
            return
        volume = preprocess_dicom(args.dicom, args.output)

    elif args.nifti:
        if not args.nifti.exists():
            print(f"Error: NIfTI path not found: {args.nifti}")
            return
        volume = preprocess_nifti(args.nifti, args.output)

    elif args.synthetic:
        volume = preprocess_synthetic(args.output)

    else:
        # Default: try to find data or use synthetic
        print("\nNo input specified. Using synthetic test volume.")
        print("  Use --dicom, --nifti, or --synthetic to specify input.")
        volume = preprocess_synthetic(args.output)

    # Verify the output
    verify_volume(volume)

    print(f"Preprocessed volume saved to: {args.output}")
    print("\nTo use this volume with the simulator:")
    print(f'  volume = PreprocessedVolume.load("{args.output}")')
    print("  simulator = FluoroSimulator(volume)")


if __name__ == "__main__":
    main()
