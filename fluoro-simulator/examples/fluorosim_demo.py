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

"""Fluoroscopy Simulator Demo.

This script demonstrates the complete fluorosim workflow:
1. Load CT volume (DICOM, NIfTI, or cached μ-volume)
2. Initialize the GPU-accelerated simulator
3. Render a LAO/RAO C-arm sweep with accurate performance metrics

Features:
- GPU warmup for accurate FPS measurement (~150 FPS on RTX A6000)
- Automatic cache detection and fallback to synthetic volume
- Configurable geometry, realism, and output settings

Run with:
    cd fluoro-simulator
    python examples/fluorosim_demo.py

To clear cache and reload from source:
    rm -rf /tmp/fluoro_cache && python examples/fluorosim_demo.py
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from fluorosim import (
    CarmGeometry,
    FluoroSimulator,
    OutputSettings,
    Pose,
    PreprocessedVolume,
    RealismSettings,
    SimulatorConfig,
    VolumePreprocessor,
)

# ============================================================================
# Configuration - Update these paths to your local data
# ============================================================================

# Path to DICOM CT data (e.g., from Kaggle Multiphase CT Angiography dataset)
# Download: kaggle datasets download -d adamhuan/multiphase-ct-anigography-2-datasets
DICOM_CT_PATH = Path("~/ct_data/excellent/excellent/0").expanduser()

# Alternative: NIfTI path (e.g., ImageCAS dataset)
NIFTI_CT_PATH = Path("~/ct_data/imagecas/1.img.nii.gz").expanduser()

_SCRIPT_DIR = Path(__file__).resolve().parent.parent

# Output directory
OUTPUT_DIR = Path(os.environ.get("FLUOROSIM_OUTPUT_DIR", str(_SCRIPT_DIR / "output"))).expanduser()

# Cache directory for preprocessed volume
CACHE_DIR = Path(os.environ.get("FLUOROSIM_CACHE_DIR", str(OUTPUT_DIR / "cache"))).expanduser()

# Alternative: pre-generated mu_volume if it exists
CACHED_MU_VOLUME = CACHE_DIR / "mu_volume.npy"


def main():
    print("=" * 70)
    print("Fluoroscopy Simulator Demo")
    print("=" * 70)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Load or preprocess the volume
    # =========================================================================
    print("\n[Step 1] Loading CT Volume...")

    # Check if we have a cached preprocessed volume
    if (CACHE_DIR / "mu_volume.npy").exists():
        print(f"  Loading cached volume from: {CACHE_DIR}")
        volume = PreprocessedVolume.load(CACHE_DIR)
    elif DICOM_CT_PATH.exists():
        print(f"  Loading DICOM from: {DICOM_CT_PATH}")
        preprocessor = VolumePreprocessor.from_dicom(DICOM_CT_PATH)
        print(f"  Shape: {preprocessor.shape}")
        print(f"  Spacing: {preprocessor.spacing_zyx_mm} mm")
        print(f"  HU range: {preprocessor.hu_range}")

        volume = preprocessor.preprocess(output_dir=CACHE_DIR)
    elif NIFTI_CT_PATH.exists():
        print(f"  Loading NIfTI from: {NIFTI_CT_PATH}")
        preprocessor = VolumePreprocessor.from_nifti(NIFTI_CT_PATH)
        print(f"  Shape: {preprocessor.shape}")
        print(f"  Spacing: {preprocessor.spacing_zyx_mm} mm")
        print(f"  HU range: {preprocessor.hu_range}")

        volume = preprocessor.preprocess(output_dir=CACHE_DIR)
    else:
        print("\n  ERROR: CT data not found at:")
        print(f"    DICOM: {DICOM_CT_PATH}")
        print(f"    NIfTI: {NIFTI_CT_PATH}")
        print("  Please update the paths in this script.")
        print("\n  Alternatively, run with a synthetic volume:")
        print("    python -m ...run_imagecas --synthetic")

        # Create synthetic volume for demo
        print("\n  Creating synthetic sphere volume for demo...")
        import numpy as np
        shape = (128, 256, 256)
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        center = np.array(shape) / 2
        dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)

        # Sphere with bone density in air
        hu_volume = np.where(dist < 50, 800.0, -900.0).astype(np.float32)

        preprocessor = VolumePreprocessor.from_numpy(
            hu_volume,
            spacing_zyx_mm=(1.0, 0.5, 0.5)
        )
        volume = preprocessor.preprocess(output_dir=CACHE_DIR)

    print("\n  Preprocessed volume:")
    print(f"    {volume}")

    # =========================================================================
    # Step 2: Create simulator with configuration
    # =========================================================================
    print("\n[Step 2] Initializing Simulator...")

    config = SimulatorConfig(
        geometry=CarmGeometry(
            detector_width_px=512,
            detector_height_px=512,
            pixel_spacing_mm=0.5,
            source_to_detector_mm=1020.0,
            source_to_isocenter_mm=510.0,
        ),
        realism=RealismSettings(
            enabled=False,  # Disable noise for clean images
            # gaussian_sigma=0.015,
            # blur_sigma_px=0.3,
        ),
        output=OutputSettings(
            save_to_disk=True,
            output_dir=str(OUTPUT_DIR),
            format="png",
        ),
    )

    simulator = FluoroSimulator(volume, config)
    print(f"  {simulator}")
    print(f"  Backend: {type(simulator.renderer).__name__}")

    # Warmup: run 2 dummy frames to JIT-compile shaders
    print("  Warming up GPU (2 frames)...")
    _ = simulator.render_frame(rotation=(0, 0, 0), translation=(0, 0, 0))
    _ = simulator.render_frame(rotation=(0, 0, 0), translation=(0, 0, 0))
    simulator._frame_times.clear()  # Reset timing stats after warmup

    # =========================================================================
    # Step 3: Render frames
    # =========================================================================
    print("\n[Step 3] Rendering Frames...")

    # Create rotation sequence (LAO/RAO sweep)
    num_frames = 20
    poses = []
    for i in range(num_frames):
        # Rotate around Y axis (LAO/RAO): -30° to +30°
        angle_y = math.radians(-30 + (60 * i / (num_frames - 1)))
        # Small cranial/caudal tilt
        angle_x = math.radians(5 * math.sin(2 * math.pi * i / num_frames))

        poses.append(Pose(
            rotation=(angle_x, angle_y, 0.0),
            translation=(0.0, 0.0, 0.0),
        ))

    print(f"  Rendering {len(poses)} frames (LAO/RAO sweep)...")

    # Render cine sequence
    cine = simulator.render_cine(poses, fps=15.0, progress=True)

    # =========================================================================
    # Step 4: Save and report
    # =========================================================================
    print("\n[Step 4] Saving Results...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = cine.save_all(OUTPUT_DIR / "frames", format="png")

    print(f"  Saved {len(saved_paths)} frames to: {OUTPUT_DIR / 'frames'}")

    # Performance metrics
    metrics = simulator.get_metrics()
    print("\n[Performance Metrics]")
    print(f"  Average FPS: {metrics.fps:.1f}")
    print(f"  Frame time: {1000/metrics.fps:.1f} ms" if metrics.fps > 0 else "  Frame time: N/A")
    print(f"  Jitter (std): {metrics.jitter_ms:.2f} ms")
    if metrics.gpu_memory_mb:
        print(f"  GPU Memory: {metrics.gpu_memory_mb:.1f} MB")

    # Summary
    print("\n" + "=" * 70)
    print("Done! Output files:")
    print(f"  Frames: {OUTPUT_DIR / 'frames'}")
    print(f"  Cache:  {CACHE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
