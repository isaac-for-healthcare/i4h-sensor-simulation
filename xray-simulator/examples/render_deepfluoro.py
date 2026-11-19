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

"""Export a matched DeepFluoro view using the xvr-data geometry adapter.

Run from xray-simulator: python -m examples.render_deepfluoro --help
The reference export retains stored DICOM values. It is not calibrated attenuation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from xray_simulator import HuToMuMapping, PreprocessingSettings, SimulatorConfig, VolumePreprocessor, xray_simulator
from xray_simulator.config import XrayPhysics
from xray_simulator.validation import attenuation_from_intensity
from xray_simulator.validation.xvr import XvrVolumeFrame, load_xvr_view, load_xvr_volume


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=Path, required=True)
    parser.add_argument("--view", default="000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binning", type=int, default=4, help="Integer factor dividing 1436; 4 gives 359x359")
    parser.add_argument("--step-mm", type=float, default=0.5)
    parser.add_argument("--window-center", type=float, default=200.0)
    parser.add_argument("--window-width", type=float, default=1600.0)
    parser.add_argument("--mu-max", type=float, default=0.05)
    parser.add_argument("--dryrun", action="store_true")
    args = parser.parse_args(argv)
    if not np.isfinite(args.step_mm) or args.step_mm <= 0:
        parser.error("--step-mm must be finite and positive")
    for value in (args.window_center, args.window_width, args.mu_max):
        if not np.isfinite(value):
            parser.error("HU-to-mu parameters must be finite")
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")

    import nibabel as nib

    view = load_xvr_view(args.subject_dir, args.view)
    volume_path = args.subject_dir / "volume.nii.gz"
    image = nib.load(volume_path)
    frame = XvrVolumeFrame.from_affine(image.shape, image.affine)
    camera = view.camera(frame, binning=args.binning)
    # Also verify the reference grid/crop in the preview, before opening a GPU.
    reference = view.load_reference(binning=args.binning)
    mapping = HuToMuMapping.from_window_level(args.window_center, args.window_width, args.mu_max)
    settings = PreprocessingSettings(hu_to_mu=mapping)
    config = SimulatorConfig.for_appearance(
        "fluoro", geometry=camera.geometry, physics=XrayPhysics(step_mm=args.step_mm)
    ).with_output(keep_intensity=True)
    manifest = {
        "dataset": "deepfluoro",
        "subject": args.subject_dir.name,
        "view": args.view,
        "camera": camera.to_dict(),
        "config": asdict(config),
        "preprocessing": asdict(settings),
        "output": str(args.output),
        "reference_signal": "Stored MONOCHROME2 values; 50-pixel border crop then block mean",
        "reference_preview": "Min/max scaled for viewing only, not preset validation",
        "render_signal": "Noise-free monochromatic intensity and dimensionless attenuation",
        "mapping_note": "Engineering baseline, not fitted scanner calibration",
    }
    if args.dryrun:
        print(json.dumps(manifest, indent=2))
        return

    values, frame = load_xvr_volume(volume_path)
    volume = VolumePreprocessor(
        values, frame.spacing_zyx_mm, origin_xyz_mm=frame.origin_xyz_mm, settings=settings, source=str(volume_path)
    ).preprocess()
    simulator = xray_simulator(volume, config)
    rendered = simulator.render_frame(pose=camera.pose)
    attenuation = attenuation_from_intensity(rendered.intensity, rendered.i0)
    if not np.isfinite(attenuation).all() or np.ptp(attenuation) == 0:
        raise RuntimeError("Nonfinite or empty render; inspect the geometry")
    manifest["source_sha256"] = {str(p): sha256(p) for p in (volume_path, view.calibration_path, view.reference_path)}
    code_root = Path(__file__).resolve().parents[1] / "xray_simulator"
    manifest["code_sha256"] = {
        str(p.relative_to(code_root)): sha256(p)
        for p in (
            code_root / "validation/xvr.py",
            code_root / "geometry.py",
            code_root / "config.py",
            code_root / "simulator.py",
            code_root / "rendering/diffdrr_slang_renderer.py",
            code_root / "rendering/diffdrr_slang.slang",
        )
    }
    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output / "reference_stored.npy", reference)
    np.save(args.output / "rendered_intensity.npy", rendered.intensity)
    np.save(args.output / "rendered_attenuation.npy", attenuation)
    np.save(args.output / "rendered_display.npy", rendered.image)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    from PIL import Image

    reference_preview = (reference - reference.min()) / max(float(np.ptp(reference)), 1.0)
    for name, array in (("reference_preview", reference_preview), ("rendered_display", rendered.image)):
        Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)).save(args.output / f"{name}.png")
    print(f"Matched view exported to {args.output}")


if __name__ == "__main__":
    main()
