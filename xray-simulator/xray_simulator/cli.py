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

"""Launch preprocessing or rendering from a validated JSON/YAML preset."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import PreprocessingSettings, SimulatorConfig
from .geometry import VIEWS, view_frame_warning, view_rotation
from .preprocessor import VolumePreprocessor
from .simulator import Pose, xray_simulator
from .volume import PreprocessedVolume, VolumeMetadata


def finite(value: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise argparse.ArgumentTypeError("must be finite")
    return result


def positive(value: str) -> float:
    result = finite(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xray-simulator", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {}
    for name, help_text in (
        ("preprocess", "Cache an attenuation volume (CPU)"),
        ("render", "Render frames with CUDA/Slang"),
    ):
        command = sub.add_parser(name, help=help_text)
        commands[name] = command
        command.add_argument("--config", type=Path, required=True, help="Validated .json, .yaml, or .yml preset")
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--synthetic", action="store_true", help="Built-in HU phantom with LPS axes")
        source.add_argument("--nifti", type=Path, help="Raw CT NIfTI; native array axes")
        source.add_argument("--dicom", type=Path, help="Raw CT DICOM directory; native array axes")
        if name == "render":
            source.add_argument("--cache", type=Path, help="Preprocessed attenuation volume")
        command.add_argument("--output", type=Path, help="New output directory; overrides preset output.output_dir")
        command.add_argument(
            "--dryrun",
            "--dry-run",
            action="store_true",
            help="Validate and print settings without loading voxels, using a GPU or writing files",
        )
    render = commands["render"]
    pose = render.add_mutually_exclusive_group()
    pose.add_argument("--view", choices=VIEWS, help="Clinical view; requires LPS volume metadata")
    pose.add_argument(
        "--rotation-deg",
        nargs=3,
        type=finite,
        metavar=("RX", "RY", "RZ"),
        help="Raw ZXY Euler angles in the volume frame; default 0 0 0",
    )
    render.add_argument("--translation-mm", nargs=3, type=finite, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    render.add_argument("--frames", type=positive_int, default=1, help="Repeat this pose with independent noise")
    render.add_argument("--fps", type=positive, default=15.0, help="Sequence metadata rate")
    render.add_argument(
        "--calibrate-display",
        nargs=2,
        type=finite,
        metavar=("P_LOW", "P_HIGH"),
        help="Fit and freeze a log display window from these volume percentiles",
    )
    return parser


def _source(args) -> dict:
    for kind in ("cache", "nifti", "dicom"):
        path = getattr(args, kind, None)
        if path is not None:
            path = path.expanduser().resolve()
            if (kind in ("cache", "dicom") and not path.is_dir()) or (kind == "nifti" and not path.is_file()):
                raise ValueError(f"{kind} input does not exist: {path}")
            if kind == "cache" and not all((path / name).is_file() for name in ("metadata.json", "mu_volume.npy")):
                raise ValueError("Cache must contain metadata.json and mu_volume.npy")
            return {"kind": kind, "path": str(path)}
    return {"kind": "synthetic", "path": None}


def _load_volume(source, settings):
    kind, path = source["kind"], source["path"]
    if kind == "cache":
        volume = PreprocessedVolume.load(path)
    else:
        if kind == "synthetic":
            z, y, x = np.mgrid[:64, :64, :64] - 31.5
            radius = np.sqrt(x * x + y * y + z * z)
            hu = np.full(radius.shape, -1000.0, dtype=np.float32)
            hu[radius < 25] = 40.0
            hu[(x - 8) ** 2 + (y + 5) ** 2 + z * z < 8**2] = 900.0
            preprocessor = VolumePreprocessor.from_numpy(
                hu, settings=settings, spacing_zyx_mm=(1.0, 1.0, 1.0), anatomical_frame="LPS"
            )
        elif kind == "nifti":
            preprocessor = VolumePreprocessor.from_nifti(path, settings=settings)
        else:
            preprocessor = VolumePreprocessor.from_dicom(path, settings=settings)
        volume = preprocessor.preprocess()
    if not np.isfinite(volume.mu_volume).all() or np.any(volume.mu_volume < 0):
        raise ValueError("Attenuation volume must be finite and nonnegative")
    return volume


def _save_frames(cine, output, image_format):
    if image_format == "png":
        from PIL import Image
    for frame in cine:
        stem = output / f"frame_{frame.frame_idx:04d}"
        if image_format == "npy":
            np.save(stem.with_suffix(".npy"), frame.image)
        elif image_format == "npz":
            np.savez_compressed(stem.with_suffix(".npz"), image=frame.image)
        else:
            Image.fromarray(np.rint(np.clip(frame.image, 0, 1) * 255).astype(np.uint8)).save(stem.with_suffix(".png"))
        if frame.intensity is not None:
            np.save(output / f"intensity_{frame.frame_idx:04d}.npy", frame.intensity)


def run(args) -> dict:
    config = SimulatorConfig.from_preset(args.config)
    source = _source(args)
    settings: PreprocessingSettings | None = config.preprocessing
    if source["kind"] == "cache":
        if settings != PreprocessingSettings():
            raise ValueError("Custom preprocessing cannot modify a cached mu-volume; preprocess the original HU source")
        settings = None
    output_path = args.output or config.output.output_dir
    if output_path is None:
        raise ValueError("Set output.output_dir in the preset or pass --output")
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing output directory: {output}")
    # The launcher owns frame export and refuses existing directories. Disable the
    # simulator's automatic writer to avoid two writers for the same frame.
    config = config.with_output(save_to_disk=False, output_dir=str(output))
    plan = {
        "command": args.command,
        "source": source,
        "output": str(output),
        "preset": config.to_dict(),
        "preprocessing": asdict(settings) if settings is not None else None,
    }
    cached_metadata = None
    if source["kind"] == "cache":
        cached_metadata = VolumeMetadata.from_dict(json.loads((Path(source["path"]) / "metadata.json").read_text()))
        plan["volume_metadata"] = cached_metadata.to_dict()
    if args.command == "render":
        if args.calibrate_display is not None:
            low, high = args.calibrate_display
            if not 0 <= low < high <= 100:
                raise ValueError("Calibration percentiles must satisfy 0 <= LOW < HIGH <= 100")
        rotation = view_rotation(args.view) if args.view else tuple(np.radians(args.rotation_deg or (0.0, 0.0, 0.0)))
        pose = Pose(rotation=rotation, translation=tuple(args.translation_mm), view=args.view)
        known_frame = (
            "LPS" if source["kind"] == "synthetic" else (cached_metadata.anatomical_frame if cached_metadata else None)
        )
        warning = view_frame_warning(pose.view, known_frame)
        if warning:
            raise ValueError(warning + " Use --rotation-deg in the source's native axes, or supply an LPS cache.")
        plan.update(
            pose=pose.to_dict(), frames=args.frames, fps=args.fps, calibration_percentiles=args.calibrate_display
        )
    if args.dryrun:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return plan
    if args.command == "render" and config.output.format == "png":
        try:
            import PIL.Image  # noqa: F401
        except ImportError as exc:
            raise ValueError("PNG output requires Pillow; install xray-simulator[cli]") from exc
    volume = _load_volume(source, settings)
    plan["volume_metadata"] = volume.metadata.to_dict()
    if args.command == "preprocess":
        output.mkdir(parents=True, exist_ok=False)
        volume.save(output)
    else:
        simulator = xray_simulator(volume, config)
        if args.calibrate_display is not None:
            simulator.calibrate_display(pose=pose, percentiles=tuple(args.calibrate_display))
        plan["preset"] = config.with_display(**asdict(simulator.display)).to_dict()
        cine = simulator.render_cine([pose] * args.frames, fps=args.fps, progress=False)
        output.mkdir(parents=True, exist_ok=False)
        _save_frames(cine, output, config.output.format)
    (output / "run.json").write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {args.command} output to {output}")
    return plan


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    return 0
