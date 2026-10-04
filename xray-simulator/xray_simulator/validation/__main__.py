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

"""Evaluate a manifest of explicitly paired images without a GPU or network access."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .metrics import DOMAINS, evaluate_pair

METRICS = ("ncc", "gradient_ncc", "mi_nats", "ssim", "rmse", "wasserstein")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, domain: str, *, mask: bool = False) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() != ".png":
        raise ValueError(f"Use a prepared .npy array or an 8-bit grayscale .png: {path}")
    if domain != "display" and not mask:
        raise ValueError("PNG cannot establish attenuation units; use prepared floating-point .npy arrays")
    from PIL import Image

    with Image.open(path) as image:
        if image.mode in ("P", "1", "LA"):
            # Interpret palette colors/alpha before checking for overlays.
            image = image.convert("RGBA")
        values = np.asarray(image)
    if values.dtype != np.uint8:
        raise ValueError("PNG inputs must be 8-bit; prepare higher-bit-depth data explicitly as .npy")
    if values.ndim == 3:
        if values.shape[2] not in (3, 4):
            raise ValueError("Expected grayscale RGB or RGBA PNG")
        if not np.array_equal(values[:, :, 0], values[:, :, 1]) or not np.array_equal(values[:, :, 0], values[:, :, 2]):
            raise ValueError("Colored PNG/overlays cannot be treated as a grayscale radiograph")
        if values.shape[2] == 4 and not np.all(values[:, :, 3] == 255):
            raise ValueError("Transparent PNGs must be prepared explicitly before comparison")
        values = values[:, :, 0]
    return values.astype(np.float64) / 255.0


def read_manifest(path: Path):
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Manifest must be an object with schema_version=1")
    if manifest.get("domain") not in DOMAINS:
        raise ValueError(f"Manifest domain must be one of {DOMAINS}")
    for key in ("data_range", "histogram_range"):
        if key not in manifest:
            raise ValueError(f"Manifest must declare {key}")
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Manifest pairs must be a nonempty list")
    seen = set()
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("Each pair must be an object")
        for key in ("dataset", "subject", "view", "reference", "rendered", "mask"):
            if not isinstance(pair.get(key), str) or not pair[key].strip():
                raise ValueError(f"Each pair needs a nonempty {key} string")
        identity = tuple(pair[key] for key in ("dataset", "subject", "view"))
        if identity in seen:
            raise ValueError(f"Duplicate dataset/subject/view: {identity}")
        seen.add(identity)
        for key in ("acquisition", "misposed"):
            if key in pair and (not isinstance(pair[key], str) or not pair[key].strip()):
                raise ValueError(f"{key} must be a nonempty string when supplied")
        for key in ("reference", "rendered", "mask", "misposed"):
            if key in pair and not (path.parent / pair[key]).is_file():
                raise ValueError(f"Missing {key} file: {pair[key]}")
    return manifest


def summarize(records: list[dict], *, seed: int = 0, bootstrap_samples: int = 2000) -> list[dict]:
    """Summarize within subjects; bootstrap whole acquisitions when identified.

    Acquisition IDs must describe independent acquisition units. Repeated cine
    frames belong to the same acquisition. Without at least two such groups,
    report descriptive statistics only, never pixel-based confidence intervals.
    """
    groups = defaultdict(list)
    for record in records:
        groups[(record["dataset"], record["subject"])].append(record)
    rng = np.random.default_rng(seed)
    summaries = []
    for (dataset, subject), group in sorted(groups.items()):
        acquisition_ids = [record.get("acquisition") for record in group]
        have_groups = all(acquisition_ids) and len(set(acquisition_ids)) >= 2
        resamples = []
        if have_groups:
            blocks = [np.flatnonzero(np.array(acquisition_ids) == key) for key in sorted(set(acquisition_ids))]
            resamples = [
                np.concatenate([blocks[i] for i in draw])
                for draw in rng.integers(len(blocks), size=(bootstrap_samples, len(blocks)))
            ]
        summary = {
            "dataset": dataset, "subject": subject, "views": len(group),
            "ci_method": "95% percentile bootstrap of whole acquisitions" if have_groups else None,
            "ci_note": "Requires independent acquisition IDs; conditional on this subject" if have_groups
            else "No CI: supply at least two independent acquisition IDs for every view in this subject",
        }
        for mode in ("as_configured", "shape_only"):
            mode_summary = {}
            for metric in METRICS:
                values = np.array([
                    (record["scores"].get(mode) or {}).get(metric)
                    if (record["scores"].get(mode) or {}).get(metric) is not None else np.nan
                    for record in group
                ], dtype=float)
                valid = values[np.isfinite(values)]
                if not valid.size:
                    mode_summary[metric] = None
                    continue
                q25, median, q75 = (float(q) for q in np.percentile(valid, [25, 50, 75]))
                # Do not fabricate a CI from changing subsets when correlations
                # are undefined in some images or bootstrap draws.
                ci = None
                if resamples and np.isfinite(values).all():
                    estimates = [np.median(values[indices]) for indices in resamples]
                    ci = [float(q) for q in np.percentile(estimates, [2.5, 97.5])]
                mode_summary[metric] = {"defined_views": int(valid.size), "median": median,
                                        "q25": q25, "q75": q75, "iqr": q75 - q25, "ci95": ci}
            summary[mode] = mode_summary
        summaries.append(summary)
    return summaries


def run(manifest_path: Path, *, seed: int = 0) -> dict:
    manifest = read_manifest(manifest_path)
    settings = {key: manifest[key] for key in ("domain", "data_range", "histogram_range")}
    settings["bins"] = manifest.get("bins", 64)
    records = []
    for pair in manifest["pairs"]:
        paths = {key: (manifest_path.parent / pair[key]).resolve()
                 for key in ("reference", "rendered", "mask", "misposed") if key in pair}
        arrays = {key: _load(path, manifest["domain"], mask=key == "mask") for key, path in paths.items()}
        scores = evaluate_pair(arrays["reference"], arrays["rendered"], arrays["mask"], **settings)
        record = {**pair, "scores": scores,
                  "input_sha256": {key: _sha256(path) for key, path in paths.items()}}
        if "misposed" in arrays:
            control = evaluate_pair(arrays["reference"], arrays["misposed"], arrays["mask"], **settings)
            good = scores["as_configured"]["gradient_ncc"]
            bad = control["as_configured"]["gradient_ncc"]
            record["misposed_scores"] = control
            record["gradient_ncc_gap"] = good - bad if good is not None and bad is not None else None
        records.append(record)
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Paired image agreement; no automatic clinical-realism or pass/fail claim",
        "manifest": manifest, "manifest_sha256": _sha256(manifest_path),
        "packages": {name: version(name) for name in ("numpy", "scipy", "scikit-image")},
        "implementation_sha256": {path.name: _sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
        "bootstrap": {"seed": seed, "samples": 2000, "unit": "acquisition within subject"},
        "pairs": records, "subjects": summarize(records, seed=seed),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New JSON report; existing files are never overwritten")
    parser.add_argument("--seed", type=int, default=0, help="Acquisition bootstrap seed")
    parser.add_argument("--dryrun", "--dry-run", action="store_true", help="Check manifest and file paths; write nothing")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError(f"Output already exists: {args.output}")
        if args.seed < 0:
            raise ValueError("seed must be nonnegative")
        if args.dryrun:
            manifest = read_manifest(args.manifest)
            print(f"Preview: {len(manifest['pairs'])} explicit pairs, domain={manifest['domain']}, output={args.output}")
            print("Manifest and file paths checked; pixel values, alignment and metrics have not been checked.")
            return 0
        report = run(args.manifest, seed=args.seed)
        serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(serialized)
        print(f"Evaluated {len(report['pairs'])} pairs in {len(report['subjects'])} subjects: {args.output}")
        return 0
    except (ValueError, OSError, ImportError, KeyError, TypeError) as exc:
        parser.exit(2, f"Validation error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
