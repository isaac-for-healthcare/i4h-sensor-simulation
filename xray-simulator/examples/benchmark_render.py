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

"""Time one DRR per pose against one batched dispatch, with and without hardware sampling.

Uses a synthetic attenuation volume, so no CT data is needed:

    python examples/benchmark_render.py --device-type cuda --batch 1 8 16
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from xray_simulator.rendering.diffdrr_slang_renderer import SlangDiffDRRConfig, SlangDiffDRRRenderer


def _volume(shape_zyx: tuple[int, int, int]) -> np.ndarray:
    """A soft-tissue box with a denser ellipsoid, roughly the attenuation range of an abdomen."""
    z, y, x = np.ogrid[: shape_zyx[0], : shape_zyx[1], : shape_zyx[2]]
    centre = [s / 2 for s in shape_zyx]
    inside = sum(((axis - c) / (0.3 * s)) ** 2 for axis, c, s in zip((z, y, x), centre, shape_zyx)) < 1.0
    mu = np.full(shape_zyx, 0.02, np.float32)
    mu[inside] = 0.05
    return mu


def _time(fn, repeats: int) -> float:
    fn()  # compile and allocate outside the timing
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return 1000.0 * (time.perf_counter() - start) / repeats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device-type", choices=("cuda", "vulkan"), default="cuda")
    parser.add_argument("--shape-zyx", type=int, nargs=3, default=(431, 311, 311))
    parser.add_argument("--spacing-zyx-mm", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--detector-px", type=int, default=1024)
    parser.add_argument("--pixel-spacing-mm", type=float, default=0.6)
    parser.add_argument("--step-mm", type=float, default=1.0)
    parser.add_argument("--batch", type=int, nargs="+", default=(1, 8, 16))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--torch", action="store_true", help="also time render_batch(as_torch=True)")
    args = parser.parse_args()

    mu = _volume(tuple(args.shape_zyx))
    renderers = {}
    for hardware in (False, True):
        cfg = SlangDiffDRRConfig(
            det_width_px=args.detector_px,
            det_height_px=args.detector_px,
            pixel_spacing_mm=args.pixel_spacing_mm,
            step_mm=args.step_mm,
            normalize=False,
            invert=False,
            device_type=args.device_type,
            hardware_sampling=hardware,
        )
        renderers[hardware] = SlangDiffDRRRenderer(mu, tuple(args.spacing_zyx_mm), cfg=cfg)

    if args.torch:
        import torch

        def sync() -> None:
            if args.device_type == "cuda":
                torch.cuda.synchronize()
    else:

        def sync() -> None:
            pass

    rng = np.random.default_rng(0)
    print(f"\n{args.device_type}, volume {tuple(args.shape_zyx)}, detector {args.detector_px}^2, step {args.step_mm} mm")
    header = f"{'poses':>5}  {'sampling':>8}  {'loop ms':>8}  {'batch ms':>8}  {'ms/pose':>7}"
    print(header + (f"  {'torch ms':>8}" if args.torch else ""))
    for count in args.batch:
        rotations = rng.uniform(-0.4, 0.4, (count, 3)).astype(np.float32)
        translations = rng.uniform(-10.0, 10.0, (count, 3)).astype(np.float32)
        for hardware, renderer in renderers.items():
            loop = _time(lambda: [renderer.render(r, t) for r, t in zip(rotations, translations)], args.repeats)
            batch = _time(lambda: renderer.render_batch(rotations, translations), args.repeats)
            row = f"{count:>5}  {'hardware' if hardware else 'software':>8}  {loop:>8.2f}  {batch:>8.2f}  {batch / count:>7.2f}"
            if args.torch:
                on_gpu = _time(lambda: (renderer.render_batch(rotations, translations, as_torch=True), sync()), args.repeats)
                row += f"  {on_gpu:>8.2f}"
            print(row)


if __name__ == "__main__":
    main()
