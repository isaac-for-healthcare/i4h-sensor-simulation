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

"""Draw DeepFluoro sample comparisons from a complete local render_and_compare.py output.

Views are chosen by a fixed rule, not by inspecting images: the subjects with the
highest, closest-to-dataset-median and lowest median gradient NCC, and for each the
view closest to that subject's median. Only DeepFluoro (CC BY-NC 4.0) imagery is
drawn; Ljubljana (CC BY-NC-ND 4.0) derivatives must not be published.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/i4h-paired-mpl')
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import binary_dilation, gaussian_filter, sobel  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

REFERENCE_EDGE, RENDER_EDGE = np.array([0x2a, 0x78, 0xd6]) / 255, np.array([0xeb, 0x68, 0x34]) / 255
DIVERGING = LinearSegmentedColormap.from_list('reference_minus_render', ['#2a78d6', '#f0efec', '#e34948'])


def edges(image, mask, fraction=0.12):
    """Strongest edges inside the scored ROI: the top fraction of Sobel magnitude."""
    smooth = gaussian_filter(np.asarray(image, dtype=np.float64), 1.0)
    magnitude = np.hypot(sobel(smooth, 0), sobel(smooth, 1))
    return mask & (magnitude >= np.quantile(magnitude[mask], 1 - fraction))


def select_views(out):
    subjects = {r['subject']: float(r['gradient_ncc_median']) for r in csv.DictReader((out / 'per_subject_summary.csv').open())
                if r['dataset'] == 'deepfluoro'}
    target = float(next(r for r in csv.DictReader((out / 'dataset_summary.csv').open())
                        if r['dataset'] == 'deepfluoro')['gradient_ncc_median_subject'])
    chosen = [('Highest subject median', max(subjects, key=subjects.get)),
              ('Closest to dataset median', min(subjects, key=lambda s: abs(subjects[s] - target))),
              ('Lowest subject median', min(subjects, key=subjects.get))]
    views = []
    for label, subject in chosen:
        records = [json.loads(p.read_text()) for p in sorted((out / 'deepfluoro' / subject).glob('*/comparison.json'))]
        record = min(records, key=lambda r: abs(r['scores']['as_configured']['gradient_ncc'] - subjects[subject]))
        views.append((label, record))
    return views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='Complete local render_and_compare.py output')
    parser.add_argument('--figure', type=Path, required=True)
    args = parser.parse_args()
    views = select_views(args.output)
    plt.rcParams.update({'font.size': 9})
    fig, axes = plt.subplots(len(views), 4, figsize=(13, 3.6 * len(views)), constrained_layout=True)
    headers = ['Real fluoroscopy\n(contrast preview)', 'Our render\n(fluoroscopy appearance)',
               'Strongest edges', 'Reference − our render\n(after gain/bias fit, scored ROI)']
    for axis, header in zip(axes[0], headers):
        axis.annotate(header, (0.5, 1.22), xycoords='axes fraction', ha='center', va='bottom', fontsize=10, weight='bold')
    for row, (label, record) in zip(axes, views):
        p = args.output / record['directory']
        mask = np.load(p / 'mask.npy')
        reference = np.load(p / 'reference_proxy.npy')
        attenuation = np.load(p / 'rendered_attenuation.npy')
        fit = record['scores']['affine_fit']
        fitted = fit['gain'] * attenuation / 6 + fit['bias']
        scores = record['scores']['as_configured']

        row[0].imshow(plt.imread(p / 'reference_contrast_preview.png'), cmap='gray', vmin=0, vmax=1)
        row[0].set_title(f'{label}\n{record["subject"]} / view {record["view"]}')
        row[1].imshow(plt.imread(p / 'fluoro_subject_window.png'), cmap='gray', vmin=0, vmax=1)
        row[1].set_title(f'NCC {scores["ncc"]:.3f}\ngradient NCC {scores["gradient_ncc"]:.3f}')

        ref_edges, our_edges = edges(reference, mask), edges(fitted, mask)
        overlay = np.repeat((0.35 * plt.imread(p / 'reference_contrast_preview.png'))[..., None], 3, axis=2)[..., :3]
        thick = lambda e: binary_dilation(e, iterations=1)  # noqa: E731
        overlay[thick(ref_edges) & ~thick(our_edges)] = REFERENCE_EDGE
        overlay[thick(our_edges) & ~thick(ref_edges)] = RENDER_EDGE
        overlay[thick(ref_edges) & thick(our_edges)] = 1.0
        row[2].imshow(overlay)
        row[2].set_title('blue: reference only · orange: ours only\nwhite: both')

        difference = np.where(mask, reference - fitted, np.nan)
        limit = float(np.nanquantile(np.abs(difference), 0.99))
        shown = row[3].imshow(difference, cmap=DIVERGING, vmin=-limit, vmax=limit)
        row[3].set_title('red: reference more attenuating\nblue: ours more attenuating')
        bar = fig.colorbar(shown, ax=row[3], fraction=0.046, pad=0.02)
        bar.set_label('Attenuation-proxy difference', fontsize=8)
        for axis in row:
            axis.axis('off')
    fig.legend(handles=[Patch(color=REFERENCE_EDGE, label='Reference edge only'), Patch(color=RENDER_EDGE, label='Our edge only'),
                        Patch(facecolor='white', edgecolor='#888888', label='Edge in both')],
               loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle('DeepFluoro: real fluoroscopy versus our simulator for matched pose and calibration\n'
                 'Views selected by a fixed rule from per-subject gradient NCC; no registration or image alignment', fontsize=11)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=90, bbox_inches='tight')
    # A 256-colour palette keeps the committed figure under the repository's 500 KB file limit.
    Image.open(buffer).convert('RGB').quantize(256, method=Image.Quantize.MEDIANCUT).save(args.figure, optimize=True)
    print(json.dumps([{'selection': label, 'subject': r['subject'], 'view': r['view'],
                       'gradient_ncc': r['scores']['as_configured']['gradient_ncc']} for label, r in views], indent=2))


if __name__ == '__main__':
    main()
