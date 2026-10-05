"""Summarize saved, paired simulator comparisons into CSV/JSON, figures and HTML."""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/i4h-paired-mpl')
sys.dont_write_bytecode = True
import matplotlib  # noqa: E402
# Configure the local plotting cache before importing Matplotlib.
import numpy as np  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def csv_file(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flat(r, diagnostic=False):
    c = r['offset_diagnostic'] if diagnostic else r
    raw, shape, control = c['scores']['as_configured'], c['scores']['shape_only'], c['misposed_scores']['as_configured']
    vessel = c.get('vessel_roi_scores', {}).get('as_configured', {})
    return {
        'dataset': r['dataset'], 'subject': r['subject'], 'view': r['view'],
        'variant': 'horizontal_offset_sign_diagnostic' if diagnostic else 'branch_as_is',
        'ncc': raw['ncc'], 'gradient_ncc': raw['gradient_ncc'], 'mi_nats': raw['mi_nats'],
        'proxy_ssim_raw': raw['ssim'], 'proxy_rmse_raw': raw['rmse'],
        'shape_ssim': shape['ssim'], 'shape_rmse': shape['rmse'], 'shape_wasserstein': shape['wasserstein'],
        'misposed_ncc': control['ncc'], 'misposed_gradient_ncc': control['gradient_ncc'],
        'gradient_ncc_gap': raw['gradient_ncc'] - control['gradient_ncc'],
        'vessel_roi_ncc': vessel.get('ncc'), 'vessel_roi_gradient_ncc': vessel.get('gradient_ncc'),
        'fluoro_default_ssim': None if diagnostic else r['display_scores']['fluoro_default']['as_configured']['ssim'],
        'fluoro_default_rmse': None if diagnostic else r['display_scores']['fluoro_default']['as_configured']['rmse'],
        'fluoro_subject_window_ssim': None if diagnostic else r['display_scores']['fluoro_subject_window']['as_configured']['ssim'],
        'fluoro_subject_window_rmse': None if diagnostic else r['display_scores']['fluoro_subject_window']['as_configured']['rmse'],
        'xray_default_ssim': None if diagnostic else r['display_scores']['xray_default']['as_configured']['ssim'],
        'xray_subject_window_ssim': None if diagnostic else r['display_scores']['xray_subject_window']['as_configured']['ssim'],
        'column_shift_px': c.get('predicted_column_shift_px', 0),
        'directory': r['directory'],
    }


def summaries(rows):
    groups = defaultdict(list)
    metrics = ['ncc', 'gradient_ncc', 'shape_ssim', 'shape_rmse', 'misposed_gradient_ncc', 'gradient_ncc_gap',
               'vessel_roi_ncc', 'vessel_roi_gradient_ncc', 'fluoro_default_ssim', 'fluoro_subject_window_ssim']
    for r in rows:
        groups[(r['dataset'], r['variant'], r['subject'])].append(r)
    subjects = []
    for (dataset, variant, subject), group in sorted(groups.items()):
        row = {'dataset': dataset, 'variant': variant, 'subject': subject, 'views': len(group),
               'aligned_beats_misposed': sum(r['gradient_ncc_gap'] > 0 for r in group)}
        for key in metrics:
            values = [r[key] for r in group if r[key] is not None]
            q = np.percentile(values, [25, 50, 75]) if values else [None] * 3
            row.update({f'{key}_q25': None if q[0] is None else float(q[0]),
                        f'{key}_median': None if q[1] is None else float(q[1]),
                        f'{key}_q75': None if q[2] is None else float(q[2])})
        subjects.append(row)
    datasets = []
    for dataset, variant in sorted({(r['dataset'], r['variant']) for r in subjects}):
        group = [r for r in subjects if r['dataset'] == dataset and r['variant'] == variant]
        result = {'dataset': dataset, 'variant': variant, 'subjects': len(group), 'views': sum(r['views'] for r in group),
                  'aligned_beats_misposed': sum(r['aligned_beats_misposed'] for r in group)}
        for key in metrics:
            values = [r[f'{key}_median'] for r in group if r[f'{key}_median'] is not None]
            result[f'{key}_median_subject'] = float(np.median(values)) if values else None
        datasets.append(result)
    return subjects, datasets


def plots(out, rows, subjects, records):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.7), constrained_layout=True)
    groups = [('deepfluoro', 'branch_as_is', 'DeepFluoro\ncurrent branch'),
              ('ljubljana', 'branch_as_is', 'Ljubljana\ncurrent branch'),
              ('ljubljana', 'horizontal_offset_sign_diagnostic', 'Ljubljana\noffset-sign diagnostic')]
    for axis, key, title in zip(axes, ['ncc', 'gradient_ncc', 'shape_ssim'], ['Intensity structure (NCC)', 'Edge structure (gradient NCC)', 'SSIM after affine intensity fit']):
        for i, (dataset, variant, label) in enumerate(groups):
            selected = [r for r in subjects if r['dataset'] == dataset and r['variant'] == variant]
            values = np.array([r[f'{key}_median'] for r in selected])
            axis.scatter(i + np.linspace(-.1, .1, len(values)), values, color=['#1666b0', '#d3603f', '#218776'][i], s=32)
            axis.plot([i - .18, i + .18], [np.median(values)] * 2, color='black', linewidth=2)
        axis.set_xticks(range(3), [g[2] for g in groups], fontsize=9)
        axis.set_title(title)
        axis.set_ylim(-.1, 1.03)
        axis.grid(axis='y', alpha=.2)
    fig.suptitle('Paired dataset comparison — one dot per subject median\nBlack bars: median across subjects; diagnostic variant is exploratory', fontsize=12)
    fig.savefig(out / 'comparison_summary.png', dpi=160)
    fig.savefig(out / 'comparison_summary.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for axis, dataset in zip(axes, ['deepfluoro', 'ljubljana']):
        for variant, color, label in [('branch_as_is', '#1666b0', 'Current branch'), ('horizontal_offset_sign_diagnostic', '#218776', 'Offset-sign diagnostic')]:
            group = [r for r in rows if r['dataset'] == dataset and r['variant'] == variant]
            if group:
                axis.scatter([r['misposed_gradient_ncc'] for r in group], [r['gradient_ncc'] for r in group], s=18, alpha=.7, c=color, label=label)
        axis.plot([-.1, 1], [-.1, 1], 'k--', linewidth=1)
        axis.set(xlim=(-.1, 1), ylim=(-.1, 1), xlabel='Misposed control gradient NCC', ylabel='Matched view gradient NCC', title=dataset.title())
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    fig.suptitle('Pose sensitivity — points above the diagonal favor the supplied pose')
    fig.savefig(out / 'pose_controls.png', dpi=160)
    plt.close(fig)
    # Every subject is represented by the first eligible view, chosen before scores.
    for dataset in ['deepfluoro', 'ljubljana']:
        selected = []
        for subject in sorted({r['subject'] for r in records if r['dataset'] == dataset}):
            selected.append(next(r for r in records if r['dataset'] == dataset and r['subject'] == subject))
        columns = 4 if dataset == 'ljubljana' else 3
        fig, axes = plt.subplots(len(selected), columns, figsize=(columns * 3, len(selected) * 2.5), squeeze=False)
        for row, record in enumerate(selected):
            p = out / record['directory']
            names = ['reference_contrast_preview.png', 'fluoro_subject_window.png', 'xray_subject_window.png']
            titles = ['Real reference (contrast preview)', 'Our fluoro (subject window)', 'Our X-ray (subject window)']
            if dataset == 'ljubljana':
                names = ['reference_contrast_preview.png', 'fluoro_subject_window.png', 'offset_diagnostic_fluoro.png', 'offset_diagnostic_xray.png']
                titles = ['Real reference (contrast preview)', 'Our fluoro: current branch', 'Our fluoro: offset diagnostic', 'Our X-ray: offset diagnostic']
            for col, (name, title) in enumerate(zip(names, titles)):
                axes[row, col].imshow(plt.imread(p / name), cmap='gray', vmin=0, vmax=1)
                axes[row, col].axis('off')
                axes[row, col].set_title(f'{record["subject"]}/{record["view"]}\n{title}', fontsize=8)
        fig.suptitle(f'{dataset.title()}: first eligible view for every subject\nReference contrast previews are independently scaled; quantitative scores use the recorded protocol.', fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, .975])
        fig.savefig(out / f'{dataset}_all_subjects.png', dpi=130)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output
    protocol = json.loads((out / 'protocol.json').read_text())
    records = [json.loads(p.read_text()) for p in sorted(out.glob('*/*/*/comparison.json'))]
    counts = {d: sum(r['dataset'] == d for r in records) for d in ['deepfluoro', 'ljubljana']}
    assert counts == {'deepfluoro': 362, 'ljubljana': 20}, counts
    assert len({(r['dataset'], r['subject'], r['view']) for r in records}) == 382
    assert all((out / d / s / 'completed.json').is_file() for d, s in {(r['dataset'], r['subject']) for r in records})
    assert not json.loads((out / 'run_status.json').read_text())['failures']
    rows = [flat(r) for r in records] + [flat(r, True) for r in records if 'offset_diagnostic' in r]
    for r in rows:
        for key in ['ncc', 'gradient_ncc', 'shape_ssim', 'shape_rmse', 'misposed_gradient_ncc']:
            assert np.isfinite(r[key]), (r['dataset'], r['subject'], r['view'], key)
    for r in records:
        p = out / r['directory']
        shape = np.load(p / 'reference_stored.npy', mmap_mode='r').shape
        for name in ['rendered_attenuation', 'fluoro_default', 'xray_default', 'fluoro_subject_window', 'xray_subject_window', 'mask']:
            a = np.load(p / f'{name}.npy', mmap_mode='r')
            assert a.shape == shape and np.isfinite(a).all(), (p, name)
    subjects, datasets = summaries(rows)
    csv_file(out / 'per_view_metrics.csv', rows)
    csv_file(out / 'per_subject_summary.csv', subjects)
    csv_file(out / 'dataset_summary.csv', datasets)
    integrations = [{**{key:r[key] for key in ['dataset','subject','view']}, **r['integration_check']} for r in records if 'integration_check' in r]
    csv_file(out / 'integration_checks.csv', integrations)
    report = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(), 'counts': counts,
              'protocol': protocol, 'dataset_summaries': datasets, 'subject_summaries': subjects,
              'integration_checks': integrations, 'pairs': records}
    dump(out / 'report.json', report)
    plots(out, rows, subjects, records)
    pretty = {'deepfluoro': 'DeepFluoro', 'ljubljana': 'Ljubljana'}
    def num(v): return '—' if v is None else f'{v:.3f}'
    summary_rows = ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in [pretty[r['dataset']], 'Current branch' if r['variant']=='branch_as_is' else 'Offset-sign diagnostic',
                       r['subjects'],r['views'],num(r['ncc_median_subject']),num(r['gradient_ncc_median_subject']),num(r['shape_ssim_median_subject']),f'{r["aligned_beats_misposed"]}/{r["views"]}']) + '</tr>' for r in datasets)
    subject_rows = ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in [pretty[r['dataset']],r['subject'],'Current branch' if r['variant']=='branch_as_is' else 'Offset diagnostic',r['views'],num(r['ncc_median']),num(r['gradient_ncc_median']),num(r['shape_ssim_median']),num(r['shape_rmse_median'])]) + '</tr>' for r in subjects)
    spec_rows = ''.join(f'<dt>{html.escape(k.replace("_", " ").capitalize())}</dt><dd>{html.escape(str(protocol[k]))}</dd>' for k in [
        'binning','integration_step_mm','deepfluoro_volume','ljubljana_volume','primary_roi','secondary_ljubljana_roi','reference_display',
        'deepfluoro_reference_proxy','ljubljana_reference_proxy','proxy_evaluation','display_evaluation','controls','integration_check','statistics'])
    galleries = []
    for r in records:
        item = {k:r[k] for k in ['dataset','subject','view','directory']}
        item['native'] = flat(r)
        item['diagnostic'] = flat(r, True) if 'offset_diagnostic' in r else None
        galleries.append(item)
    max_integration = max(r['relative_l2'] for r in integrations)
    template = '''<!DOCTYPE html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Paired X-ray and fluoroscopy comparison</title><style>
body{font-family:system-ui,sans-serif;margin:0;background:#f4f6f8;color:#192632;line-height:1.55}main{max-width:1250px;margin:auto;padding:32px}h1{font-size:2.1rem;line-height:1.15}h2{margin-top:36px}p{max-width:1000px}.lead{font-size:1.15rem}.card{background:white;border:1px solid #dce4ec;border-radius:12px;padding:22px;margin:20px 0}.tag{display:inline-block;padding:5px 12px;background:#dcebf5;border-radius:20px;margin-right:8px;font-size:.9rem}.note{border-left:5px solid #b87922;background:#fff8e9;padding:16px}a{color:#146299}table{border-collapse:collapse;width:100%;font-size:.91rem}th,td{padding:10px;border-bottom:1px solid #e0e7ed;text-align:left}th{background:#eef3f7}select{padding:8px;margin:4px 10px 4px 0;border:1px solid #a5b3bf;border-radius:5px}label{display:inline-block;font-weight:600}.controls{display:flex;gap:8px;flex-wrap:wrap}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.grid img{width:100%;background:#000}.grid figure{margin:0}.grid figcaption{font-size:.9rem;font-weight:600}img.plot{width:100%;max-width:1200px}.scroll{overflow-x:auto}dt{font-weight:600;margin-top:10px}dd{margin-left:0;color:#40515e}.small{font-size:.9rem;color:#526470}code{font-family:ui-monospace,monospace}button{padding:8px;border:1px solid #a5b3bf;background:white;border-radius:5px;cursor:pointer}@media(max-width:750px){main{padding:16px}.grid{grid-template-columns:1fr}h1{font-size:1.65rem}}</style>
<main><span class="tag">382 matched views</span><span class="tag">16 subjects</span><span class="tag">CUDA / Slang renderer</span>
<h1>Our simulator against DeepFluoro and Ljubljana</h1>
<p class="lead">Generated both X-ray and fluoroscopy appearances using each dataset's own volume and camera pose. All 362 eligible DeepFluoro views and all 20 primary Ljubljana views completed; four upstream-flagged DeepFluoro poses were excluded.</p>
<p class="small">Source: public sensor-simulation PR #73, commit <code>@@COMMIT@@</code>. Output includes 1,528 primary generated PNGs (two appearances × two display windows × 382 views), raw intensity/attenuation arrays, real reference arrays, masks, independent misposed renders, and complete per-view measurements.</p>
<div class="note"><strong>Ljubljana exposed a detector-offset discrepancy.</strong> The current branch's first frontal view was displaced horizontally by approximately 81 pixels, matching twice its stored detector offset divided by the binned pitch. All 20 views were additionally rendered with the horizontal offset sign reversed. This exploratory geometry diagnostic is listed separately; the simulator source and primary results remain unchanged. No image translation, registration, or pose optimization was applied.</div>
<h2>Measured agreement</h2><p>Each number below is the median of per-subject medians, giving every subject equal weight. NCC and gradient NCC measure structural agreement. Shape SSIM follows a per-image nonnegative gain/bias fit and therefore does not establish brightness or physical calibration. No pass/fail threshold was set.</p>
<div class="scroll card"><table><thead><tr><th>Dataset</th><th>Geometry</th><th>Subjects</th><th>Views</th><th>NCC</th><th>Gradient NCC</th><th>Shape SSIM</th><th>Aligned &gt; misposed</th></tr></thead><tbody>@@SUMMARY@@</tbody></table></div>
<img class="plot" src="comparison_summary.png" alt="Per-subject structural agreement"><p><a href="comparison_summary.pdf">Exportable PDF figure</a> · <a href="per_view_metrics.csv">All per-view scores (CSV)</a> · <a href="per_subject_summary.csv">Subject summaries (CSV)</a> · <a href="report.json">Full report (JSON)</a></p>
<h2>Inspect every matched view</h2><div class="card"><div class="controls"><label>Dataset<br><select id="dataset"></select></label><label>Subject<br><select id="subject"></select></label><label>View<br><select id="view"></select></label><label>Render geometry<br><select id="geometry"><option value="native">Current branch</option><option value="diagnostic">Horizontal offset diagnostic</option></select></label><label>Generated display<br><select id="display"><option value="subject_window">Frozen subject window</option><option value="default">Default window [0,6]</option></select></label><label>Reference display<br><select id="reference"><option value="contrast">Contrast preview</option><option value="stored">Stored 16-bit scale</option></select></label></div><p id="scores"></p><div class="grid"><figure><figcaption>Dataset reference — fluoroscopy polarity</figcaption><img id="refimg"></figure><figure><figcaption>Our fluoroscopy render</figcaption><img id="fluoroimg"></figure><figure><figcaption>Our X-ray render — opposite polarity</figcaption><img id="xrayimg"></figure></div><p class="small">Contrast previews rescale the real reference for viewing only. The generated subject window is derived from the first simulated view and frozen for the subject. Scores use the saved protocol. X-ray and fluoroscopy share the same simulated transport; these are still images, without temporal or dose-specific effects.</p><a id="pairlink">View this pair's full measurements and geometry</a></div>
<h2>Per-subject results</h2><details class="card"><summary>Show all subject summaries</summary><div class="scroll"><table><thead><tr><th>Dataset</th><th>Subject</th><th>Geometry</th><th>Views</th><th>NCC</th><th>Gradient NCC</th><th>Shape SSIM</th><th>Shape RMSE</th></tr></thead><tbody>@@SUBJECTS@@</tbody></table></div></details>
<h2>Pose and integration checks</h2><img class="plot" src="pose_controls.png"><p>The control adds 5 mm along simulator X and 5° about world Z, then re-renders. It uses the same reference and ROI. The first eligible view of each subject also completed a 0.25 mm integration check against the standard 0.5 mm render. Maximum relative L2 difference: <strong>@@INTEGRATION@@</strong>. This is numerical consistency, not external validation. <a href="integration_checks.csv">All integration checks</a>.</p>
<h2>All-subject image sheets</h2><p>These sheets show the first eligible view for every subject, selected before scoring.</p><details class="card"><summary>DeepFluoro: all six subjects</summary><img class="plot" src="deepfluoro_all_subjects.png"></details><details class="card"><summary>Ljubljana: all ten subjects and offset diagnostic</summary><img class="plot" src="ljubljana_all_subjects.png"></details>
<h2>Interpretation and limits</h2><p>The test measures agreement for matched anatomy and poses. The broad detector ROI retains residual instruments, collimation and background; an additional model-defined vessel ROI is available for Ljubljana in the JSON and CSV. Ljubljana uses an explicitly uncalibrated vessel-contrast model. Reference signal preparation is documented; neither the raw proxy errors nor the fitted shape scores establish calibrated attenuation. Default and frozen-subject-window display scores are reported separately.</p><p>Acquisition independence is not established, so no confidence interval is fabricated from repeated views. No scanner calibration, registration refinement, temporal detector behavior, motion, or stochastic noise was fitted. All dataset images and generated images remain local.</p>
<h2>Reproduce the run</h2><p><a href="protocol.json">Exact protocol, source hashes and versions</a> · <a href="render_and_compare.py">Rendering/comparison script</a> · <a href="build_report.py">Report builder</a> · <a href="run_status.json">Completion status</a></p><details class="card"><summary>Methods and parameters</summary><dl>@@METHODS@@</dl></details><p class="small">Data: <a href="https://huggingface.co/datasets/eigenvivek/xvr-data">xvr-data (pinned revision in protocol)</a>. Original datasets: <a href="https://github.com/rg2/DeepFluoroLabeling-IPCAI2020">DeepFluoro</a> and <a href="https://lit.fe.uni-lj.si/en/research/resources/3D-2D-GS-CA/">Ljubljana 3D-2D-GS-CA</a>. Reference preprocessing: <a href="https://github.com/eigenvivek/xvr/blob/caa55cc8096294cf70a218126bf16008dee0dec7/src/xvr/io/xray.py">xvr source</a>.</p></main>
<script>const pairs=@@PAIRS@@;const $=x=>document.getElementById(x);function options(id,values){$(id).innerHTML='';for(const value of values){const o=document.createElement('option');o.value=value;o.textContent=value;$(id).appendChild(o)}}function datasets(){options('dataset',[...new Set(pairs.map(x=>x.dataset))]);subjects()}function subjects(){options('subject',[...new Set(pairs.filter(x=>x.dataset===$('dataset').value).map(x=>x.subject))]);views()}function views(){options('view',pairs.filter(x=>x.dataset===$('dataset').value&&x.subject===$('subject').value).map(x=>x.view));show()}function show(){const p=pairs.find(x=>x.dataset===$('dataset').value&&x.subject===$('subject').value&&x.view===$('view').value);$('geometry').options[1].disabled=!p.diagnostic;if(!p.diagnostic)$('geometry').value='native';const diag=$('geometry').value==='diagnostic';const r=diag?p.diagnostic:p.native;const window=$('display').value;$('display').disabled=diag;const source=p.directory+'/';$('refimg').src=source+($('reference').value==='contrast'?'reference_contrast_preview.png':'reference_stored_display.png');$('fluoroimg').src=source+(diag?'offset_diagnostic_fluoro.png':'fluoro_'+window+'.png');$('xrayimg').src=source+(diag?'offset_diagnostic_xray.png':'xray_'+window+'.png');$('scores').textContent='NCC '+r.ncc.toFixed(3)+' · Gradient NCC '+r.gradient_ncc.toFixed(3)+' · Shape SSIM '+r.shape_ssim.toFixed(3)+' · Misposed gradient NCC '+r.misposed_gradient_ncc.toFixed(3)+(diag?' · Exploratory offset-sign variant':'');$('pairlink').href=source+'comparison.json'}$('dataset').onchange=subjects;$('subject').onchange=views;['view','geometry','display','reference'].forEach(x=>$(x).onchange=show);datasets();</script></html>'''
    for key, value in {'COMMIT': protocol['simulator_commit'][:12], 'SUMMARY': summary_rows, 'SUBJECTS': subject_rows, 'METHODS': spec_rows,
                       'INTEGRATION': f'{max_integration:.3%}', 'PAIRS': json.dumps(galleries, allow_nan=False)}.items():
        template = template.replace('@@' + key + '@@', value)
    (out / 'REPORT.html').write_text(template)
    shutil.copy2(__file__, out / 'build_report.py')
    dump(out / 'verification.json', {'paired_views': len(records), 'primary_generated_pngs': 4 * len(records),
        'unique_pairs': True, 'expected_counts_met': True, 'finite_equal_shape_arrays': True,
        'subject_completions': 16, 'offset_diagnostic_views': 20, 'integration_checks': len(integrations),
        'renderer_source_changed': False})
    print(json.dumps({'output': str(out / 'REPORT.html'), 'dataset_summary': datasets, 'max_relative_integration_l2': max_integration}, indent=2))


if __name__ == '__main__':
    main()
