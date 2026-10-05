"""Build the measured DRR-RATE report and a local paired-image browser."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/i4h-drr-rate-mpl')
sys.dont_write_bytecode = True
# Configure the plotting cache and backend before importing plotting modules.
import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def natural(text):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', text)]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def csvfile(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def flat(r):
    raw, fitted = r['scores']['as_configured'], r['scores']['shape_only']
    control = r['misposed_scores']['as_configured']
    display = r['default_display_scores']['as_configured']
    styled = r.get('reference_style_display_scores', {}).get('as_configured', {})
    return {**{k:r[k] for k in ['case', 'subject', 'view', 'model']},
        'ncc': raw['ncc'], 'gradient_ncc': raw['gradient_ncc'], 'mi_nats': raw['mi_nats'],
        'shape_ssim': fitted['ssim'], 'shape_rmse': fitted['rmse'],
        'misposed_ncc': control['ncc'], 'misposed_gradient_ncc': control['gradient_ncc'],
        'gradient_ncc_gap': r['gradient_ncc_gap'],
        'default_display_ssim': display['ssim'], 'default_display_rmse': display['rmse'],
        'reference_style_display_ssim': styled.get('ssim'), 'reference_style_display_rmse': styled.get('rmse'),
        'directory': r['directory']}


METRICS = ['ncc', 'gradient_ncc', 'shape_ssim', 'shape_rmse', 'misposed_gradient_ncc', 'gradient_ncc_gap',
           'default_display_ssim', 'default_display_rmse', 'reference_style_display_ssim', 'reference_style_display_rmse']


def summary(group, labels):
    result = {**labels, 'comparisons': len(group), 'matched_beats_misposed': sum(r['gradient_ncc_gap'] > 0 for r in group)}
    for key in METRICS:
        values = [r[key] for r in group if r[key] is not None]
        q = np.percentile(values, [25, 50, 75]) if values else [None]*3
        for name, value in zip(['q25', 'median', 'q75'], q):
            result[f'{key}_{name}'] = None if value is None else float(value)
    return result


def figures(out, records, subjects, rows):
    models = [('reference_recipe', 'Matched DRR-RATE\nsignal model', '#197a9b'), ('stock_hu_mapping', 'Stock simulator\nHU mapping', '#b4662d')]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), constrained_layout=True)
    for ax, key, title in zip(axes, ['ncc', 'gradient_ncc', 'shape_ssim'], ['NCC', 'Gradient NCC', 'SSIM after affine intensity fit']):
        for i, (model, label, color) in enumerate(models):
            values = [r[key+'_median'] for r in subjects if r['model'] == model]
            ax.scatter(i+np.linspace(-.1, .1, len(values)), values, color=color, s=22)
            ax.plot([i-.18, i+.18], [np.median(values)]*2, c='black', linewidth=2)
        ax.set_xticks([0, 1], [r[1] for r in models])
        ax.set(title=title, ylim=(-.05, 1.04))
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('X-ray comparison with published DRR-RATE projections\nEach dot: one subject’s AP/lateral median; black bars: cohort median', fontsize=12)
    fig.savefig(out/'comparison_summary.png', dpi=160)
    fig.savefig(out/'comparison_summary.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
    for ax, (model, label, color) in zip(axes, models):
        group = [r for r in rows if r['model'] == model]
        ax.scatter([r['misposed_gradient_ncc'] for r in group], [r['gradient_ncc'] for r in group], c=color, s=25)
        ax.plot([-.2, 1], [-.2, 1], 'k--', linewidth=1)
        ax.set(title=label.replace('\n', ' '), xlabel='Misposed gradient NCC', ylabel='Matched gradient NCC', xlim=(-.2, 1.02), ylim=(-.2, 1.02))
        ax.grid(alpha=.2)
    fig.suptitle('Pose controls: all 40 views per signal model')
    fig.savefig(out/'pose_controls.png', dpi=160)
    plt.close(fig)
    lookup = {(r['case'], r['view'], r['model']): r for r in records}
    cases = sorted({r['case'] for r in records}, key=natural)
    # Fixed subject order; all subjects shown, with no selection by score.
    for view in ['AP', 'LATERAL']:
        fig, axes = plt.subplots(20, 3, figsize=(10, 48))
        for i, case in enumerate(cases):
            recipe = out/lookup[case, view, 'reference_recipe']['directory']
            stock = out/lookup[case, view, 'stock_hu_mapping']['directory']
            files = [recipe/'reference.png', recipe/'reference_style_display.png', stock/'xray_contrast_preview.png']
            titles = ['Published reference', 'Our renderer: matched recipe/export', 'Our renderer: stock mapping, contrast preview']
            for ax, path, title in zip(axes[i], files, titles):
                ax.imshow(plt.imread(path), cmap='gray', vmin=0, vmax=1)
                ax.set_title(case+' / '+view+'\n'+title, fontsize=8)
                ax.axis('off')
        fig.tight_layout()
        fig.savefig(out/f'all_subjects_{view.lower()}.png', dpi=110)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output
    protocol = json.loads((out/'protocol.json').read_text())
    records = [json.loads(p.read_text()) for p in out.glob('*/*/*/comparison.json')]
    records.sort(key=lambda r:(natural(r['case']), r['view'], r['model']))
    assert len(records) == 80
    assert len({(r['case'], r['view']) for r in records}) == 40
    assert len({r['subject'] for r in records}) == 20
    assert all((out/r['case']/'completed.json').is_file() for r in records)
    assert not json.loads((out/'run_status.json').read_text())['failures']
    assert all(hashlib.sha256((Path(protocol['simulator_repository'])/f).read_bytes()).hexdigest() == h for f,h in protocol['source_hashes'].items())
    for r in records:
        for name in ['reference', 'attenuation', 'misposed_attenuation', 'mask', 'xray_default']:
            a = np.load(out/r['directory']/(name+'.npy'), mmap_mode='r')
            assert a.shape == (512, 512) and np.isfinite(a).all()
        assert all(c['maximum_requested_steps'] <= c['shader_limit'] for c in r['step_capacity'].values())
        assert r['floor_transmission_fraction'] == 0
    rows = [flat(r) for r in records]
    assert all(np.isfinite(r[k]) for r in rows for k in ['ncc', 'gradient_ncc', 'shape_ssim', 'shape_rmse', 'default_display_ssim'])
    models = ['reference_recipe', 'stock_hu_mapping']
    subjects = [summary([r for r in rows if r['model']==m and r['subject']==s], {'model':m, 'subject':s}) for m in models for s in sorted({r['subject'] for r in rows}, key=natural)]
    cohort = []
    for m in models:
        group = [r for r in subjects if r['model']==m]
        medians = [{k:r[k+'_median'] for k in METRICS} for r in group]
        result = summary(medians, {'model':m, 'subjects':20})
        result.update(comparisons=40, matched_beats_misposed=sum(r['matched_beats_misposed'] for r in group))
        cohort.append(result)
    by_view = [summary([r for r in rows if r['model']==m and r['view']==v], {'model':m, 'view':v}) for m in models for v in ['AP','LATERAL']]
    integrations = [{**{k:r[k] for k in ['case','subject','view','model']}, **r['integration_check']} for r in records if 'integration_check' in r]
    assert len(integrations) == 20
    for name, values in [('per_view_metrics.csv', rows), ('per_subject_summary.csv', subjects), ('cohort_summary.csv', cohort), ('projection_summary.csv', by_view), ('integration_checks.csv', integrations)]:
        csvfile(out/name, values)
    report = {'schema_version':1, 'protocol':protocol, 'paired_views':40, 'subjects':20, 'models':2,
              'cohort_summary':cohort, 'projection_summary':by_view, 'subject_summary':subjects, 'integration_checks':integrations, 'pairs':records}
    dump(out/'report.json', report)
    figures(out, records, subjects, rows)
    labels = {'reference_recipe':'Matched DRR-RATE signal model', 'stock_hu_mapping':'Stock simulator HU mapping'}
    table_rows = '\n'.join(f'| {labels[r["model"]]} | {r["ncc_median"]:.3f} | {r["gradient_ncc_median"]:.3f} | {r["shape_ssim_median"]:.3f} | {r["matched_beats_misposed"]}/40 |' for r in cohort)
    max_l2 = max(r['relative_l2'] for r in integrations)
    recipe, stock = cohort
    readme = f'''# X-ray comparison against DRR-RATE

Generated and compared **40 matched X-ray views from 20 distinct CT-RATE validation subjects**, with AP and lateral views for each subject. The reference images are published DRR-RATE PNGs generated by the independent Siddon–Jacobs renderer. The i4h simulator source was unchanged at commit `{protocol['simulator_commit']}`.

## Measured agreement

Each value is the median across per-subject AP/lateral medians. Shape SSIM includes an in-sample gain/bias fit; it does not establish intensity calibration.

| Signal model | NCC | Gradient NCC | Shape SSIM | Matched beats misposed |
| --- | ---: | ---: | ---: | ---: |
{table_rows}

![Structural comparison](comparison_summary.png)

The matched recipe uses the published -100 HU threshold: `mu_proxy = max(trunc(HU)+100,0) * 1e-5 /mm`. The stock mapping uses the simulator's actual `HuToMuMapping()` defaults, from -1000 HU / 0 per mm to 3000 HU / 0.02 per mm. Both use the same source-derived projection geometry. No registration, pose optimization, or reference-fitted window was applied.

Using the matched signal model and emulating the reference export (alpha-integral, signed-short quantization, generated-image min/max scaling, 8-bit output), display SSIM was **{recipe['reference_style_display_ssim_median']:.3f}**. This export normalization is part of matching the published synthetic pipeline; it does not validate absolute brightness. The simulator's fixed `[0,6]` X-ray window scored **{recipe['default_display_ssim_median']:.3f}** with the matched signal model and **{stock['default_display_ssim_median']:.3f}** with the stock HU mapping.

## Scope and interpretation

This is an independent **synthetic X-ray renderer comparison**. It adds a chest dataset beyond the earlier DeepFluoro/Ljubljana experiment. Strong agreement under the matched recipe supports geometric and projection-structure consistency with the published DRRs. Results for the stock mapping are reported separately because its signal model differs from DRR-RATE.

DRR-RATE images are synthetic; this is not validation against real diagnostic radiographs. The experiment does not validate dose, spectrum, scatter, noise, detector response, temporal behavior, clinical findings or task performance. No pass/fail threshold was defined. The cohort is a deterministic 20-subject subset, not the full validation split.

## Verification and controls

- All 40 pairs completed under both mappings, producing 80 scored comparisons and independent misposed controls.
- Every subject's AP view with the matched recipe was also rendered at 0.25 mm versus the normal 0.5 mm integration step. Maximum relative L2 difference: **{max_l2:.3%}**.
- Every render was checked against the shader's 2048-step limit. No transmission values hit the attenuation extraction floor.
- The control increments the camera translation parameter by 5 mm along simulator X and premultiplies its orientation by a 5° world-Z rotation. Complete poses are retained in each record.
- All source CT files passed upstream SHA-256 checks. The CT-RATE v1 HU intercept/slope and voxel-spacing corrections came from the authors' validation metadata.
- All simulator source hashes match the pre-run snapshot. Numerical integration checks measure consistency, not external physical calibration.

## Inspect the results

- [Interactive image report](REPORT.html), containing every AP/lateral pair and both mappings.
- [All per-view metrics](per_view_metrics.csv), [subject summaries](per_subject_summary.csv), [cohort summaries](cohort_summary.csv), and [AP/lateral summaries](projection_summary.csv).
- [Full numerical report](report.json), [protocol and source hashes](protocol.json), [cohort](cohort.json), and [integration checks](integration_checks.csv).
- [All-subject AP sheet](all_subjects_ap.png) and [lateral sheet](all_subjects_lateral.png).
- [Rendering script](render_and_compare.py) and [report builder](build_report.py).

The raw arrays and image gallery remain local. References and generated anatomy images retain their dataset restrictions; the simulator license does not relicense them.

## Sources

- [DRR-RATE](https://huggingface.co/datasets/farrell236/DRR-RATE), revision `{protocol['drr_revision']}`; Hou et al., [Shadow and Light](https://arxiv.org/html/2406.03688v1).
- [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE), revision `{protocol['ct_revision']}` and its validation metadata/correction note.
- [Reference generator](https://github.com/farrell236/midas-journal-784/blob/889727fe0049bf89091f9c2d943299f428ba2a65/getDRRSiddonJacobsRayTracing.cxx).
- [Siddon–Jacobs implementation reviewed](https://github.com/InsightSoftwareConsortium/ITKTwoProjectionRegistration/blob/fc9714977f5c053a0b68eb5a3761812496ab781b/include/itkSiddonJacobsRayCastInterpolateImageFunction.hxx).
'''
    (out/'README.md').write_text(readme)
    summary_html = ''.join('<tr>'+''.join(f'<td>{x}</td>' for x in [labels[r['model']],f"{r['ncc_median']:.3f}",f"{r['gradient_ncc_median']:.3f}",f"{r['shape_ssim_median']:.3f}",f"{r['matched_beats_misposed']}/40"] )+'</tr>' for r in cohort)
    methods = ''.join(f'<dt>{html.escape(k.replace("_"," "))}</dt><dd>{html.escape(str(v))}</dd>' for k,v in protocol.items() if k not in ('source_hashes','packages'))
    gallery = []
    for case in sorted({r['case'] for r in rows}, key=natural):
        for view in ['AP','LATERAL']:
            gallery.append({'case':case,'view':view,**{model:next(r for r in rows if r['case']==case and r['view']==view and r['model']==model) for model in models}})
    template = '''<!DOCTYPE html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>X-ray comparison against DRR-RATE</title>
<style>body{font:16px/1.5 system-ui,sans-serif;background:#f3f6f8;color:#1b2935;margin:0}main{max-width:1450px;padding:28px;margin:auto}h1{font-size:2rem}h2{margin-top:32px}.card{background:white;border:1px solid #dae2e8;border-radius:10px;padding:20px;margin:20px 0}.note{background:#fff5e6;border-left:5px solid #be8a35;padding:18px}table{border-collapse:collapse;width:100%}th,td{padding:10px;border-bottom:1px solid #dce4e9;text-align:left}th{background:#edf2f5}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.grid img{width:100%}.grid figure{margin:0}.grid figcaption{font-weight:600;font-size:.85rem;min-height:44px}.plot{max-width:1200px;width:100%}select{padding:8px;margin:4px 16px 4px 0}a{color:#156483}.small{font-size:.9rem;color:#52616d}dt{font-weight:700;margin-top:12px}dd{margin:0}.scroll{overflow-x:auto}@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:550px){.grid{grid-template-columns:1fr}main{padding:12px}}</style>
<main><h1>Our X-ray renderer against DRR-RATE</h1><p>40 matched AP/lateral views · 20 distinct CT-RATE validation subjects · two signal models · unchanged CUDA/Slang simulator</p>
<div class="note"><strong>What this validates:</strong> consistency with published synthetic X-rays from another renderer. It does not establish agreement with real diagnostic radiographs or calibrated scanner physics. The matched DRR-RATE signal model and the simulator's stock HU mapping are reported separately.</div>
<h2>Structural agreement</h2><p>Numbers are medians across per-subject AP/lateral medians. Shape SSIM includes a nonnegative gain/bias fit on the scored image; it does not measure absolute calibration.</p><div class="card scroll"><table><thead><tr><th>Signal model</th><th>NCC</th><th>Gradient NCC</th><th>Shape SSIM</th><th>Matched &gt; misposed</th></tr></thead><tbody>@@SUMMARY@@</tbody></table></div><img class="plot" src="comparison_summary.png"><p><a href="README.md">Written report</a> · <a href="per_view_metrics.csv">Per-view CSV</a> · <a href="projection_summary.csv">AP/lateral summaries</a> · <a href="comparison_summary.pdf">PDF figure</a> · <a href="report.json">Complete numerical report</a></p>
<h2>Every matched image pair</h2><div class="card"><label>Case <select id="case"></select></label><label>View <select id="view"><option>AP</option><option>LATERAL</option></select></label><p id="metrics"></p><div class="grid"><figure><figcaption>Published DRR-RATE reference</figcaption><a id="reference-link"><img id="reference"></a></figure><figure><figcaption>Our renderer: matched signal + reference-style export</figcaption><a id="recipe-link"><img id="recipe"></a></figure><figure><figcaption>Our renderer: matched signal + default X-ray window</figcaption><a id="recipe-default-link"><img id="recipe-default"></a></figure><figure><figcaption>Our renderer: stock HU mapping + default X-ray window</figcaption><a id="stock-link"><img id="stock"></a></figure></div><p class="small">The reference-style export uses the generated image's own min/max, matching the source generator's normalization. Both default-window images use the simulator's fixed [0,6] log window. Click an image for full resolution. No reference-fitted window or spatial registration was applied.</p><a id="recipe-json">Matched-model measurements</a> · <a id="stock-json">Stock-model measurements</a></div>
<h2>Controls and numerical checks</h2><img class="plot" src="pose_controls.png"><p>Every comparison has a separately rendered pose control. Every subject's AP projection with the matched signal model also has a 0.25 mm versus 0.5 mm integration check; maximum relative L2 difference: <strong>@@INTEGRATION@@</strong>. All rays stayed within the shader step limit, and no transmission values reached the extraction floor. <a href="integration_checks.csv">Integration checks</a> · <a href="verification.json">Verification</a>.</p>
<h2>All subjects</h2><p>Shown in fixed numeric subject order, without selecting by similarity score. Stock-mapping contrast previews in these sheets are independently normalized for viewing.</p><details class="card"><summary>AP views</summary><img class="plot" src="all_subjects_ap.png"></details><details class="card"><summary>Lateral views</summary><img class="plot" src="all_subjects_lateral.png"></details>
<h2>Protocol and limits</h2><p>The 20 subjects and first reconstruction per subject were selected before scoring. CT-RATE v1 intensity and spacing corrections were applied from the authors' metadata. Camera geometry follows the published generator's transforms. The matched signal model uses the -100 HU threshold; the stock model uses the simulator's default HU ramp. Full numerical records retain source hashes, geometry, poses, integration settings and metric settings.</p><p>This is a subset of the validation split. No clinical task, dose, spectrum, scatter, detector noise or temporal behavior was validated, and no acceptance threshold was defined. Reference and generated anatomy images remain local.</p><details class="card"><summary>Exact methods</summary><dl>@@METHODS@@</dl></details><p><a href="protocol.json">Protocol</a> · <a href="cohort.json">Selected cohort and CT hashes</a> · <a href="render_and_compare.py">Rendering script</a> · <a href="build_report.py">Report builder</a> · <a href="run_status.json">Completion status</a></p><p class="small">Sources: <a href="https://huggingface.co/datasets/farrell236/DRR-RATE">DRR-RATE</a>, <a href="https://huggingface.co/datasets/ibrahimhamamci/CT-RATE">CT-RATE</a>, <a href="https://arxiv.org/html/2406.03688v1">Hou et al., Shadow and Light</a>. This experiment uses the unmodified simulator at <code>@@COMMIT@@</code>.</p></main>
<script>const pairs=@@PAIRS@@;const $=id=>document.getElementById(id);for(const name of [...new Set(pairs.map(p=>p.case))]){const o=document.createElement('option');o.value=name;o.textContent=name;$('case').appendChild(o)}function show(){const p=pairs.find(p=>p.case===$('case').value&&p.view===$('view').value);const a=p.reference_recipe,b=p.stock_hu_mapping;const image=(id,path)=>{$(id).src=path;$(id+'-link').href=path};image('reference',a.directory+'/reference.png');image('recipe',a.directory+'/reference_style_display.png');image('recipe-default',a.directory+'/xray_default.png');image('stock',b.directory+'/xray_default.png');$('metrics').textContent='Matched recipe: NCC '+a.ncc.toFixed(3)+' · Gradient NCC '+a.gradient_ncc.toFixed(3)+' · Reference-style display SSIM '+a.reference_style_display_ssim.toFixed(3)+' | Stock HU mapping: NCC '+b.ncc.toFixed(3)+' · Gradient NCC '+b.gradient_ncc.toFixed(3);$('recipe-json').href=a.directory+'/comparison.json';$('stock-json').href=b.directory+'/comparison.json'}$('case').onchange=show;$('view').onchange=show;show();</script></html>'''
    for key,value in {'SUMMARY':summary_html,'METHODS':methods,'INTEGRATION':f'{max_l2:.3%}','COMMIT':protocol['simulator_commit'][:12],'PAIRS':json.dumps(gallery,allow_nan=False)}.items():
        template = template.replace('@@'+key+'@@', value)
    (out/'REPORT.html').write_text(template)
    shutil.copy2(__file__, out/'build_report.py')
    dump(out/'verification.json', {'paired_views':40,'subjects':20,'model_comparisons':80,'misposed_controls':80,'integration_checks':20,
        'finite_arrays_and_metrics':True,'source_hashes_match':True,'all_rays_within_step_limit':True,'no_transmission_floor_hits':True})
    print(json.dumps({'report':str(out/'REPORT.html'),'cohort_summary':cohort,'max_integration_relative_l2':max_l2},indent=2))


if __name__ == '__main__':
    main()
