"""Compare the public i4h X-ray renderer with matched published DRR-RATE views."""
from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/i4h-drr-rate-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.dont_write_bytecode = True
REPO = Path(os.environ['I4H_SENSOR_SIMULATION_REPO']).resolve()
DATA = Path(os.environ['I4H_XRAY_VALIDATION_DATA']).resolve()
COHORT = Path(__file__).with_name('cohort.json')
sys.path.insert(0, str(REPO / 'xray-simulator'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def protocol():
    return {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'simulator_repository': str(REPO),
        'simulator_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'source_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip()),
        'drr_revision': '500d7868a84478eb0bafcdee793d1246f492c986',
        'ct_revision': 'deeca4d89e9f978d4d1bccd88a55071ddbb146bb',
        'cohort': 'First 20 distinct validation subjects in natural numeric order; first scan/reconstruction per subject, selected before scoring. All are absent from the upstream non-chest list. AP and LATERAL for every subject: 40 pairs.',
        'pilot': 'valid_1_a_1 was used to verify the source-derived camera conversion. No pose, registration or intensity parameters were fitted to reference images. The pilot also tested the earlier fluoroscopy example mapping; that pilot-only variant is not part of this cohort report.',
        'backend': 'Unmodified public xray_simulator / SlangDiffDRRRenderer, CUDA, NVIDIA RTX A6000',
        'source_generator': {'repository': 'https://github.com/farrell236/midas-journal-784', 'commit': '889727fe0049bf89091f9c2d943299f428ba2a65', 'file': 'getDRRSiddonJacobsRayTracing.cxx'},
        'siddon_implementation_reviewed': {'repository': 'https://github.com/InsightSoftwareConsortium/ITKTwoProjectionRegistration', 'commit': 'fc9714977f5c053a0b68eb5a3761812496ab781b', 'tag': 'v2.0.1'},
        'input_preparation': 'CT-RATE v1: HU = stored * metadata.RescaleSlope + metadata.RescaleIntercept; XY/Z spacing from validation_metadata.csv. Preserve voxel array indexing. The reference generator resets origin and its Siddon traversal ignores direction cosines. Renderer origin is -0.5 * shape_xyz * spacing_xyz; no resampling.',
        'geometry': '512x512; detector pitch 0.51 mm; source-to-projection-plane 1000 mm. Reference transforms: CT translation (0,300,0) mm; Rz=0 AP, -90 deg LATERAL. After upstream camera rotation and output row flip: source relative to CT box center is Rz.T@(0,-1300,0); projection plane center Rz.T@(0,-300,0); columns Rz.T@(1,0,0), rows Rz.T@(0,0,-1). Simulator SID=500 is an equivalent parameterization, translation Rz.T@(0,-800,0).',
        'reference_recipe': 'mu_proxy = max(trunc(HU)+100,0) * 1e-5 /mm; emulate the published -100 HU threshold and signed-short input. This is a renderer-comparison proxy, not calibrated attenuation. Scaling is fixed for every subject.',
        'stock_hu_mapping': 'Public HuToMuMapping() defaults, evaluated by the public hu_to_mu function: -1000 HU -> 0/mm, 3000 HU -> 0.02/mm, linearly interpolated and clipped at those endpoints.',
        'integration_step_mm': 0.5,
        'primary_roi': 'Central 90% detector rectangle (26-pixel margin on every edge at 512x512), fixed before scores; not an anatomy segmentation.',
        'structural_evaluation': 'Published 8-bit DRR PNG /255 versus simulated attenuation /6; attenuation_proxy domain, data_range=1, histogram_range=[0,1], 64 bins. NCC and gradient NCC are primary structural scores. Shape SSIM/RMSE use an in-sample nonnegative gain/bias fit and do not establish calibration.',
        'default_display': 'Actual public xray appearance, fixed log window [0,6]. Evaluate against the published PNG separately from structural metrics.',
        'reference_style_display': 'For reference_recipe only: convert the physical-length integral to the source algorithm alpha-integral by dividing by 1e-5 * source-to-pixel ray length; truncate/clamp to signed short, min/max-scale the generated projection to [0,1], truncate to uint8. This emulates reference export preprocessing, without fitting to a reference image. Record its display scores separately.',
        'controls': 'Independent render after +5 mm along simulator X and +5 degrees about simulator world Z, for both mappings and both views; same reference and ROI.',
        'integration_checks': 'Reference-recipe AP view for every subject also at 0.25 mm. Evaluate relative L2 and RMSE on the same ROI. Assert per-ray requested steps stay within the shader 2048-step limit for every render.',
        'statistics': 'Per-view, per-subject median, AP/LATERAL summaries and median/IQR across subject medians. No pass/fail threshold or population-generalization claim.',
        'limitations': ['DRR-RATE references are synthetic; agreement is consistency with another renderer, not real diagnostic-X-ray validation.', 'No calibrated dose, spectrum, scatter, detector MTF/NPS, noise, temporal or clinical-task validation.', '20 selected subjects, not the complete 3039-reconstruction validation split.', 'Reference uses piecewise-constant Siddon integration; the simulator uses trilinear interpolation and fixed-step integration.', 'Dataset and generated anatomy images stay local.'],
        'sources': ['https://huggingface.co/datasets/farrell236/DRR-RATE', 'https://huggingface.co/datasets/ibrahimhamamci/CT-RATE', 'https://arxiv.org/html/2406.03688v1'],
        'metadata_sha256': digest(DATA / 'ct_rate/dataset/metadata/validation_metadata.csv'),
        'non_chest_list_sha256': digest(DATA / 'ct_rate/dataset/metadata/no_chest_valid.txt'),
        'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'nibabel', 'pillow', 'torch', 'slangpy', 'scikit-image']},
        'source_hashes': {str(p.relative_to(REPO)): digest(p) for p in sorted((REPO / 'xray-simulator/xray_simulator').rglob('*')) if p.suffix in ('.py', '.slang')},
    }


def worker(args):
    import nibabel as nib
    import numpy as np
    from PIL import Image
    from xray_simulator import HuToMuMapping, Pose, SimulatorConfig, xray_simulator
    from xray_simulator.config import CarmGeometry, XrayPhysics
    from xray_simulator.geometry import euler_zxy_to_matrix, matrix_to_euler_zxy, view_matrix
    from xray_simulator.hu_mapping import hu_to_mu
    from xray_simulator.validation import attenuation_from_intensity, evaluate_pair
    from xray_simulator.volume import PreprocessedVolume, VolumeMetadata

    pair = next(r for r in json.loads(COHORT.read_text()) if r['case'] == args.case)
    meta = next(r for r in csv.DictReader((DATA / 'ct_rate/dataset/metadata/validation_metadata.csv').open()) if r['VolumeName'] == args.case + '.nii.gz')
    assert args.case + '.nii.gz' not in (DATA / 'ct_rate/dataset/metadata/no_chest_valid.txt').read_text()
    directory = args.output / args.case
    directory.mkdir()
    image = nib.load(DATA / pair['ct_local_path'])
    hu = image.get_fdata(dtype=np.float32, caching='unchanged')
    hu *= float(meta['RescaleSlope'])
    hu += float(meta['RescaleIntercept'])
    assert np.isfinite(hu).all() and hu.min() >= -32768 and hu.max() <= 32767
    xy = ast.literal_eval(meta['XYSpacing'])
    spacing = np.array([xy[1], xy[0], float(meta['ZSpacing'])])
    assert np.isfinite(spacing).all() and (spacing > 0).all()
    shape = np.array(image.shape)
    origin = -0.5 * shape * spacing
    geometry = CarmGeometry(source_to_detector_mm=1000, source_to_isocenter_mm=500, detector_width_px=512, detector_height_px=512, pixel_spacing_mm=.51)
    mask = np.zeros((512, 512), bool)
    mask[26:-26, 26:-26] = True
    rr, cc = np.mgrid[:512, :512]
    detector_local = np.stack([(cc-255.5)*.51, (rr-255.5)*.51, np.full((512, 512), 1000.)], axis=-1)
    raylength = np.linalg.norm(detector_local, axis=-1)
    dump(directory / 'input.json', {'case': args.case, 'subject': pair['subject'], 'ct_sha256': pair['ct_sha256'], 'ct_revision': pair['ct_revision'],
        'shape_xyz': shape.tolist(), 'nifti_affine_before_metadata': image.affine.tolist(), 'spacing_xyz_mm': spacing.tolist(), 'origin_xyz_mm': origin.tolist(),
        'rescale_slope': float(meta['RescaleSlope']), 'rescale_intercept': float(meta['RescaleIntercept']), 'hu_range': [float(hu.min()), float(hu.max())]})

    def length_check(pose, step):
        rotation = euler_zxy_to_matrix(pose.rotation)
        source = np.array(pose.translation) + rotation @ np.array([0., 0., -500.])
        directions = (detector_local / raylength[..., None]) @ rotation.T
        inv = np.divide(1., directions, out=np.full_like(directions, np.inf), where=np.abs(directions) > 1e-12)
        t0, t1 = (origin-source)*inv, (-origin-source)*inv
        near = np.maximum(np.min(np.stack([t0, t1]), axis=0).max(axis=-1), 0)
        far = np.max(np.stack([t0, t1]), axis=0).min(axis=-1)
        longest = float(np.maximum(far-near, 0).max())
        steps = int(longest/step)+1
        assert steps <= 2048, (args.case, longest, step, steps)
        return {'longest_ray_in_volume_mm': longest, 'maximum_requested_steps': steps, 'shader_limit': 2048}

    def metric(reference, candidate, domain='attenuation_proxy'):
        return evaluate_pair(reference, candidate, mask, domain=domain, data_range=1., histogram_range=(0., 1.))

    records = []
    for model in ['reference_recipe', 'stock_hu_mapping']:
        if model == 'reference_recipe':
            mu = np.maximum(np.trunc(hu)+100, 0)*np.float32(1e-5)
        else:
            mu = hu_to_mu(hu, HuToMuMapping())
        mu = np.ascontiguousarray(mu.transpose(2, 1, 0))
        volume = PreprocessedVolume(mu, VolumeMetadata(shape_zyx=mu.shape, spacing_zyx_mm=tuple(spacing[::-1]), origin_xyz_mm=tuple(origin), mu_range=(float(mu.min()), float(mu.max()))))
        config = SimulatorConfig.for_appearance('xray', geometry=geometry, physics=XrayPhysics(step_mm=.5)).with_output(keep_intensity=True)
        sim = xray_simulator(volume, config)
        for view in ['AP', 'LATERAL']:
            begin = time.monotonic()
            out = directory / view / model
            out.mkdir(parents=True)
            theta = np.deg2rad(0 if view == 'AP' else -90)
            rz = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            rotation = rz.T @ view_matrix('ap')
            pose = Pose(rotation=matrix_to_euler_zxy(rotation), translation=tuple(rz.T @ np.array([0, -800, 0])))
            r5 = euler_zxy_to_matrix((0, 0, float(np.deg2rad(5))))
            control_pose = Pose(rotation=matrix_to_euler_zxy(r5 @ rotation), translation=tuple(np.array(pose.translation) + [5, 0, 0]))
            capacity = {'matched': length_check(pose, .5), 'misposed': length_check(control_pose, .5)}
            frame = sim.render_frame(pose=pose)
            attenuation = attenuation_from_intensity(frame.intensity, frame.i0).astype(np.float32)
            control = sim.render_frame(pose=control_pose)
            control_a = attenuation_from_intensity(control.intensity, control.i0).astype(np.float32)
            path = DATA / pair['ap_image' if view == 'AP' else 'lateral_image']
            png = Image.open(path)
            assert png.mode == 'L' and png.size == (512, 512)
            reference = np.asarray(png, dtype=np.float32)/255.
            scores = metric(reference, attenuation/6)
            misposed = metric(reference, control_a/6)
            preview = (attenuation-attenuation.min())/np.ptp(attenuation)
            arrays = {'reference': reference, 'intensity': frame.intensity, 'attenuation': attenuation,
                      'misposed_attenuation': control_a, 'mask': mask, 'xray_default': frame.image, 'xray_contrast_preview': preview}
            record = {'case': args.case, 'subject': pair['subject'], 'view': view, 'model': model, 'directory': str(out.relative_to(args.output)),
                'reference_sha256': digest(path), 'ct_sha256': pair['ct_sha256'], 'scores': scores, 'misposed_scores': misposed,
                'gradient_ncc_gap': scores['as_configured']['gradient_ncc']-misposed['as_configured']['gradient_ncc'],
                'default_display_scores': metric(reference, frame.image, 'display'), 'geometry': asdict(geometry), 'pose': pose.to_dict(),
                'control_pose': control_pose.to_dict(), 'step_capacity': capacity, 'floor_transmission_fraction': float(np.mean(frame.intensity <= 1e-8))}
            if model == 'reference_recipe':
                alpha_integral = attenuation / (1e-5 * raylength)
                shorts = np.clip(np.trunc(alpha_integral), -32768, 32767)
                matched_display = np.floor((shorts-shorts.min()) / np.ptp(shorts) * 255).astype(np.uint8).astype(np.float32)/255.
                arrays['reference_style_display'] = matched_display
                record['reference_style_display_scores'] = metric(reference, matched_display, 'display')
                record['reference_style_parameters'] = {'short_min': float(shorts.min()), 'short_max': float(shorts.max())}
                if view == 'AP':
                    capacity['fine'] = length_check(pose, .25)
                    fine_sim = xray_simulator(volume, replace(config, physics=XrayPhysics(step_mm=.25)))
                    fine_frame = fine_sim.render_frame(pose=pose)
                    fine_a = attenuation_from_intensity(fine_frame.intensity, fine_frame.i0)
                    difference = attenuation[mask]-fine_a[mask]
                    record['integration_check'] = {'coarse_mm': .5, 'fine_mm': .25, 'rmse': float(np.sqrt(np.mean(difference**2))),
                                                  'relative_l2': float(np.linalg.norm(difference)/np.linalg.norm(fine_a[mask]))}
                    arrays['fine_attenuation'] = fine_a
                    del fine_sim, fine_frame
                    gc.collect()
            for name, values in arrays.items():
                assert values.shape == (512, 512) and np.isfinite(values).all()
                np.save(out / f'{name}.npy', values, allow_pickle=False)
            for name in ['reference', 'mask', 'xray_default', 'xray_contrast_preview', 'reference_style_display']:
                if name in arrays:
                    Image.fromarray(np.rint(np.clip(arrays[name], 0, 1)*255).astype(np.uint8)).save(out / f'{name}.png')
            record['elapsed_seconds'] = time.monotonic()-begin
            dump(out / 'comparison.json', record)
            records.append(record)
            print(f'PROGRESS {args.case} {view} {model} NCC={scores["as_configured"]["ncc"]:.4f} GNCC={scores["as_configured"]["gradient_ncc"]:.4f} control={misposed["as_configured"]["gradient_ncc"]:.4f}', flush=True)
        del sim, volume, mu
        gc.collect()
    dump(directory / 'completed.json', {'case': args.case, 'views': 2, 'models': 2, 'comparisons': len(records)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / 'protocol.json', protocol())
    shutil.copy2(__file__, args.output / 'render_and_compare.py')
    shutil.copy2(COHORT, args.output / 'cohort.json')
    shutil.copy2(Path(__file__).with_name('selection.json'), args.output / 'selection.json')
    failures = []
    for pair in json.loads(COHORT.read_text()):
        case = pair['case']
        print('START ' + case, flush=True)
        command = [sys.executable, '-u', __file__, '--output', str(args.output), '--case', case, '--worker']
        with (args.output / f'{case}.log').open('w') as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith('PROGRESS'):
                    print(line, end='', flush=True)
            code = process.wait()
        if code:
            failures.append({'case': case, 'exit_code': code})
            print(f'FAILED {case} code={code}', flush=True)
        else:
            print('DONE ' + case, flush=True)
    dump(args.output / 'run_status.json', {'failures': failures, 'finished_utc': datetime.now(timezone.utc).isoformat()})
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
