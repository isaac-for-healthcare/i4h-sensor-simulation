"""Render matched xvr-data views with the public i4h simulator and retain comparisons."""
from __future__ import annotations

import argparse
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

os.environ.setdefault('MPLCONFIGDIR', '/tmp/i4h-paired-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.dont_write_bytecode = True
REPO = Path(os.environ['I4H_SENSOR_SIMULATION_REPO']).resolve()
DATA = Path(os.environ['I4H_VALIDATION_DATA_ROOT']).resolve()
sys.path.insert(0, str(REPO / 'xray-simulator'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def protocol():
    return {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'simulator_repository': str(REPO),
        'simulator_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'source_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip()),
        'data_root': str(DATA),
        'data_revision': 'a17273e3eadbd793bd861f3598a80ce4590c1124',
        'dataset_inventory_sha256': digest(DATA.parent / 'manifest.json') if (DATA.parent / 'manifest.json').is_file() else None,
        'backend': 'public xray_simulator.xray_simulator / SlangDiffDRRRenderer, CUDA',
        'binning': 4, 'integration_step_mm': 0.5,
        'views': 'all 362 eligible DeepFluoro and 20 Ljubljana primary views; exclude four upstream-flagged poses',
        'deepfluoro_volume': 'unchanged example mapping: HU window center 200, width 1600, mu_max 0.05/mm',
        'ljubljana_volume': 'explicit uncalibrated vessel-concentration model: mu_proxy/mm = max(volume_value, 0) * 5e-6; no HU interpretation; same coefficient for all ten subjects',
        'reference_grid': 'dataset adapter crop and 4x4 block mean; matched pose and independent x/y pitch; no registration or resizing',
        'primary_roi': 'central 90% rectangle of the cropped detector (5% margin each edge), fixed before scoring; detector ROI, not an anatomy segmentation',
        'secondary_ljubljana_roi': 'primary ROI intersected with a 10-pixel dilation of aligned projected vessel support (attenuation > 1e-4); mask frozen for candidate/control; model-defined ROI, not an independent annotation',
        'reference_display': 'stored unsigned 16-bit DICOM / 65535, no window fitting; diagnostic counterpart is 1-reference_display',
        'deepfluoro_reference_proxy': 's=(reference-ROI_min)/(ROI_max-ROI_min), clipped to [0,1]; proxy=log(2/(1+s))/log(2), following xvr log-linearization with an explicit final unit scale; not calibrated attenuation',
        'ljubljana_reference_proxy': 's=(reference-ROI_min)/(ROI_max-ROI_min), clipped to [0,1]; proxy=1-s because primary references are subtraction angiograms; no second logarithm',
        'proxy_evaluation': 'reference proxy vs simulated line integral / 6 (the fixed default display-window width), without clipping; data_range=1, histogram_range=[0,1], 64 bins. Primary structural scores NCC/gradient NCC. Affine-adjusted shape-only SSIM/RMSE are in-sample descriptive fits; raw cross-scale errors are not physical calibration errors.',
        'display_evaluation': 'fixed default log window [0,6] and separately a subject window derived solely from the first eligible generated view (1st/99th attenuation percentiles), then frozen. Both X-ray and fluoro use the same transport with opposite polarity.',
        'controls': 'independent render with +5 mm along simulator X and +5 degrees about simulator world Z; all masks/reference preparation unchanged',
        'integration_check': 'first eligible view per subject also rendered at 0.25 mm; compare line integral with 0.5 mm (numerical consistency, not external validation)',
        'ljubljana_offset_diagnostic': 'additional exploratory render negates only the horizontal detector offset from the current adapter. The pilot had an approximately 81-pixel horizontal displacement, equal to twice the stored x offset / binned pitch. Keep branch-as-is results primary; use this deterministic metadata-sign variant without image registration or pose fitting, with its own misposed control. This is not a source-code patch or a validated convention correction.',
        'statistics': 'per-view and per-subject median/IQR; dataset summary median across subject medians. No acquisition bootstrap because independence metadata is not established.',
        'limitations': ['No patient images are published or committed.', 'No physical scanner calibration, noise/dose model, detector lag, temporal sequence or registration refinement.', 'Unmatched instruments, collimation, truncation and acquisition processing can affect scores.', 'Ljubljana vessel volume and reference contrast filling may differ.'],
        'sources': [
            'https://huggingface.co/datasets/eigenvivek/xvr-data',
            'https://github.com/eigenvivek/xvr/blob/caa55cc8096294cf70a218126bf16008dee0dec7/src/xvr/io/xray.py',
            'https://lit.fe.uni-lj.si/en/research/resources/3D-2D-GS-CA/',
        ],
        'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'torch', 'slangpy', 'scikit-image', 'nibabel', 'pydicom', 'pillow']},
        'source_hashes': {str(p.relative_to(REPO)): digest(p) for p in sorted((REPO / 'xray-simulator/xray_simulator').rglob('*')) if p.suffix in ('.py', '.slang')},
    }


def worker(args):
    import gc

    import numpy as np
    import pydicom
    import torch
    from PIL import Image
    from scipy.ndimage import binary_dilation
    from xray_simulator import (
        HuToMuMapping,
        Pose,
        PreprocessingSettings,
        SimulatorConfig,
        VolumePreprocessor,
        xray_simulator,
    )
    from xray_simulator.config import XrayPhysics
    from xray_simulator.display import calibrate_display
    from xray_simulator.geometry import euler_zxy_to_matrix, matrix_to_euler_zxy
    from xray_simulator.validation import attenuation_from_intensity, evaluate_pair
    from xray_simulator.validation.xvr import EXCLUDED_VIEWS, load_xvr_view, load_xvr_volume
    from xray_simulator.volume import PreprocessedVolume, VolumeMetadata

    torch.set_num_threads(1)
    sub = DATA / args.dataset / args.subject
    out = args.output / args.dataset / args.subject
    out.mkdir(parents=True, exist_ok=True)
    values, frame = load_xvr_volume(sub / 'volume.nii.gz')
    if args.dataset == 'deepfluoro':
        settings = PreprocessingSettings(hu_to_mu=HuToMuMapping.from_window_level(200, 1600, 0.05))
        volume = VolumePreprocessor(values, frame.spacing_zyx_mm, origin_xyz_mm=frame.origin_xyz_mm,
                                    settings=settings, source=str(sub / 'volume.nii.gz')).preprocess()
        volume_model = asdict(settings)
    else:
        mu = np.maximum(values, 0) * np.float32(5e-6)
        volume = PreprocessedVolume(mu, VolumeMetadata(
            shape_zyx=mu.shape, spacing_zyx_mm=frame.spacing_zyx_mm, origin_xyz_mm=frame.origin_xyz_mm,
            source=str(sub / 'volume.nii.gz'), mu_range=(float(mu.min()), float(mu.max())),
        ))
        volume_model = {'model': 'linear nonnegative 3D-DSA vessel contrast proxy', 'coefficient_per_mm_per_stored_unit': 5e-6,
                        'input_range': [float(values.min()), float(values.max())], 'input_is_HU': False}
        del mu
    del values
    write_json(out / 'subject.json', {'dataset': args.dataset, 'subject': args.subject, 'volume_model': volume_model,
                                    'volume_sha256': digest(sub / 'volume.nii.gz'), 'frame': frame.to_dict()})
    paths = [p for p in sorted((sub / 'xrays').glob('*.pt'))
             if args.dataset != 'deepfluoro' or (args.subject, p.stem) not in EXCLUDED_VIEWS]
    if args.limit:
        paths = paths[:args.limit]
    sim = None
    previous_geometry = None
    calibrated = None
    all_records = []
    for index, path in enumerate(paths):
        begin = time.monotonic()
        view = load_xvr_view(sub, path.stem, dataset=args.dataset)
        cam = view.camera(frame, binning=4)
        directory = out / path.stem
        directory.mkdir(exist_ok=True)
        reference = view.load_reference(binning=4).astype(np.float32)
        ds = pydicom.dcmread(view.reference_path, stop_before_pixels=True)
        if int(ds.BitsStored) != 16 or int(ds.PixelRepresentation) != 0:
            raise ValueError('Protocol expects unsigned 16-bit references')
        if cam.geometry != previous_geometry:
            if sim is not None:
                del sim
                gc.collect()
            config = SimulatorConfig.for_appearance('fluoro', geometry=cam.geometry, physics=XrayPhysics(step_mm=0.5)).with_output(keep_intensity=True)
            sim = xray_simulator(volume, config)
            previous_geometry = cam.geometry
        rendered = sim.render_frame(pose=cam.pose)
        attenuation = attenuation_from_intensity(rendered.intensity, rendered.i0).astype(np.float32)
        if not np.isfinite(attenuation).all() or np.ptp(attenuation) == 0:
            raise ValueError('Nonfinite/empty matched render')
        if calibrated is None:
            calibrated = calibrate_display(rendered.intensity, i0=rendered.i0)
            write_json(out / 'display_calibration.json', {'source_view': path.stem, 'settings': asdict(calibrated),
                                                         'fitted_to_real_reference': False})
        angle = np.deg2rad(5.0)
        rz = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        control_pose = Pose(rotation=matrix_to_euler_zxy(rz @ euler_zxy_to_matrix(cam.pose.rotation)),
                            translation=tuple(np.array(cam.pose.translation) + [5.0, 0.0, 0.0]))
        control = sim.render_frame(pose=control_pose)
        control_a = attenuation_from_intensity(control.intensity, control.i0).astype(np.float32)
        h, w = reference.shape
        mask = np.zeros((h, w), bool)
        my, mx = int(np.ceil(0.05 * h)), int(np.ceil(0.05 * w))
        mask[my:h-my, mx:w-mx] = True
        low, high = float(reference[mask].min()), float(reference[mask].max())
        if high <= low:
            raise ValueError('Constant reference ROI')
        standardized = np.clip((reference - low) / (high - low), 0, 1)
        reference_proxy = (np.log(2 / (1 + standardized)) / np.log(2) if args.dataset == 'deepfluoro' else 1 - standardized).astype(np.float32)
        ref_display = reference / 65535.0
        display_images = {
            'fluoro_default': rendered.image,
            'xray_default': rendered.with_appearance('xray').image,
            'fluoro_subject_window': rendered.with_appearance(calibrated).image,
            'xray_subject_window': rendered.with_appearance(replace(calibrated, polarity='diagnostic')).image,
        }
        np.testing.assert_allclose(display_images['xray_default'] + display_images['fluoro_default'], 1, atol=2e-7)
        arrays = {'reference_stored': reference, 'reference_display': ref_display, 'reference_proxy': reference_proxy,
                  'rendered_intensity': rendered.intensity, 'rendered_attenuation': attenuation,
                  'misposed_attenuation': control_a, 'mask': mask, **display_images}
        scores = evaluate_pair(reference_proxy, attenuation / 6, mask, domain='attenuation_proxy', data_range=1., histogram_range=(0., 1.))
        control_scores = evaluate_pair(reference_proxy, control_a / 6, mask, domain='attenuation_proxy', data_range=1., histogram_range=(0., 1.))
        record = {'dataset': args.dataset, 'subject': args.subject, 'view': path.stem,
                  'directory': str(directory.relative_to(args.output)), 'scores': scores, 'misposed_scores': control_scores,
                  'gradient_ncc_gap': scores['as_configured']['gradient_ncc'] - control_scores['as_configured']['gradient_ncc'],
                  'reference_preparation': {'roi_min': low, 'roi_max': high, 'bits': int(ds.BitsStored)},
                  'camera': cam.to_dict(), 'control_pose': control_pose.to_dict(),
                  'reference_sha256': digest(view.reference_path), 'calibration_sha256': digest(view.calibration_path),
                  'render_ms': float(rendered.timestamp_ms), 'display_scores': {}}
        for name, arr in display_images.items():
            ref = 1 - ref_display if name.startswith('xray') else ref_display
            record['display_scores'][name] = evaluate_pair(ref, arr, mask, domain='display', data_range=1., histogram_range=(0., 1.))
        if args.dataset == 'ljubljana':
            vessel_roi = mask & binary_dilation(attenuation > 1e-4, structure=np.ones((21, 21), bool))
            arrays['vessel_roi'] = vessel_roi
            record['vessel_roi_scores'] = evaluate_pair(reference_proxy, attenuation / 6, vessel_roi, domain='attenuation_proxy', data_range=1., histogram_range=(0., 1.))
            record['vessel_roi_misposed_scores'] = evaluate_pair(reference_proxy, control_a / 6, vessel_roi, domain='attenuation_proxy', data_range=1., histogram_range=(0., 1.))
            xoff, yoff = cam.geometry.detector_offset_xy_mm
            diagnostic_geometry = replace(cam.geometry, detector_offset_xy_mm=(-xoff, yoff))
            diagnostic = xray_simulator(volume, replace(config, geometry=diagnostic_geometry))
            diagnostic_frame = diagnostic.render_frame(pose=cam.pose)
            diagnostic_a = attenuation_from_intensity(diagnostic_frame.intensity, diagnostic_frame.i0).astype(np.float32)
            diagnostic_control = diagnostic.render_frame(pose=control_pose)
            diagnostic_control_a = attenuation_from_intensity(diagnostic_control.intensity, diagnostic_control.i0).astype(np.float32)
            diagnostic_roi = mask & binary_dilation(diagnostic_a > 1e-4, structure=np.ones((21, 21), bool))
            record['offset_diagnostic'] = {
                'geometry': asdict(diagnostic_geometry),
                'predicted_column_shift_px': float(2 * xoff / cam.geometry.pixel_spacing_mm),
                'scores': evaluate_pair(reference_proxy, diagnostic_a / 6, mask, domain='attenuation_proxy', data_range=1., histogram_range=(0., 1.)),
                'misposed_scores': evaluate_pair(reference_proxy, diagnostic_control_a / 6, mask, domain='attenuation_proxy', data_range=1., histogram_range=(0., 1.)),
                'vessel_roi_scores': evaluate_pair(reference_proxy, diagnostic_a / 6, diagnostic_roi, domain='attenuation_proxy', data_range=1., histogram_range=(0., 1.)),
            }
            arrays.update(offset_diagnostic_attenuation=diagnostic_a, offset_diagnostic_misposed_attenuation=diagnostic_control_a,
                          offset_diagnostic_vessel_roi=diagnostic_roi)
            display_images.update(offset_diagnostic_fluoro=diagnostic_frame.with_appearance(calibrated).image,
                                  offset_diagnostic_xray=diagnostic_frame.with_appearance(replace(calibrated, polarity='diagnostic')).image)
            arrays.update({k: v for k, v in display_images.items() if k.startswith('offset_diagnostic')})
            del diagnostic, diagnostic_frame, diagnostic_control
            gc.collect()
        if index == 0:
            fine = xray_simulator(volume, replace(config, physics=XrayPhysics(step_mm=0.25)))
            fine_frame = fine.render_frame(pose=cam.pose)
            fine_a = attenuation_from_intensity(fine_frame.intensity, fine_frame.i0)
            diff = attenuation[mask] - fine_a[mask]
            record['integration_check'] = {'coarse_mm': 0.5, 'fine_mm': 0.25,
                'rmse': float(np.sqrt(np.mean(diff**2))),
                'relative_l2': float(np.linalg.norm(diff) / max(np.linalg.norm(fine_a[mask]), 1e-12))}
            arrays['fine_attenuation'] = fine_a.astype(np.float32)
            del fine, fine_frame
            gc.collect()
        for name, arr in arrays.items():
            np.save(directory / f'{name}.npy', arr, allow_pickle=False)
        pngs = {'reference_stored_display': ref_display, 'reference_contrast_preview': standardized,
                'reference_proxy': reference_proxy, 'mask': mask, **display_images}
        if 'vessel_roi' in arrays:
            pngs['vessel_roi'] = arrays['vessel_roi']
        for name, arr in pngs.items():
            Image.fromarray(np.rint(np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(directory / f'{name}.png')
        record['elapsed_seconds'] = time.monotonic() - begin
        write_json(directory / 'comparison.json', record)
        all_records.append(record)
        if index % 10 == 0 or index + 1 == len(paths):
            print(f'PROGRESS {args.dataset}/{args.subject} {index+1}/{len(paths)} view={path.stem} NCC={scores["as_configured"]["ncc"]:.3f} GNCC={scores["as_configured"]["gradient_ncc"]:.3f} control={control_scores["as_configured"]["gradient_ncc"]:.3f} seconds={record["elapsed_seconds"]:.2f}', flush=True)
    write_json(out / 'completed.json', {'views': len(all_records), 'views_list': [r['view'] for r in all_records]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', choices=['deepfluoro', 'ljubljana'])
    parser.add_argument('--subject')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        if digest(__file__) != digest(args.output / 'render_and_compare.py'):
            raise ValueError('Resume requires the unchanged benchmark script')
    else:
        write_json(args.output / 'protocol.json', protocol())
        shutil.copy2(__file__, args.output / 'render_and_compare.py')
    datasets = [args.dataset] if args.dataset else ['deepfluoro', 'ljubljana']
    failures = []
    for dataset in datasets:
        subjects = [args.subject] if args.subject else [p.name for p in sorted((DATA / dataset).glob('subject*'))]
        for subject in subjects:
            if args.resume and (args.output / dataset / subject / 'completed.json').exists():
                print(f'SKIP completed {dataset}/{subject}', flush=True)
                continue
            cmd = [sys.executable, '-u', __file__, '--output', str(args.output), '--dataset', dataset, '--subject', subject, '--worker']
            if args.limit:
                cmd += ['--limit', str(args.limit)]
            print(f'START {dataset}/{subject}', flush=True)
            with (args.output / f'{dataset}_{subject}.log').open('w') as log:
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith('PROGRESS'):
                        print(line, end='', flush=True)
                result = process.wait()
            if result:
                failures.append({'dataset': dataset, 'subject': subject, 'exit_code': result})
                print(f'FAILED {dataset}/{subject} code={result}', flush=True)
            else:
                print(f'DONE {dataset}/{subject}', flush=True)
    write_json(args.output / 'run_status.json', {'failures': failures, 'finished_utc': datetime.now(timezone.utc).isoformat()})
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
