# Launching X-ray simulation from JSON/YAML configuration

Configure scanner parameters through the Python API or a
[versioned JSON/YAML preset](preset-schema.md). The optional `xray-simulator`
command (also `python -m xray_simulator`) reads that preset and launches
preprocessing or rendering. Both entry points use `SimulatorConfig.from_preset()`.

## Quick start

From the repository root:

```bash
python -m pip install -e './xray-simulator[cli,slang]'
python -m xray_simulator render --help

# Validate and preview without loading voxels, initializing a GPU, or writing output.
python -m xray_simulator render \
  --config xray-simulator/examples/presets/fluoroscopy.yaml \
  --synthetic --view ap --output output/fluoroscopy --dryrun

# Render with the same configuration.
python -m xray_simulator render \
  --config xray-simulator/examples/presets/fluoroscopy.yaml \
  --synthetic --view ap --output output/fluoroscopy
```

Use a new output directory for each run. `.json`, `.yaml`, and `.yml` presets
are interchangeable; matching fluoroscopy and radiograph examples are in
`examples/presets/`. `--dryrun` validates settings and source paths but cannot
prove that a source can be decoded or that CUDA is available.

## Python API

The launcher is optional. With a raw CT volume, pass the preset's preprocessing
settings explicitly:

```python
from xray_simulator import Pose, SimulatorConfig, VolumePreprocessor, xray_simulator

config = SimulatorConfig.from_preset("xray-simulator/examples/presets/fluoroscopy.yaml")
config = config.with_display(gamma=1.2)
volume = VolumePreprocessor.from_nifti("ct.nii.gz", settings=config.preprocessing).preprocess()
simulator = xray_simulator(volume, config=config)
frame = simulator.render_frame(pose=Pose(rotation=(0.0, 0.0, 0.0)))
frame.save("frame.npy")
config.save_preset("custom-scanner.json")
```

NIfTI input requires nibabel; DICOM input requires SimpleITK. Both are available
in the package's `all` extra. The native-axis loaders retain the source orientation.
Named clinical views require LPS volume metadata, so prepare an LPS cache externally
or use explicitly chosen rotations in the native array frame. The synthetic
phantom is already labeled LPS.

## Preset controls

| Parameter | Preset field |
| --- | --- |
| Fluoroscopy/radiograph polarity | `post_processing.display.polarity`: `fluoro` / `diagnostic` |
| Scaling, windowing, gamma | `post_processing.display.scaling`, `log_window`, `window`, `gamma` |
| Gain and offset | `post_processing.realism.gain`, `bias` |
| Noise and detector blur | `post_processing.realism.poisson_photons`, `gaussian_sigma`, `blur_sigma_px`, `seed` |
| Enable intensity effects | `post_processing.realism.enabled` |
| Incident intensity and sampling | `beam.i0`, `beam.step_mm` |
| Geometry | `geometry.source_to_detector_mm`, `geometry.source_to_isocenter_mm` |
| Detector | `detector.width_px`, `detector.height_px`, `detector.pixel_spacing_mm` |
| HU clipping and transfer function | `preprocessing.clip_hu`, `hu_clip_min`, `hu_clip_max`, `hu_to_mu` |
| Image format and retained intensity | `output.format`, `output.keep_intensity` |

Set `realism.enabled: true` to apply gain/bias, Poisson noise, Gaussian noise, then
blur before display mapping. The same explicit enable flag applies in the API
and launcher. `i0` is an uncalibrated intensity surrogate. Noise and blur expose
existing approximations; scatter, persistence, sharpening, physical detector ADC
and calibrated dose/AEC are not implemented. See the [schema reference](preset-schema.md)
for all fields, units, defaults, and validation rules.

## Launcher controls

The launcher accepts only the configuration path and run-specific inputs/actions:

| Argument | Behavior |
| --- | --- |
| `--config FILE` | Required JSON/YAML preset. |
| `--synthetic`, `--nifti FILE`, `--dicom DIR`, `--cache DIR` | Choose exactly one input; cache input is render-only. |
| `--output DIR` | New destination; overrides `output.output_dir` from the preset. |
| `--view NAME` | Named clinical view for LPS volumes. |
| `--rotation-deg RX RY RZ` | Raw Euler rotation (ZXY convention), mutually exclusive with `--view`; defaults to zero. |
| `--translation-mm X Y Z` | Isocenter displacement from volume center. |
| `--frames N`, `--fps RATE` | Repeat the pose; FPS is sequence metadata, not exposure or pacing. |
| `--calibrate-display LOW HIGH` | Fit and freeze a log window using these first-view percentiles; default is no calibration. |
| `--dryrun` / `--dry-run` | Validate and print the resolved plan. |

Paths, including a relative `output.output_dir`, are interpreted from the process
working directory. A preset alone does not select a patient, volume, or pose.
Only `render` accepts pose, sequence, and display-calibration arguments. Scanner
flags such as `--gamma`, `--gain`, and `--hu-clip` have been replaced by preset fields.

## Preprocessing and caches

The same preset configures the preprocessing API and launcher:

```bash
python -m xray_simulator preprocess \
  --config xray-simulator/examples/presets/fluoroscopy.json \
  --synthetic --output output/cache

python -m xray_simulator render \
  --config xray-simulator/examples/presets/fluoroscopy.yaml \
  --cache output/cache --view ap --output output/cached-frames
```

A cache has already been converted from HU to attenuation. The launcher rejects
nondefault `preprocessing` settings when rendering a cache so requested mappings
are never silently ignored. To change the mapping, preprocess the raw source again;
then use a preset with the default preprocessing section (or omit that section)
for the cached render. The cache metadata retains its original mapping.

## Output and reproducibility

`output.format` selects `npy` (float32), `npz` (compressed, array key `image`), or
`png` (8-bit grayscale; requires the `cli` extra). `output.keep_intensity: true`
adds float32 `intensity_XXXX.npy` files containing intensity after realism and
before display mapping. The example presets select NPY and retain intensity.

The launcher always exports into its new output directory, regardless of the
preset's `save_to_disk` flag. It disables the simulator's automatic frame writer
to keep a single export path. `run.json` records the effective reusable `preset`,
input, preprocessing, volume metadata, pose, frame count, and FPS. Calibration
updates the recorded display window. Fixed seeds reproduce a run while advancing
per frame; a null seed requests fresh randomness. Independent draws do not model
temporal detector persistence.

## Direct Docker usage

Build from the repository root and bind an output directory:

```bash
docker build -t xray_simulator -f xray-simulator/Dockerfile xray-simulator
mkdir -p output
docker run --rm --gpus all -v "$PWD/output:/output" xray_simulator \
  python -m xray_simulator render \
  --config examples/presets/fluoroscopy.yaml \
  --synthetic --view ap --output /output/fluoroscopy
```

Mount a custom preset read-only when needed, for example
`-v "$PWD/custom.yaml:/config/scanner.yaml:ro"`, and pass
`--config /config/scanner.yaml`. Inputs must also use container-visible paths.
