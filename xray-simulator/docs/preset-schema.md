# X-ray Preset Schema & API

Presets are versioned JSON documents describing an X-ray/fluoroscopy simulator
configuration. They cover modality, beam, geometry, detector, and post-processing,
and can also store output and metrics settings. The
[v1 JSON Schema](../xray_simulator/schemas/preset-v1.schema.json) is bundled with the
Python package and uses JSON Schema Draft 2020-12.

## Load, edit, save, and render

Install the base package with `pip install -e ./xray-simulator` from the repository
root. Reading, validating, and saving presets does not import a GPU backend.
Rendering still requires the usual [Slang installation](../README.md#installation).

Run the following from `xray-simulator/`:

```python
from xray_simulator import SimulatorConfig

config = SimulatorConfig.from_preset("examples/presets/fluoroscopy.json")
config = config.with_geometry(detector_width_px=256, detector_height_px=256)
saved_path = config.save_preset("custom-preset.json")
assert SimulatorConfig.from_preset(saved_path) == config
```

Pass the result directly to the rendering API with a preprocessed volume:

```python
from xray_simulator import Pose, PreprocessedVolume, SimulatorConfig, xray_simulator

volume = PreprocessedVolume.load("/path/to/preprocessed/cache")
config = SimulatorConfig.from_preset("custom-preset.json")
simulator = xray_simulator(volume, config=config)
frame = simulator.render_frame(pose=Pose.ap())
```

The [fluoroscopy example](../examples/presets/fluoroscopy.json) enables noise and blur
with dark dense structures. The [radiograph example](../examples/presets/radiograph.json)
disables realism and displays dense structures brightly. Both use `modality: "xray"`.
They are illustrative simulation settings, not calibrated scanner protocols.

## API contract

| API | Result and behavior |
| --- | --- |
| `SimulatorConfig.from_preset(path)` | Read a UTF-8 `.json` file, validate it, fill v1 defaults, and return a `SimulatorConfig`. |
| `config.save_preset(path)` | Validate the configuration, write all settings including defaults, and return an absolute `Path`. |
| `SimulatorConfig.from_dict(data)` | Validate a JSON-compatible dictionary and fill defaults without modifying the input. |
| `config.to_dict()` | Return a fresh JSON-compatible dictionary with all fields explicit; display windows are arrays. |
| `xray_simulator.get_preset_schema()` | Return a fresh dictionary containing the bundled v1 schema, also available in installed wheels. |

Paths accept `str` or `pathlib.Path`. File APIs require the `.json` extension
(case-insensitive); YAML is not supported. Saving validates before writing, then
atomically replaces the destination. The parent directory must already exist.
Malformed documents, duplicate keys, unsupported versions, and invalid values raise
`ValueError`; validation messages identify the relevant field or section. File
loading also includes the preset path in validation errors. File-system failures
such as a missing file or unwritable directory retain their usual `OSError` types.

For dictionary-based applications:

```python
from xray_simulator import SimulatorConfig, get_preset_schema

document = SimulatorConfig().to_dict()
document["detector"]["width_px"] = 1024
config = SimulatorConfig.from_dict(document)
schema = get_preset_schema()
```

## Structure and defaults

The six required top-level fields are `schema_version`, `modality`, `beam`,
`geometry`, `detector`, and `post_processing`. Nested fields may be omitted; the
API fills their defaults from the versioned schema. This is a complete minimal preset:

```json
{
  "schema_version": 1,
  "modality": "xray",
  "beam": {},
  "geometry": {},
  "detector": {},
  "post_processing": {}
}
```

Its configuration matches `SimulatorConfig()` today. The schema stores the v1
defaults explicitly so loading v1 documents does not depend on future dataclass
defaults. Saving always expands defaults. Unknown keys are rejected at every
level, including keys for settings the current renderer does not implement.

All numbers must be finite. Booleans do not count as numbers or integers, and
numeric strings are not coerced. Following JSON Schema semantics, numbers with no
fractional part (for example `512.0`) are accepted for integer fields and converted
to Python integers when constructing detector dimensions or the random seed.

### Modality and version

| Field | Type / allowed value | Required |
| --- | --- | --- |
| `schema_version` | Integer `1` | Yes |
| `modality` | String `"xray"`, covering fluoroscopy and radiography | Yes |

The API rejects unsupported versions and modalities. Fluoroscopy versus radiograph
appearance is selected by `post_processing.display.polarity`. Ultrasound is outside
this schema. Future incompatible formats must use a new schema version.

### Beam

`beam` maps to `XrayPhysics` and the renderer's Beer-Lambert intensity model,
`I = i0 * exp(-integral(mu ds))`.

| Field | Type / constraint | Default | Meaning |
| --- | --- | --- | --- |
| `i0` | Number > 0 | `1.0` | Unattenuated intensity, in arbitrary intensity units. |
| `step_mm` | Number > 0 | `0.5` | Ray-marching integration step in millimeters. Smaller steps increase sampling work. |

The current renderer takes an attenuation volume. It does not model tube voltage,
energy spectra, filtration, or tube current/exposure, so fields such as `kvp` and
`mas` are rejected. `i0` is not a dose or tube-voltage setting. Volume paths, HU-to-μ
mapping, patient orientation, acquisition poses, and cine timing are supplied
through their existing APIs separately from a configuration preset.

### Geometry

`geometry` maps to the distance fields of `CarmGeometry`.

| Field | Type / constraint | Default | Unit |
| --- | --- | --- | --- |
| `source_to_detector_mm` | Number > 0 | `1020.0` | mm, source-to-detector distance (SDD) |
| `source_to_isocenter_mm` | Number > 0 and < SDD | `510.0` | mm, source-to-isocenter distance (SID) |

The API checks `SID < SDD` after filling defaults. Magnification at the isocenter
is `SDD / SID`; view orientation and translation are supplied as a `Pose` at render time.

### Detector

| Field | Type / constraint | Default | Meaning |
| --- | --- | --- | --- |
| `width_px` | Integer ≥ 1 | `512` | Columns; maps to `CarmGeometry.detector_width_px`. |
| `height_px` | Integer ≥ 1 | `512` | Rows; maps to `CarmGeometry.detector_height_px`. |
| `pixel_spacing_mm` | Number > 0 | `0.5` | Square pixel pitch in mm; maps to `CarmGeometry.pixel_spacing_mm`. |

Physical detector width and height are pixel count multiplied by pixel spacing.
Rendered arrays have shape `(height_px, width_px)`.

### Post-processing: realism

`post_processing.realism` maps to `RealismSettings`. When enabled, the existing
pipeline applies gain and bias, clips negative intensity, adds Poisson noise, adds
Gaussian noise with nonnegative clipping, then applies Gaussian blur. Display
mapping follows realism. When disabled, all realism operations are skipped.

| Field | Type / constraint | Default | Meaning |
| --- | --- | --- | --- |
| `enabled` | Boolean | `false` | Enable this stage. |
| `gain` | Number ≥ 0 | `1.0` | Intensity multiplier. |
| `bias` | Number | `0.0` | Offset in intensity units. |
| `poisson_photons` | Number ≥ 0 | `0.0` | Expected photon count at intensity **1**, not at `i0`; zero disables Poisson noise. |
| `gaussian_sigma` | Number ≥ 0 | `0.0` | Additive noise standard deviation in intensity units; zero disables it. |
| `blur_sigma_px` | Number ≥ 0 | `0.0` | Gaussian blur standard deviation in pixels; zero disables it. |
| `seed` | Integer ≥ 0 or `null` | `0` | Random seed, or fresh randomness when null. |

The existing realism implementation creates a random generator for each frame.
A fixed seed therefore repeats the random sequence for each invocation, rather
than advancing one generator across a cine sequence. Presets preserve that behavior.

### Post-processing: display

`post_processing.display` maps to `DisplaySettings`.

| Field | Type / constraint | Default | Meaning |
| --- | --- | --- | --- |
| `polarity` | `"fluoro"` or `"diagnostic"` | `"fluoro"` | Dense structures dark or bright, respectively. |
| `scaling` | `"log"`, `"transmission"`, `"window"`, or `"per_frame"` | `"log"` | Intensity-to-display mapping. |
| `log_window` | Two numbers `[low, high]`, 0 ≤ low < high | `[0.0, 6.0]` | Dimensionless optical-depth interval for log scaling. |
| `window` | Two numbers `[low, high]`, 0 ≤ low < high ≤ 1 | `[0.0, 1.0]` | Transmission interval for window scaling. |
| `gamma` | Number > 0 | `1.0` | Display gamma; values above one brighten midtones. |

Log scaling uses `-ln(I / i0)`. Transmission scaling uses `I / i0`; window scaling
stretches the chosen transmission interval. Per-frame scaling uses each frame's
minimum and maximum, so it can change brightness across a cine sequence. Both
window pairs are validated even when their scaling mode is inactive. See
[Image Appearance](architecture-and-api.md#image-appearance) for the display model.

The appearance alias `"xray"` from `SimulatorConfig.for_appearance()` is serialized
as the canonical polarity `"diagnostic"`. The preset format accepts only canonical
polarities. Deprecated `physics.normalize` and `physics.invert` are not preset
fields; saving a legacy configuration resolves them into effective display settings
and emits the existing deprecation warning. Loading sets the legacy flags to `None`.

### Optional runtime settings

These top-level fields can be omitted entirely. They allow a preset to preserve
all supported `SimulatorConfig` settings.

| Field | Type / constraint | Default | Meaning |
| --- | --- | --- | --- |
| `backend` | `"slang"` | `"slang"` | Rendering backend; not initialized by preset loading. |
| `output.save_to_disk` | Boolean | `false` | Save rendered frames automatically. |
| `output.output_dir` | Nonempty string or `null` | `null` | Destination directory; a string is required when `save_to_disk` is true. |
| `output.format` | `"png"`, `"npy"`, or `"npz"` | `"png"` | Frame file format. |
| `output.keep_in_gpu` | Boolean | `false` | Existing `OutputSettings` flag; actual residency follows renderer capabilities. |
| `output.keep_intensity` | Boolean | `false` | Retain intensity before display mapping for later remapping. |
| `metrics.enabled` | Boolean | `false` | Enable metrics collection. |
| `metrics.track_fps` | Boolean | `true` | Track frame rate when metrics are enabled. |
| `metrics.track_gpu_usage` | Boolean | `true` | Track GPU usage when metrics are enabled. |
| `metrics.track_jitter` | Boolean | `true` | Track frame-time jitter when metrics are enabled. |

`output` maps to `OutputSettings`; `metrics` maps to `MetricsSettings`. Relative
output directories are interpreted against the process working directory, just as
with a manually constructed config, rather than the preset's directory. A Python
`Path` output directory is saved as a JSON string and loaded as a string. Loading a
preset creates no output directories and writes no frames.

## Validation outside Python

Use the bundled schema with a Draft 2020-12 validator for field types, allowed keys,
ranges, and required sections. For example:

```python
import json
from jsonschema import Draft202012Validator
from xray_simulator import SimulatorConfig, get_preset_schema

with open("custom-preset.json", encoding="utf-8") as stream:
    document = json.load(stream)
Draft202012Validator(get_preset_schema()).validate(document)
config = SimulatorConfig.from_dict(document)  # Fill defaults and check field relationships.
```

Standard schema validators do not insert defaults. Portable JSON Schema also does
not express `SID < SDD` or compare the two window entries; these checks run in the
Python API. The API additionally rejects non-finite values, and the file loader
rejects duplicate keys that a typical JSON parser would silently overwrite. Use
`SimulatorConfig.from_preset()` for complete file validation before rendering.
