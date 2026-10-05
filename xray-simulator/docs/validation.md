# X-ray and fluoroscopy image validation

The validation module measures agreement between explicitly paired images of the
**same subject, anatomy, projection and detector grid**. It runs on the CPU and
does not require a renderer, GPU, network connection or downloaded dataset.

This is evaluation tooling, not a claim that the simulator reproduces a clinical
scanner. The repository includes analytical metric tests and a
[paired DeepFluoro/Ljubljana comparison report](reports/deepfluoro-ljubljana/README.md)
covering 382 matched views, with the dataset geometry conventions it relies on
and the changes since its previous version. A separate
[DRR-RATE X-ray comparison report](reports/drr-rate/README.md) covers 40 AP/lateral
views from 20 CT subjects against an independent synthetic renderer, with both
the matched DRR-RATE signal model and the stock simulator HU mapping. It includes
misposed controls and integration-step checks. Dataset images remain local;
these structural comparisons do not establish physical or clinical validation.

## Install and run

From the repository root:

```bash
python -m pip install -e './xray-simulator[validation]'
python -m xray_simulator.validation pairs.json --output results/report.json --dryrun
python -m xray_simulator.validation pairs.json --output results/report.json
```

The preview checks manifest structure and file existence. The actual run also
checks pixel values, masks and dimensions. It never resizes, registers, inverts,
or independently normalizes images. An existing report is never overwritten.

Use this manifest structure, with paths relative to the manifest:

```json
{
  "schema_version": 1,
  "domain": "attenuation_proxy",
  "data_range": 1.0,
  "histogram_range": [0.0, 1.0],
  "bins": 64,
  "provenance": {
    "dataset_revision": "record the exact dataset revision",
    "simulator_commit": "record the renderer commit",
    "preprocessing": "record crop, polarity, intensity mapping and mask construction",
    "geometry": "record CT orientation/affine, pose convention, SDD, detector pitch and principal point",
    "render_settings": "record HU-to-mu mapping, step size, display settings and noise seeds"
  },
  "pairs": [
    {
      "dataset": "example-dataset",
      "subject": "subject01",
      "view": "000",
      "reference": "prepared/reference.npy",
      "rendered": "prepared/rendered.npy",
      "mask": "prepared/anatomy_mask.npy"
    }
  ]
}
```

The identifiers declare the intended pairing; matching dimensions cannot prove
that the subject and camera are actually matched. The caller must establish that
correspondence. Use one manifest per experiment, signal domain and fixed metric
scale. Do not mix display presets or ablation variants into a subject summary.
The full manifest, input SHA-256 hashes, metric implementation hashes and library
versions are retained in the report.

Optional per-pair fields:

| Field | Meaning |
| --- | --- |
| `misposed` | Path to a render with a documented 3D pose perturbation. The report includes its scores and the aligned-minus-misposed gradient NCC difference. |
| `acquisition` | Identifier of an independent acquisition within this subject. All frames from the same correlated cine sequence share an ID. |

The report contains every image's scores and median, quartiles and IQR separately
for each dataset/subject. When every view has an acquisition ID and at least two
independent acquisitions are present, it also computes a 95% percentile bootstrap
interval for the median by resampling whole acquisitions (2,000 draws, `--seed 0`
by default). This interval is conditional on that subject. With fewer groups or
undefined scores, the corresponding interval is `null`. Small numbers of
acquisitions give unstable intervals; do not interpret a narrow interval from
repeated frames as evidence about a patient population. Subjects are never pooled
into a single score.

## Signal domains and export

| Domain | Input | Reports |
| --- | --- | --- |
| `display` | Prepared float arrays in [0, 1], or 8-bit grayscale PNGs | As-configured scores in normalized grayscale; no attenuation fit or radial attenuation profile |
| `attenuation` | Float arrays of calibrated `A = -ln(I / I0)` | As-configured and shape-only scores in dimensionless line-integral units |
| `attenuation_proxy` | Explicitly prepared, consistently oriented attenuation-like arrays with uncalibrated scale | The same two evaluations, labeled arbitrary proxy units |

Array inputs are 2D `.npy` files loaded with `allow_pickle=False`. The fixed anatomy
mask is a boolean/0–1 `.npy` array or a black/white PNG. RGB/RGBA PNGs are accepted
only when all color channels match and alpha is opaque. Colored overlays are
rejected. Grayscale captions, collimator borders and baked-in markers still need
to be excluded by the caller. For 16-bit images and DICOM, perform an explicit
preparation step; the runner does not guess rescale slopes, modality LUTs,
photometric interpretation or window levels.

To retain simulator intensity alongside its displayed frame:

```python
import numpy as np

from xray_simulator import Pose, SimulatorConfig, xray_simulator
from xray_simulator.validation import attenuation_from_intensity

# `volume` is a preprocessed volume in the canonical patient frame.
config = SimulatorConfig.for_appearance("fluoro").with_output(keep_intensity=True)
simulator = xray_simulator(volume, config)
frame = simulator.render_frame(pose=Pose.ap())

np.save("rendered_display.npy", frame.image)
np.save("rendered_intensity.npy", frame.intensity)
np.save("rendered_attenuation.npy", attenuation_from_intensity(frame.intensity, frame.i0))
```

This illustrates export only: a benchmark comparison must use its calibrated
pose and intrinsics instead of the default AP pose and detector. Record the
configuration and CT affine with the export. Retained intensity is after enabled
realism effects and before display processing. Disable realism for the baseline
transport check and record each enabled effect for ablations.

`attenuation_from_intensity` floors transmission at `1e-8` by default; record any
changed floor and inspect how many rays reach it. Intensity above `I0` gives
negative attenuation and is not silently clipped. A windowed or min/max-normalized
PNG cannot be converted back to calibrated attenuation by taking its logarithm.
Likewise, unknown vendor processing generally makes real radiographs suitable for
an explicitly documented proxy comparison, not absolute attenuation RMSE.

For a display run, set `data_range` to 1 and `histogram_range` to `[0, 1]`. For
attenuation/proxy runs, choose and document the SSIM range and histogram bounds
before evaluation, keeping them fixed across the cohort and its ablations.
Do not choose new ranges independently for every candidate image.

## Metric definitions

| Metric | Implementation and interpretation |
| --- | --- |
| NCC | Zero-mean correlation within the fixed anatomy mask. Constant-signal correlation is undefined (`null`). |
| Gradient NCC | NCC of Sobel-gradient magnitude. Uses only centers whose complete 3×3 neighborhood is inside the mask. Primary structural score. |
| Mutual information | Joint histogram with 64 bins per axis by default, natural logarithm, fixed bounds. Outliers enter the boundary bins and their fractions are reported. |
| SSIM | Gaussian window, sigma 1.5, 11×11 support, population covariance and explicit `data_range`. Only windows entirely inside the mask contribute. |
| RMSE | Root mean squared difference inside the mask. Units follow the chosen signal domain. |
| Wasserstein distance | One-dimensional distance between the masked pixel-value distributions in the chosen domain. Uses all values, independent of the MI binning. |
| Radial profiles | Masked mean attenuation/proxy in 16 annuli about detector-array center; radius is in pixels. These are descriptive profiles, not a beam-hardening measurement. |

SSIM parameters follow the [scikit-image API](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity).
Histogram distance uses [SciPy's Wasserstein implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html).

For attenuation domains, **shape-only** fits `reference ≈ gain * rendered + bias`
by least squares on the same mask with `gain >= 0`. It does not fit pose or spatial
warps. The as-configured result always retains the original inputs. A constant
render has no identifiable gain; a zero-gain optimum is flagged so polarity or
pairing errors are not hidden. Fitting gain removes absolute attenuation-scale
information: even an excellent fitted RMSE does not establish physical calibration.

Gradient magnitude is insensitive to additive offset and its NCC is invariant to
positive global gain, but a nonlinear tone curve can change local edge weights.
It also cannot distinguish opposite image polarities by itself. Inspect NCC and
the declared display convention alongside it. MI depends on histogram settings;
Wasserstein distance ignores spatial arrangement. No individual score establishes
clinical realism.

## Evaluation protocol

1. **Verify geometry and transport.** Use analytical slabs/spheres, known point
   projections and step-size convergence before comparing real anatomy. Existing
   geometry tests provide CPU checks; successful metric tests do not verify the
   GPU renderer or a scanner model.
2. **Prepare matched references.** Follow the [dataset guide](validation-datasets.md).
   Render the same volume at the reference pose with its own detector geometry.
   Verify coordinate conventions, cropping and polarity before selecting masks.
   Include an independent deliberately misposed render, for example a specified
   5 mm translation and 5° rotation about named axes. An image-space shift is only
   a metric smoke test, not a calibrated 3D control.
3. **Separate shape and calibration.** Evaluate uncalibrated real images in a
   documented proxy domain. Evaluate the actual fixed display preset separately
   in the display domain, without per-image gain/bias or window fitting. Fit
   parameters on training subjects and evaluate on held-out subjects.
4. **Run component ablations.** Compare the baseline with HU-to-μ alternatives,
   detector blur, gain/bias, noise and display settings, recording all parameters
   and seeds. Independent noise realizations need not improve pixel RMSE even if
   the noise model is more realistic. Assess them with repeated realizations and
   noise statistics instead of demanding monotonic RMSE improvement.
5. **Report relative acceptance.** Pre-register the required gradient-NCC margin
   over the misposed control and acceptable held-out calibration error. The tool
   reports the gap and does not invent pass/fail thresholds. Keep per-subject
   distributions and acquisition-level uncertainty visible.

Detector MTF/NPS, dose-dependent DQE, scatter and temporal behavior require
additional calibrated acquisitions or phantoms. Registration accuracy, catheter
localization, task performance and observer studies are separate validation
stages. The [xvr geometry adapter and single-view exporter](validation-datasets.md#geometry-adapter-for-the-downloaded-xvr-release)
prepare matched dataset geometry separately. Cohort calibration, physics/task
evaluations and automatic clinical acceptance are not implemented by this metric runner.

## Test without patient data

```bash
python -m pip install -e './xray-simulator[dev,validation]'
python -m pytest xray-simulator/tests/test_validation.py
```

The tests check analytical identity/affine behavior, sensitivity to spatial
misalignment, polarity ambiguity, fixed mask support, histogram outliers,
Beer–Lambert inversion, input rejection and reproducible acquisition bootstrap.
They exercise the CLI end to end with generated arrays and run in CPU CI.
