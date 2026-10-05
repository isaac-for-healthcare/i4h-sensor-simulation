# Reference datasets for image validation

Download data separately from source code and keep original dataset cards and
licenses. The simulator's Apache license does not relicense third-party data.
Keep dataset files, access tokens and generated patient images outside version control.

| Dataset | Use | Needed for paired comparison |
| --- | --- | --- |
| DeepFluoro | Real pelvis fluoroscopy; 6 subjects and 366 views in the referenced release | Matching subject volume, reference view, calibrated pose/intrinsics and fixed anatomy mask |
| Ljubljana | Real cerebral angiography; 10 subjects and 20 primary views in the referenced release | Matching 3D angiography volume, 2D acquisition, pose/intrinsics and explicitly chosen signal preparation |
| DRR-RATE | Independent synthetic chest projections | Exact CT-RATE case/reconstruction and projection settings; its PNGs alone do not provide a paired simulator benchmark |

## DeepFluoro and Ljubljana

The [author-hosted xvr-data release](https://huggingface.co/datasets/eigenvivek/xvr-data)
provides DICOM/NIfTI versions, per-view pose/intrinsics and fiducials. It is a
different layout from the original DiffPose HDF5 files. Use the corresponding
[xvr loading/evaluation code](https://github.com/eigenvivek/xvr) to establish
coordinate conventions; do not feed these files to an unrelated HDF5 loader.

The dataset card identifies DeepFluoro as CC BY-NC 4.0 and Ljubljana as
CC BY-NC-ND 4.0, with the latter remix hosted by permission. Follow the original
terms and cite the original papers linked in the card. Dataset availability is
not permission for unrestricted commercial use or redistribution.

Example acquisition with the Hugging Face CLI, after reviewing the dataset terms:

```bash
hf download eigenvivek/xvr-data --repo-type dataset \
  --revision a17273e3eadbd793bd861f3598a80ce4590c1124 \
  --include 'deepfluoro/**' 'ljubljana/**' README.md \
  --local-dir /path/to/validation-data/fluoroscopy --dry-run

hf download eigenvivek/xvr-data --repo-type dataset \
  --revision a17273e3eadbd793bd861f3598a80ce4590c1124 \
  --include 'deepfluoro/**' 'ljubljana/**' README.md \
  --local-dir /path/to/validation-data/fluoroscopy
```

Preparation checks for this pinned release:

- DeepFluoro reference DICOMs are 1536×1536, while pose companions specify a
  1436×1436 image grid. Account for the 50-pixel crop on every edge and the
  associated principal point. Do not resize one image until its array shape
  happens to match the other.
- The [upstream evaluator](https://github.com/eigenvivek/xvr/blob/caa55cc8096294cf70a218126bf16008dee0dec7/experiments/evaluate.py)
  flags DeepFluoro subject01 views `003`/`050` and subject04 views `002`/`004`.
  Record exclusions before evaluation rather than filtering by the resulting
  similarity score.
- Inspect each view's SDD, pixel pitch and principal-point offsets. The simulator's
  default centered detector and raw Euler frame do not automatically reproduce
  the release's camera model. Use the adapter below; the default AP pose is not a dataset pose.
- The Ljubljana angiography volume is not automatically a calibrated HU CT.
  Document its intensity interpretation and compare DSA-like references to the
  appropriate signal, rather than treating them as ordinary transmission images.
- Load trusted `.pt` companions with `torch.load(..., map_location="cpu",
  weights_only=True)`. Record the dataset revision, pose convention and any
  volume-affine conversion alongside the prepared arrays.

CT-RATE is **not required** for either fluoroscopy dataset: their matched volumes
are already part of this release.

## Geometry adapter for the downloaded xvr release

`xray_simulator.validation.xvr` adapts the pinned **NIfTI/DICOM/PT xvr-data layout**.
It supports DeepFluoro and Ljubljana camera geometry. It does not accept the
original DeepFluoro HDF5 camera matrices, automatically register images, or infer
a scanner's attenuation/processing model.

Install file-loading and rendering dependencies separately from the CPU metrics:

```bash
python -m pip install -e './xray-simulator[datasets,validation,slang]'
```

From `xray-simulator/`, preview and then export a matched DeepFluoro view:

```bash
python -m examples.render_deepfluoro \
  --subject-dir /path/to/validation-data/fluoroscopy/deepfluoro/subject01 \
  --view 000 --binning 4 --output output/deepfluoro_subject01_000 --dryrun

python -m examples.render_deepfluoro \
  --subject-dir /path/to/validation-data/fluoroscopy/deepfluoro/subject01 \
  --view 000 --binning 4 --output output/deepfluoro_subject01_000
```

The exporter refuses to overwrite an existing directory. It saves stored reference
values, rendered intensity, attenuation and display arrays, preview PNGs, and a
manifest with geometry, transforms, HU mapping, source hashes and code hashes.
The reference PNG is scaled for viewing only. The raw reference is not calibrated
attenuation; prepare a documented proxy and fixed mask before running the metric
runner. Default HU/display settings are an engineering baseline, not a fitted
fluoroscopy preset. No similarity acceptance is claimed by the exporter.

The geometry API can also be used directly:

```python
from xray_simulator.validation.xvr import load_xvr_view, load_xvr_volume

values_zyx, frame = load_xvr_volume(subject_dir / "volume.nii.gz")
view = load_xvr_view(subject_dir, "000", dataset="deepfluoro")
camera = view.camera(frame, binning=4)
reference = view.load_reference(binning=4)
# Preprocess values_zyx using frame.spacing_zyx_mm and frame.origin_xyz_mm.
# Use camera.geometry in SimulatorConfig, and render_frame(pose=camera.pose).
# camera.project(fiducials) returns zero-based (column, row) pixel centers.
```

For Ljubljana use `dataset="ljubljana"` and a view such as `"frontal"`. The
geometry conversion is the same, but the 3D angiography values need an explicit
signal model; do not pass them through a HU mapping by assumption. The example
exporter deliberately handles only DeepFluoro CTs.

### Coordinate contract

The adapter follows the pinned [xvr evaluator](https://github.com/eigenvivek/xvr/blob/caa55cc8096294cf70a218126bf16008dee0dec7/experiments/evaluate.py)
and the camera/volume conventions of [DiffDRR 0.6.0](https://pypi.org/project/diffdrr/0.6.0/):

- CT coordinates are NIfTI RAS, centered at the midpoint of the first and last
  voxel centers. `warp.txt` is **not** applied: the evaluator uses the supplied
  NIfTI directly, centered in this way.
- The stored pose is applied after the AP camera reorientation, with
  `reverse_x_axis=False`. The image row axis has the opposite sign to the
  camera's vertical axis.
- The adapter preserves the voxel samples and spacing. To express the camera
  with proper Euler rotations, it reverses voxel Y when needed to obtain a
  left-handed grid-to-RAS transform. Both coordinate reflections cancel. The
  renderer frame is explicitly unlabeled; anatomical AP/PA presets must not be
  mixed into it. Orthogonal oblique affines work without interpolation; sheared
  affines are rejected.
- The renderer origin is the volume box corner, half a voxel before its first
  sample center. Source location is preserved exactly. SID is chosen as SDD/2
  solely to parameterize the same source/detector pose; it is not inferred
  patient distance.
- Stored `x0/y0` values are principal-point offsets in mm from the image
  center, along image columns and rows. The adapter sets
  `principal_point_px = ((width-1)/2 + x0/dx, (height-1)/2 + y0/dy)` with
  separate horizontal/vertical pitches. This matches xvr's registration, which
  negates `x0` (only) before constructing DiffDRR's detector
  ([xvr commit 4c68e0c](https://github.com/eigenvivek/xvr/commit/4c68e0cebc6d2c0f8e04690ff054951a4299f90f)):
  DiffDRR 0.6.0's detector grid reverses the row axis but not the column axis
  before adding its constructor offsets, so the raw stored values place the
  principal point correctly in Y but mirrored in X. DeepFluoro's offsets are
  half a native pixel and cannot distinguish the two; Ljubljana's horizontal
  offsets reach 28 mm and the GPU data test checks them.
- DeepFluoro references lose exactly 50 pixels on each edge to match the
  companion's 1436×1436 grid. No further principal-point shift is applied to the
  already-cropped calibration. Ljubljana references use the supplied grid.
- Integer binning must divide both dimensions. It multiplies detector pitches,
  keeps the physical principal point and FOV fixed, and block-averages stored reference
  values. It does not model detector blur or average simulated subpixel rays.
- Coordinates returned by `camera.project` place the first pixel center at
  `(0, 0)`. DiffDRR's continuous intrinsic projection uses pixel-boundary
  coordinates; subtract 0.5 on both axes for comparison.

The four upstream questionable DeepFluoro poses fail closed unless
`include_excluded=True` is explicitly requested. The DICOM loader accepts the
release's single-frame MONOCHROME2 stored values; other polarities, modality LUTs,
or nonidentity rescale transforms require explicit preparation.

For centered-RAS points, the independent pinhole equation used by the tests is
`[u, v, 1] ~ K * inverse(P * AP) * [X, Y, Z, 1]`, with
`fx = SDD/dx`, `fy = -SDD/dy`,
`cx = (width-1)/2 + x0/dx`, `cy = (height-1)/2 + y0/dy`.
CPU tests cover this equation, voxel-center placement, oblique affines, cropping,
binning, exclusions and invalid geometry. A `gpu` test projects a synthetic bead
through the actual shader with an off-center principal point and unequal detector
pitches. With `XVR_DATA_ROOT` set to a local xvr-data download, a `gpu`/`slow`
test renders all 20 primary Ljubljana views and requires the vessel edges to lie
within one binned pixel of the reference angiograms.

## DRR-RATE and CT-RATE

The [DRR-RATE dataset card](https://huggingface.co/datasets/farrell236/DRR-RATE)
describes Siddon–Jacobs projections of CT-RATE and gives the generation command:
a −100 HU threshold, a 300 mm y translation and a −90° z rotation for lateral
views. Other geometry comes from the referenced generator and its defaults;
do not assume the simulator's AP/lateral presets reproduce them.

The validation split can be downloaded independently:

```bash
hf download farrell236/DRR-RATE --repo-type dataset \
  --revision 500d7868a84478eb0bafcdee793d1246f492c986 \
  --include 'valid/**' README.md \
  --local-dir /path/to/validation-data/drr-rate --dry-run

hf download farrell236/DRR-RATE --repo-type dataset \
  --revision 500d7868a84478eb0bafcdee793d1246f492c986 \
  --include 'valid/**' README.md \
  --local-dir /path/to/validation-data/drr-rate
```

This pinned split contains 3,039 case/reconstruction identifiers with AP and lateral
views (6,078 PNGs). Match the full reconstruction identifier, not just a patient
number, to its source CT. For example, `valid_1_a_1.png` corresponds to
`dataset/valid/valid_1/valid_1_a/valid_1_a_1.nii.gz` in CT-RATE.

[CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) has an access gate
and additional use conditions. Review them and obtain access through the dataset
host before downloading the selected source volumes. Neither the metric tool nor
this guide accepts terms, requests access or bypasses a gate on a user's behalf.
Record the exact CT revision and file hashes used for a run.

Agreement with these synthetic DRRs checks consistency with another renderer.
It does not establish agreement with real diagnostic radiographs. Comparing an
unrelated generated chest image to an arbitrary DRR-RATE image using SSIM/NCC is
not a paired validation experiment.

## Produce metric inputs

Keep the source data read-only. In a separate preparation directory, export a
reference array, its matched rendered array, and a fixed anatomy mask for each
view. Record all geometry and intensity transformations in the manifest described
in [Image validation](validation.md). Mask captions, collimator boundaries and
unmatched instruments using a rule established before evaluating candidates.

The metric runner deliberately starts from prepared arrays. It does not infer
correspondence from filenames, convert dataset poses, download patient data or
claim that display pixel values are calibrated attenuation.
