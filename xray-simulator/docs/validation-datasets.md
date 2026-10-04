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
  the release's camera model. A verified coordinate/geometry adapter is required.
- The Ljubljana angiography volume is not automatically a calibrated HU CT.
  Document its intensity interpretation and compare DSA-like references to the
  appropriate signal, rather than treating them as ordinary transmission images.
- Load trusted `.pt` companions with `torch.load(..., map_location="cpu",
  weights_only=True)`. Record the dataset revision, pose convention and any
  volume-affine conversion alongside the prepared arrays.

CT-RATE is **not required** for either fluoroscopy dataset: their matched volumes
are already part of this release.

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
