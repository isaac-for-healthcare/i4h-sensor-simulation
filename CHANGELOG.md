# Isaac for Healthcare - Sensor Simulation - Changelog

## Unreleased

- Removed the `i4h` CLI wrapper. Run the X-ray simulator through its Python examples,
  the `xray-simulator` config-file launcher or its Dockerfile, and the ultrasound
  simulator through `build_and_run.sh`
- Fixed X-ray DICOM loading applying RescaleSlope and RescaleIntercept twice, which
  left series with a nonzero intercept offset in HU (1024 HU low for the common
  -1024 intercept). Delete preprocessed caches (`mu_volume.npy` and `metadata.json`)
  built from DICOM before this fix and preprocess again: `PreprocessedVolume.load` and
  the demo reuse an existing cache without re-reading the source. NIfTI and synthetic
  caches are not affected
- The ultrasound Docker guide's "Download mesh data" step now gives the command: it copies
  the meshes the image already contains into the mounted checkout, so `examples/server.py`
  finds `mesh/Tumor1.obj`

## Version 0.6.0

- `i4h` CLI minor update to pin CLI to a specific commit for reproducibility

## Version 0.5.0

- Fluoro simulator API and rendering pipeline
- `i4h` CLI with support for fluoro simulator and ultrasound raytracing
- UX improvements

## Version 0.4.0

- Contact awareness for ultrasound probe
- `pip install` support for building the raytracing simulator
- GitHub workflow for building the ultrasound raytracing simulator

## Version 0.3.0

- Getting started tutorial
- Technical description of the ultrasound simulator
- Docker build instructions

## Version 0.2.0

- Phased array and linear probe geometries
- CUDA-based speckle noise filter
- Frequency-dependent attenuation

## Version 0.1.0

- Initial ultrasound raytracing simulator
- Probe configuration with setters and pose definition
- Example applications and server example
