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
- Fixed the X-ray simulator example docs and Dockerfile header: the folder is
  `xray-simulator`, the example scripts cache to `xray-simulator/output/cache` (not
  `/tmp/fluoro_cache`), and the Docker demo command now runs the demo into the mount
- The ultrasound Docker guide's "Download mesh data" step now gives the command: it copies
  the meshes the image already contains into the mounted checkout, so `examples/server.py`
  finds `mesh/Tumor1.obj`
- Fixed X-ray `stream()` with `output.save_to_disk` writing every frame to
  `frame_0000.<format>`, which kept only the last frame. It now writes one
  `frame_NNNN.<format>` file per yielded frame, as `render_cine()` does
- Fixed X-ray `stream()` realism reusing the same noise field on every frame with a fixed
  seed. It now advances the seed per frame like `render_cine()`, changing noisy stream
  output after frame zero while keeping repeated streams reproducible. The output docs
  also explain stale files, single-frame overwrites and unbounded stream disk usage
- The ultrasound Docker guide now builds in `.venv-docker/` and `build-docker/`, so the host
  build in the same tree (`.venv/`, `build-release/`) works before or after it, and
  `docker build` no longer copies either build into the image. The ultrasound guides say
  how to clean a tree built by an earlier version of the Docker guide
- Ultrasound native extensions are now installed into each Python environment instead of
  the shared source tree, so host and container builds also stay separate with the same
  Python ABI. Remove legacy `ultrasound_simulator/ray_sim_python*.so` files and reinstall
  in both environments. The Docker image defaults subsequent installs to `build-docker/`

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
