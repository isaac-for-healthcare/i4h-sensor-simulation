# Examples & Test Data

This document covers test data sources, example scripts, and how to run the examples.

## Table of Contents

1. [Test Data](#test-data)
2. [Examples](#examples)
3. [Running the Examples](#running-the-examples)

---

## Test Data

Example CT data is not included in this repository. You can use public datasets for testing.

### Recommended: Multiphase CT Angiography Dataset (Kaggle)

We recommend the [Multiphase CT Angiography dataset](https://www.kaggle.com/datasets/adamhuan/multiphase-ct-anigography-2-datasets) from Kaggle for testing.

**Download and Setup:**

```bash
# 1. Install Kaggle CLI (if not already installed)
pip install kaggle

# 2. Download the dataset (requires Kaggle API credentials)
kaggle datasets download -d adamhuan/multiphase-ct-anigography-2-datasets

# 3. Extract to a local directory
unzip multiphase-ct-anigography-2-datasets.zip -d ~/ct_data

# The extracted structure will be:
# ~/ct_data/excellent/excellent/0/  (high-quality scans)
# ~/ct_data/moderate/moderate/0/    (moderate-quality scans)
```

**Using with Docker:**

To use the downloaded CT data inside the Docker container, mount the host directory:

```bash
# Run container with CT data mounted
docker run --rm --gpus all -v ~/ct_data:/root/ct_data -it fluorosim bash

# Inside container, run the demo (will use mounted CT data)
python examples/fluorosim_demo.py
```

> **Note:** The demo script looks for data at `/root/ct_data/excellent/excellent/0` inside the container. The volume mount `-v ~/ct_data:/root/ct_data` maps your host's `~/ct_data` to the container's `/root/ct_data`.

**Preprocess the CT Volume:**

```python
from fluorosim import VolumePreprocessor, PreprocessedVolume
from pathlib import Path

# Path to extracted DICOM directory
dicom_path = Path("~/ct_data/excellent/excellent/0").expanduser()

# Load and preprocess
preprocessor = VolumePreprocessor.from_dicom(dicom_path)
print(f"Volume shape: {preprocessor.shape}")        # e.g., (553, 512, 512)
print(f"Spacing (ZYX): {preprocessor.spacing_zyx_mm} mm")
print(f"HU range: {preprocessor.hu_range}")

# Convert HU → μ and cache
volume = preprocessor.preprocess(output_dir="/tmp/fluoro_cache")
print(f"μ range: [{volume.mu_volume.min():.4f}, {volume.mu_volume.max():.4f}] mm⁻¹")
```

**Or use the preprocessing script:**

```bash
cd fluoro-simulator

# Preprocess DICOM data
python examples/preprocess_ct.py --dicom ~/ct_data/excellent/excellent/0

# Preprocess NIfTI data
python examples/preprocess_ct.py --nifti ~/ct_data/volume.nii.gz

# Use synthetic test data (no download needed)
python examples/preprocess_ct.py --synthetic
```

**Expected preprocessing output:**

The script will display:

| Output Field | Description |
| ------------ | ----------- |
| Volume shape (Z, Y, X) | Dimensions of the loaded CT volume in voxels |
| Voxel spacing (mm) | Physical size of each voxel |
| HU range | Min/max Hounsfield Unit values in the input |
| μ range | Min/max linear attenuation coefficients after conversion (mm⁻¹) |
| Output path | Location of the cached preprocessed volume |

A verification step confirms the output is valid (correct dtype, non-empty μ range).

### Alternative: The Cancer Imaging Archive (TCIA)

Public datasets from [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/) can also be used for testing.

---

## Examples

### Command Line Usage

```bash
cd fluoro-simulator

# Step 1: Preprocess CT data (choose one)
python examples/preprocess_ct.py --synthetic                      # Synthetic test data
python examples/preprocess_ct.py --dicom ~/ct_data/dicom_folder   # DICOM series
python examples/preprocess_ct.py --nifti ~/ct_data/volume.nii.gz  # NIfTI file

# Step 2: Run simulator (uses cached volume from /tmp/fluoro_cache)
python examples/fluorosim_demo.py
```

**Expected simulator output:**

The demo executes in four steps:

| Step | Description |
| ---- | ----------- |
| 1. Loading CT Volume | Loads the preprocessed μ-volume from cache (or creates synthetic) |
| 2. Initializing Simulator | Initializes GPU resources, displays device info and backend |
| 3. Rendering Frames | Renders a LAO/RAO sweep sequence, shows real-time FPS |
| 4. Saving Results | Saves frames to `/tmp/fluorosim_output/frames/` |

**Performance metrics** are reported at the end: average FPS, frame time (ms), and timing jitter.

> **Typical performance:** ~100-300 FPS depending on GPU, volume size, and detector resolution.
> **Note: Output Directories**
> The demo saves frames to **two locations** (they are identical copies):
>
> - `/tmp/fluorosim_output/` — Saved automatically by the simulator when `save_to_disk=True` is set in `OutputSettings`
> - `/tmp/fluorosim_output/frames/` — Saved explicitly by the demo script via `cine.save_all()`
> This is intentional for demonstration purposes. In your own code, use **one approach**:
> - Set `save_to_disk=True` in config for automatic saving, **or**
> - Call `cine.save_all()` manually for explicit control (recommended)

### Single Frame Rendering

```python
from fluorosim import VolumePreprocessor, FluoroSimulator, PreprocessedVolume

# Load cached volume (or preprocess from CT)
volume = PreprocessedVolume.load("/tmp/fluoro_cache")

# Create simulator
simulator = FluoroSimulator(volume)

# Render AP view (anterior-posterior, no rotation)
frame_ap = simulator.render_frame(rotation=(0, 0, 0))
frame_ap.save("/tmp/ap_view.png")

# Render 30° LAO view
import math
frame_lao = simulator.render_frame(rotation=(0, math.radians(30), 0))
frame_lao.save("/tmp/lao_30.png")

# Render 15° cranial + 20° RAO
frame_combined = simulator.render_frame(
    rotation=(math.radians(15), math.radians(-20), 0)
)
frame_combined.save("/tmp/cranial_rao.png")
```

### Cine Sequence (Video)

```python
from fluorosim import FluoroSimulator, Pose, PreprocessedVolume
import math

volume = PreprocessedVolume.load("/tmp/fluoro_cache")
simulator = FluoroSimulator(volume)

# Generate 360° rotational angiography sequence
poses = [
    Pose(rotation=(0, math.radians(angle), 0))
    for angle in range(0, 360, 3)  # 120 frames, 3° per frame
]

cine = simulator.render_cine(poses, fps=30.0)
cine.save_all("/tmp/rotational_angio", format="png")

# Convert to numpy for further processing
frames = cine.to_numpy()  # Shape: (120, 512, 512)
```

### Real-time Streaming

```python
from fluorosim import FluoroSimulator, Pose, PreprocessedVolume

volume = PreprocessedVolume.load("/tmp/fluoro_cache")
simulator = FluoroSimulator(volume)

def pose_generator():
    """Generate poses from external controller or RL environment."""
    angle = 0.0
    while True:
        yield Pose(rotation=(0, angle, 0))
        angle += 0.01  # Continuous rotation

# Stream frames (useful for interactive applications)
for frame in simulator.stream(pose_generator(), max_frames=100):
    process_frame(frame.image)  # Your processing function
```

### Gradient-Based Pose Optimization (2D/3D Registration)

```python
import torch
from fluorosim.rendering.diffdrr_slang_renderer import TorchSlangDiffDRR, SlangDiffDRRConfig
from fluorosim import PreprocessedVolume

# Load volume
volume = PreprocessedVolume.load("/tmp/fluoro_cache")

# Create differentiable renderer
config = SlangDiffDRRConfig(det_width_px=256, det_height_px=256)
drr = TorchSlangDiffDRR(volume.mu_volume, volume.spacing_zyx_mm, config)

# Initialize pose parameters
rotation = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
translation = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)

# Target X-ray (from real fluoroscopy)
target = torch.from_numpy(load_target_xray())  # Your target image

# Optimize pose
optimizer = torch.optim.Adam([rotation, translation], lr=0.01)

for step in range(100):
    optimizer.zero_grad()

    synthetic = drr(rotation, translation)
    loss = ((synthetic - target) ** 2).mean()  # MSE loss

    loss.backward()  # Gradients via Slang autodiff
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}: loss={loss.item():.6f}")

print(f"Optimized rotation: {rotation.detach().numpy()}")
print(f"Optimized translation: {translation.detach().numpy()}")
```

---

## Running the Examples

The `examples/` directory contains two main scripts that demonstrate the complete workflow.

### Step 1: Preprocess CT Data (`preprocess_ct.py`)

This script converts CT volumes (in Hounsfield Units) to linear attenuation coefficients (μ) required for X-ray simulation.

```bash
cd fluoro-simulator

# Option A: Use synthetic test data (no download required)
python examples/preprocess_ct.py --synthetic

# Option B: Preprocess DICOM series
python examples/preprocess_ct.py --dicom ~/ct_data/dicom_folder

# Option C: Preprocess NIfTI file
python examples/preprocess_ct.py --nifti ~/ct_data/volume.nii.gz

# Specify custom output directory
python examples/preprocess_ct.py --synthetic --output /path/to/cache
```

**Arguments:**

| Argument | Description |
| --------- | ----------- |
| `--dicom PATH` | Path to DICOM series directory |
| `--nifti PATH` | Path to NIfTI file (.nii or .nii.gz) |
| `--synthetic` | Generate a synthetic test volume (sphere in air) |
| `--output PATH` | Output directory for cached volume (default: `/tmp/fluoro_cache`) |

**Expected Output:**

The script displays volume metadata (shape, spacing, HU range), performs HU→μ conversion, runs verification checks, and saves to the output directory. On success, it prints the path and usage instructions:

```python
volume = PreprocessedVolume.load("/tmp/fluoro_cache")
simulator = FluoroSimulator(volume)
```

### Step 2: Run the Simulator Demo (`fluorosim_demo.py`)

This script loads a preprocessed volume and renders a C-arm sweep animation.

```bash
cd fluoro-simulator

# Run the demo (auto-loads cached volume from /tmp/fluoro_cache)
python examples/fluorosim_demo.py
```

**What it does:**

1. Loads cached μ-volume from `/tmp/fluoro_cache` (or creates synthetic if not found)
2. Initializes `FluoroSimulator` with the configured detector resolution
3. Performs GPU warmup for accurate FPS measurement
4. Renders a sequence of frames in a LAO/RAO (left/right oblique) sweep
5. Saves frames to `/tmp/fluorosim_output/frames/`
6. Reports performance metrics (FPS, frame time, jitter)

**Expected Output:**

The demo displays initialization info (GPU device, Slang module, texture sizes), renders frames with real-time FPS updates, and reports final performance metrics:

| Metric | Description |
| ------ | ----------- |
| Average FPS | Frames per second (excluding warmup) |
| Frame time | Average time per frame in milliseconds |
| Jitter (std) | Standard deviation of frame times |

Output frames are saved to `/tmp/fluorosim_output/frames/` as PNG images.

### Complete Workflow

```bash
cd fluoro-simulator

# 1. Install the package
pip install -e .[all]

# 2. Preprocess CT data (choose one)
python examples/preprocess_ct.py --synthetic                     # Quick test
python examples/preprocess_ct.py --dicom ~/ct_data/dicom_folder  # Real data

# 3. Run the simulator
python examples/fluorosim_demo.py

# 4. View output frames
ls /tmp/fluorosim_output/frames/
# frame_0000.png  frame_0001.png  frame_0002.png  ...
```

## Known Warnings

### Slang Compiler Warning (safe to ignore)

You may see this warning during initialization:

```text
[WARN] Slang compiler warnings:
warning 41012: entry point 'renderDRR_forward_entrypoint' uses additional capabilities
that are not part of the specified profile 'unknown'. The profile setting is automatically
updated to include these capabilities: 'cuda_sm_2_0'
```

**This is harmless.** The Slang shader compiler auto-detects that the code uses CUDA-specific features and enables the required capabilities. The renderer works correctly — this is purely informational.

### OptiX Warnings (Safe to Ignore)

```text
[WARN] (rhi) layer: Failed to initialize OptiX 9.0: Library not found (OPTIX_ERROR_LIBRARY_NOT_FOUND)
[WARN] (rhi) layer: Failed to initialize OptiX 8.1: Library not found (OPTIX_ERROR_LIBRARY_NOT_FOUND)
```

**This is harmless.** The renderer uses CUDA compute shaders, not OptiX ray tracing. Slang probes for OptiX availability but falls back gracefully when it's not installed.
