# Raytracing Ultrasound Simulator

A high-performance GPU-accelerated ultrasound simulator using NVIDIA OptiX raytracing with Python bindings.

## Features

- GPU acceleration with CUDA and NVIDIA OptiX
- Python interface for ease of use
- Real-time simulation capabilities

## Requirements

- [CUDA 12.6+](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#)
- [NVIDIA Driver 555+](https://www.nvidia.com/en-us/drivers/)
- [CMake 3.24+](https://cmake.org/)
- [NVIDIA OptiX SDK 8.1](https://developer.nvidia.com/designworks/optix/downloads/legacy)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/isaac-for-healthcare/i4h-sensor-simulation.git
   cd i4h-sensor-simulation/ultrasound
   ```

2. Download and set up OptiX SDK 8.1:
   - Download OptiX SDK 8.1 from the [NVIDIA Developer website](https://developer.nvidia.com/designworks/optix/downloads/legacy)
   - Extract the downloaded OptiX SDK archive
   - Place the extracted directory inside the `ultrasound/third_party/optix` directory, maintaining the following structure:
     ```
     ultrasound/third_party/
     └── optix
         └── NVIDIA-OptiX-SDK-8.1.0-<platform>  # Name may vary based on the platform
             ├── include
             │   └── internal
             └── SDK
                 ├── cuda
                 └── sutil
     ```

3. Download mesh data (if required):
   ```bash
   <step to download mesh data> # Download mesh data
   ```

4. Install Python dependencies and create virtual environment:
#### Option A: Using uv
   ```bash
   uv sync && source .venv/bin/activate
   ```

#### Option B: Using conda
```bash
# Create environment and install dependencies
conda create -n ultrasound python=3.10
conda activate ultrasound
pip install -e .
conda install -c conda-forge qt pyqt

5. Build the project
# Build the project
cmake -DPYTHON_EXECUTABLE=$(which python) -DCMAKE_BUILD_TYPE=Release -B build-release && cmake --build build-release -j $(nproc)


6. Run examples
#### using uv
uv run examples/sphere_sweep.py

#### using conda
# Using the system's libstdc++ with LD_PRELOAD if your conda environment's version is too old
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python examples/sphere_sweep.py
```


```bash
# C++ example
./build-release/examples/cpp/ray_sim_example

# Sweep example (uv)
uv run examples/sphere_sweep.py

# Sweep example (conda)
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python examples/sphere_sweep.py

# Web interface (open http://localhost:8000 afterward)
# With uv:
uv run examples/server.py
# With conda:
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python examples/server.py
```

## Basic Usage

```python
import raysim.cuda as rs
import numpy as np

# Create world and add objects
world = rs.World("water")
material_idx = materials.get_index("fat")
sphere = rs.Sphere([0, 0, -145], 40, material_idx)
world.add(sphere)

# Create simulator
simulator = rs.RaytracingUltrasoundSimulator(world, materials)

# Configure probe
probe = rs.UltrasoundProbe(rs.Pose(position=[0, 0, 0], rotation=[0, np.pi, 0]))

# Set simulation parameters
sim_params = rs.SimParams()
sim_params.t_far = 180.0

# Run simulation
b_mode_image = simulator.simulate(probe, sim_params)
```

## Development

For development, VSCode with the dev container is recommended:
1. Open project in VSCode with Dev Containers extension
2. Use command palette (`Ctrl+Shift+P`) to run `CMake: Configure`
3. Build with `F7` or `Ctrl+Shift+B`

### Pre-commit Hooks

```bash
# For uv users
uv pip install -e ".[dev]" && pre-commit install

# For conda users
pip install -e ".[dev]" && pre-commit install
```
