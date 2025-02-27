# Raytracing Ultrasound Simulator

A high-performance GPU-accelerated ultrasound simulator using NVIDIA OptiX raytracing with Python bindings.

## Features

- GPU acceleration with CUDA and NVIDIA OptiX
- Python interface for ease of use
- Real-time simulation capabilities

## Requirements

- CUDA 12.6+
- NVIDIA Driver 555+
- CMake 3.22.1+
- NVIDIA OptiX SDK 8.1

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

4. Build the project:
   ```bash
   cmake -B build
   cmake --build build -j
   ```

5. Install Python dependencies:
   ```bash
   uv sync
   ```

## Running Examples

### C++ Example
```bash
./build/example/cpp/ray_sim_example
```

### Python Example
```bash
uv run examples/sphere_sweep.py
```

### Web Interface
```bash
uv run examples/server.py
# Open http://localhost:8000 in your browser
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
# Install dev dependencies and hooks
uv pip install -e ".[dev]"
pre-commit install

# Manually run hooks
pre-commit run --all-files
```
