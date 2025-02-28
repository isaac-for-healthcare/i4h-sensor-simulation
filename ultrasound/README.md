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
   ```bash
   uv sync
   ```

5. Build the project:
   ```bash
   cmake -DPYTHON_EXECUTABLE=$(python3 -c "import sys; print(sys.executable)") -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE --no-warn-unused-cli -S$(pwd)

   cmake -DCMAKE_BUILD_TYPE:STRING=Release -B build-release && cmake --build build-release --config Release --target all -j 66
   ```

## Running Examples

### C++ Example
```bash
./build/example/cpp/ray_sim_example
```
You can find the output frames are saved under ultrasound_sweep folder.

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
