# Raytracing Ultrasound Simulator

A high-performance GPU-accelerated ultrasound simulator using NVIDIA OptiX raytracing with Python bindings.

## Features

- GPU acceleration with CUDA and NVIDIA OptiX
- Python interface for ease of use
- Real-time simulation capabilities
- Support for curvilinear, linear, and phased array ultrasound probe simulation

## Benchmark Results
To reproduce these results, run `python examples/benchmark.py`.
```
Benchmark Results:
        Total frames: 200
        Average frame time: 0.0073 seconds
        Average FPS: 136.28
        Minimum FPS: 59.66
        Maximum FPS: 249.62
        Date: 2025-03-16 07:38:46

        System Information:
        GPU: NVIDIA RTX 6000 Ada Generation (48.0 GB, Driver: 565.57.01)
        CPU: AMD Ryzen Threadripper PRO 7975WX 32-Cores (64 cores)

```
## Requirements

- [CUDA 12.6+](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#)
- [NVIDIA Driver 555+](https://www.nvidia.com/en-us/drivers/)
- [CMake 3.24+](https://cmake.org/)
- [NVIDIA OptiX SDK 8.1](https://developer.nvidia.com/designworks/optix/downloads/legacy)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/isaac-for-healthcare/i4h-sensor-simulation.git
   cd i4h-sensor-simulation/ultrasound-raytracing
   ```

2. Download and set up OptiX SDK 8.1:
   - Download OptiX SDK 8.1 from the [NVIDIA Developer website](https://developer.nvidia.com/designworks/optix/downloads/legacy)
   - Extract the downloaded OptiX SDK archive
   - Place the extracted directory inside the `ultrasound-raytracing/third_party/optix` directory, maintaining the following structure:
     ```
     ultrasound-raytracing/third_party/
     └── optix
         └── NVIDIA-OptiX-SDK-8.1.0-<platform>  # Name may vary based on the platform
             ├── include
             │   └── internal
             └── SDK
                 ├── cuda
                 └── sutil
     ```

   Please note that the downloaded file is a shell script, you need to make it executable and run it before moving it to the `ultrasound-raytracing/third_party/optix` directory:

     ```bash
     chmod +x <path_to_downloaded_file>/NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64-35015278.sh
     ./<path_to_downloaded_file>/NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64-35015278.sh
     ```

3. Download mesh data:
   - The mesh data is a part of the Isaac for Healthcare asset package. You can download it by installing the asset helper tool:
   ```bash
   pip install git+ssh://git@github.com/isaac-for-healthcare/i4h-asset-catalog.git
   ```

   - Then you can download and extract the data to `~/.cache/i4h-assets/`
   ```bash
   i4h-asset-retrieve
   ```

   - The mesh data will be extracted to `~/.cache/i4h-assets/<sha256_hash>/Props/ABDPhantom/Organs`
   - You can copy the `Organs` folder to the `mesh` directory

   ```bash
   cp -r ~/.cache/i4h-assets/<sha256_hash>/Props/ABDPhantom/Organs mesh
   ```

4. Install Python dependencies and create virtual environment:

   **Option A: Using uv**
   ```bash
   uv sync && source .venv/bin/activate
   ```

   **Option B: Using conda**
   ```bash
   # Create environment and install dependencies
   conda create -n ultrasound python=3.10 libstdcxx-ng -c conda-forge -y

   conda activate ultrasound
   pip install -e .
   ```

5. Build the project

> Note: Before building, ensure the cuda compiler `nvcc` is installed.
>
>  ```bash
>  $ which nvcc
>  ```
>
>  If nvcc is not found, ensure cuda-toolkit is installed and can be found in `$PATH` and `$LD_LIBRARY_PATH` e.g.:
> ```bash
> export PATH=/usr/local/cuda/bin/:$PATH
> export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
> ```

   CMake 3.24.0 or higher is required to build the project, you can use `cmake --version` to check your version. If an older version is installed, you need to upgrade it:

   ```bash
   pip install cmake==3.24.0
   hash -r   # Reset terminal path cache
   ```

   Then you can build the project by:

   ```bash
   cmake -DPYTHON_EXECUTABLE=$(which python) -DCMAKE_BUILD_TYPE=Release -B build-release
   cmake --build build-release -j $(nproc)
   ```

>   Note:
>
>   - In the [CMake setup file](./cmake/SetupCUDA.cmake), the default value for `CMAKE_CUDA_ARCHITECTURES` is set to `native`. This setting **may >cause compilation failures** on systems with multiple NVIDIA GPUs that have different compute capabilities.
>
>   - If you experience this issue, try specifying the GPU you want to use by setting the environment variable `export CUDA_VISIBLE_DEVICES=> >.<selected device number>` before building the project.
>
6. Run examples

   **Using uv**
   ```bash
   # Basic example
   uv run examples/sphere_sweep.py

   # Web interface (open http://localhost:8000 afterward)
   uv run examples/server.py
   ```

   **Using conda**
   ```bash
   # Using the system's libstdc++ with LD_PRELOAD if your conda environment's version is too old
   python examples/sphere_sweep.py

   # Web interface
   python examples/server.py
   ```

   **C++ example**
   ```bash
   ./build-release/examples/cpp/ray_sim_example
   ```

## Basic Usage

```python
import raysim.cuda as rs
import numpy as np

# Create materials
materials = rs.Materials()

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
