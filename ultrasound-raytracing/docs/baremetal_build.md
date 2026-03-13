# Raysim – Bare-metal installation

1. Clone this repository:

   ```bash
   git clone https://github.com/isaac-for-healthcare/i4h-sensor-simulation.git
   cd i4h-sensor-simulation/ultrasound-raytracing
   ```

2. The required OptiX header files are fetched automatically by CMake from the
   open-source [optix-dev](https://github.com/NVIDIA/optix-dev) repository during the
   configure step.  All you need is a recent NVIDIA driver (OptiX is part of the
   driver) and CUDA ≥ 12.6 which are already listed in the requirements section.

3. Install Python dependencies and create virtual environment:

   > Note: The `pip install -e .[all]` command will automatically run the CMake build process. This requires:
   > - CUDA toolkit with `nvcc` in your `$PATH`
   > - CMake 3.24.0 or higher (install with `pip install cmake==3.24.0` if needed)
   > - CUDA paths properly set:
>
   > ```bash
   > export PATH=/usr/local/cuda/bin:$PATH
   > export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   >   ```

## Option A: Using uv

   ```bash
   uv sync && source .venv/bin/activate
   uv pip install -e .[all]
   ```

## Option B: Using conda

   ```bash
   # Create environment and install dependencies
   conda create -n ultrasound python=3.11 libstdcxx-ng -c conda-forge -y
   conda activate ultrasound
   pip install -e .[all]
   ```

1. Download mesh data:

   ```bash
   cd ultrasound-raytracing  # ensure you're in the correct directory
   pip install git+https://github.com/isaac-for-healthcare/i4h-asset-catalog.git
   # Download mesh data (~527MB)
   i4h-asset-retrieve --download-dir assets --sub-path Props/ABDPhantom/Organs --version 0.2.0
   # Note: The hash below is specific to version 0.2.0
   ln -s assets/8c0bf782eab2f44f1cc82da60eb10f6be8f941406d291b7fbfbdb53c05b3d149/Props/ABDPhantom/Organs mesh
   ```

   > **Troubleshooting Build Issues:**
   >
   > - In the [CMake setup file](../cmake/SetupCUDA.cmake), the default value for `CMAKE_CUDA_ARCHITECTURES` is set to `native`. This setting **may cause compilation failures** on systems with multiple NVIDIA GPUs that have different compute capabilities.
   >
   > - If you experience this issue, try specifying the GPU you want to use by setting the environment variable `export CUDA_VISIBLE_DEVICES=<selected device number>` before running `pip install`.

2. Run examples

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

   To build and run C++ examples, you can manually run CMake with `BUILD_EXAMPLES=ON`:

   ```bash
   cmake -DPYTHON_EXECUTABLE=$(which python) -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON -B build-release
   cmake --build build-release -j $(nproc)
   ./build-release/examples/cpp/ray_sim_example
   ```
