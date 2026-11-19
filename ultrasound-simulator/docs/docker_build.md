# ultrasound_simulator – Quick Docker Guide

Run the ultrasound-simulator on **any Linux host with an NVIDIA GPU** in three short steps.

---

## 1 · Build the image (done once)

```bash
cd ultrasound-simulator
# ~10 min on first run, <3 GB compressed
docker build \
  --build-arg USER_UID=$(id -u) \
  --build-arg USER_GID=$(id -g) \
  -f .devcontainer/Dockerfile \
  -t ultrasound_simulator \
  .
```

> The image already contains CUDA 13, Python 3.12, CMake and a non-root
> user called `ultrasound_simulator`.

CUDA 13 covers compute capability 7.5 through 12.1, which includes Blackwell GPUs
such as GB10 (DGX Spark, sm_121) and RTX PRO Blackwell (sm_120). It needs a 580 or
newer driver. On an older driver, or on a pre-Turing GPU, build against CUDA 12
instead:

```bash
docker build \
  --build-arg USER_UID=$(id -u) \
  --build-arg USER_GID=$(id -g) \
  --build-arg CUDA_MAJOR=12 \
  -f .devcontainer/Dockerfile \
  -t ultrasound_simulator \
  .
```

Check your driver with `nvidia-smi --query-gpu=driver_version,compute_cap --format=csv`.

---

## 2 · Download mesh data (done once)

The image built in step 1 already contains the organ meshes (`Props/ABDPhantom/Organs`,
asset catalog version 0.2.0, ~527 MB). Step 3 mounts this directory over the container's
working directory, so copy them into it once:

```bash
cd ultrasound-simulator # ensure you are in the sensor directory

docker run --rm -v "$(pwd)":/ultrasound_simulator ultrasound_simulator \
  cp -rL /home/ultrasound_simulator/workspace/mesh /ultrasound_simulator/mesh
ls mesh/Tumor1.obj
```

> The copy needs no GPU, so ignore the CUDA image's warning that none is attached. It runs
> as the image's user, which step 1 created with your UID and GID, so the files are yours
> on the host. `mesh/` is git-ignored. To download the meshes on the host
> instead, follow "Download mesh data" in the [bare-metal guide](baremetal_build.md).

---

## 3 · Start a container and set up the project (done after every `git pull`)

```bash
cd ultrasound-simulator # ensure you are in the sensor directory

docker run -it --rm --gpus all --network host \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$(pwd)":/ultrasound_simulator  \
  --workdir /ultrasound_simulator      \
  ultrasound_simulator
```

Inside the container, create a virtual environment and install the Python dependencies.
The `all` extra provides the packages the examples import (matplotlib, tqdm, flask and
the rest); `-e .` alone installs only numpy:

```bash
python3 -m venv .venv-docker
source .venv-docker/bin/activate
python -m pip install -e ".[all]" --config-settings=build-dir=build-docker
```

The editable install also builds the native extension and installs it into
`.venv-docker/`; no separate CMake invocation is needed. The image sets
`SKBUILD_BUILD_DIR=build-docker`, so subsequent plain `pip install -e ".[all]"`
commands also use the container's build directory.

> **Building on the host in the same tree:** the container's virtual environment
> (`.venv-docker/`) and build directory (`build-docker/`) record container paths
> such as `/ultrasound_simulator`, so they are kept apart from the `.venv/` and
> `build-release/` that the [bare-metal guide](baremetal_build.md) and the
> [getting-started guide](ultrasound_simulator_getting_started.md#prerequisites-and-installation)
> use on the host. Each environment gets its own native extension, even with the same
> Python version. Both builds work in the same tree, in either order, and step 1 leaves all of
> these directories out of the image. Earlier versions of this guide used `.venv/` and
> `build-release/` inside the container; in a tree they built, run `rm -rf build-release` and
> replace the environment with `uv venv --clear` before building on the host. Also remove
> legacy in-source extensions with `rm -f ultrasound_simulator/ray_sim_python*.so` before
> reinstalling in both environments.

---

## 4 · Run an example

```bash
# still inside the same container (venv active)
python examples/sphere_sweep.py         # head-less benchmark
# OR
python examples/server.py               # web-GUI on port 8000
```

Now open your browser **on the host** at:

```text
http://localhost:8000
```

If you are on a different machine within the same network, replace
`localhost` with the host's IP address (e.g. `http://192.168.1.42:8000`).

> **Running on a Remote Server**: If you're running this on a remote server, you can forward the port to your local machine using SSH:
>
> ```bash
> ssh -L 8000:localhost:8000 username@remote-server
> ```
>
> Then access the web interface at `http://localhost:8000` on your local machine.

## Troubleshooting

### RuntimeError: Optix call `optixInit()` failed with 7804 (OPTIX_ERROR_LIBRARY_NOT_FOUND): Library not found

If you encounter an error similar to:

```text
RuntimeError: Optix call `optixInit()` failed with 7804 (OPTIX_ERROR_LIBRARY_NOT_FOUND): Library not found
```

This usually means that your system's **CUDA driver is too old** and does not include the required OptiX libraries. Make sure you have a recent NVIDIA GPU driver installed that supports OptiX 7.4+ and CUDA 12.x. Check your driver version with `nvidia-smi` and update from [NVIDIA's official driver download page](https://www.nvidia.com/Download/index.aspx) if necessary.

### nvcc fatal : Unsupported gpu architecture 'compute_120' or 'compute_121'

If the build in step 3 ends with:

```text
-- CMAKE_CUDA_ARCHITECTURES not defined, setting it to `native`
...
nvcc fatal   : Unsupported gpu architecture 'compute_121'
```

the container's CUDA toolkit is older than the GPU. `native` asks nvcc to target
whatever GPU it can see, and CUDA 12.6 only goes up to sm_90, so it rejects
Blackwell. Rebuild the image with CUDA 13, which is now the default:

```bash
docker build --build-arg CUDA_MAJOR=13 -f .devcontainer/Dockerfile -t ultrasound_simulator .
```

To build for one specific architecture instead of whatever is installed, pass it
explicitly and skip the `native` detection altogether:

```bash
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=121" python -m pip install -e ".[all]" \
  --config-settings=build-dir=build-docker
```
