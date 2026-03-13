# Raysim – Quick Docker Guide

Run the ultrasound-raytracing simulator on **any Linux host with an NVIDIA GPU** in three short steps.

---

## 1 · Build the image (done once)

```bash
cd ultrasound-raytracing
# ~10 min on first run, <3 GB compressed
docker build \
  --build-arg USER_UID=$(id -u) \
  --build-arg USER_GID=$(id -g) \
  -f .devcontainer/Dockerfile \
  -t raysim \
  .
```

> The image already contains CUDA 12.6, Python 3.12, CMake and a non-root
> user called `raysim`.

---

## 2 · Download mesh data (done once)

Follow the steps described in [ultrasound-raytracing README](../README.md) to install the mesh assets within the project environment.

```text
```

> The mesh data will be available inside the container through the volume mount.

---

## 3 · Start a container and set up the project (done after every `git pull`)

```bash
cd ultrasound-raytracing # ensure you are in the sensor directory

docker run -it --rm --gpus all --network host \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$(pwd)":/raysim  \
  --workdir /raysim      \
  raysim
```

Inside the container, create a virtual environment and install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

With the virtual environment active, build the native extension:

```bash
cmake -S . -B build -DPYTHON_EXECUTABLE=$PWD/.venv/bin/python \
      -DCMAKE_BUILD_TYPE=Release && \
cmake --build build -j$(nproc)
```

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
