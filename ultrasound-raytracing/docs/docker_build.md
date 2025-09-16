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

```bash
cd ultrasound-raytracing  # ensure you're in the correct directory
# Download and link the mesh data (~527MB)
i4h-asset-retrieve --download-dir assets --sub-path Props/ABDPhantom/Organs --version 0.2.0
# Note: The hash below is specific to version 0.2.0
ln -s assets/8c0bf782eab2f44f1cc82da60eb10f6be8f941406d291b7fbfbdb53c05b3d149/Props/ABDPhantom/Organs mesh
```

> The mesh data will be available inside the container through the volume mount.

---
## 3 · Start a container and set up the project (done after every `git pull`)

```bash
docker run -it --rm --gpus all --runtime=nvidia --network host \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$(pwd)":/raysim  \
  --workdir /raysim      \
  raysim

# INSIDE the container – copy/paste:
sudo apt-get update && sudo apt-get install -y python3.12-venv
python3 -m venv .venv && \
source .venv/bin/activate && \
pip install -e . && \
pip install git+https://github.com/isaac-for-healthcare/i4h-asset-catalog.git

cmake -S . -B build -DPYTHON_EXECUTABLE=$PWD/.venv/bin/python \
      -DCMAKE_BUILD_TYPE=Release && \
cmake --build build -j$(nproc)
```

This creates a local virtual-env, installs all Python dependencies, and
builds the CUDA/OptiX binary.  You only need to repeat these few lines
when the source tree changes.

---
## 4 · Run an example

```bash
# still inside the same container (venv active)
python examples/sphere_sweep.py         # head-less benchmark
# OR
python examples/server.py               # web-GUI on port 8000
```

Now open your browser **on the host** at:

```
http://localhost:8000
```

If you are on a different machine within the same network, replace
`localhost` with the host's IP address (e.g. `http://192.168.1.42:8000`).

> **Running on a Remote Server**: If you're running this on a remote server, you can forward the port to your local machine using SSH:
> ```bash
> ssh -L 8000:localhost:8000 username@remote-server
> ```
> Then access the web interface at `http://localhost:8000` on your local machine.
