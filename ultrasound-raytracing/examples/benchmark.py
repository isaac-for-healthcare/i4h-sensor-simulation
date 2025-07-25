# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import subprocess
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import raysim.cuda as rs


def get_gpu_info():
    """Get GPU information using nvidia-smi."""
    try:
        # First check if nvidia-smi is available
        subprocess.run(["which", "nvidia-smi"], check=True, capture_output=True)

        # Get detailed GPU info
        nvidia_smi = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=gpu_name,memory.total,driver_version",
                    "--format=csv,nounits,noheader",
                ]
            )
            .decode()
            .strip()
        )

        # Take only the first line if multiple GPUs
        first_gpu = nvidia_smi.split("\n")[0]

        # Split only on the last two commas since GPU name might contain commas
        parts = first_gpu.split(",")
        if len(parts) < 3:
            return "GPU information incomplete (unexpected nvidia-smi output format)"

        # Reconstruct GPU name from all parts except last two
        gpu_name = ",".join(parts[:-2]).strip()
        memory = parts[-2].strip()
        driver = parts[-1].strip()

        # Add units back to memory
        try:
            memory_val = float(memory)
            if memory_val >= 1024:
                memory = f"{memory_val/1024:.1f} GB"
            else:
                memory = f"{memory_val} MB"
        except ValueError:
            memory = f"{memory} MB"  # Fallback if parsing fails

        return f"{gpu_name} ({memory}, Driver: {driver})"
    except subprocess.CalledProcessError:
        return "GPU information unavailable: nvidia-smi not found"
    except Exception as e:
        return f"GPU information unavailable: {str(e)}"


def get_cpu_info():
    """Get CPU information."""
    try:
        with open("/proc/cpuinfo") as f:
            cpu_info = f.read()
        model_name = re.search(r"model name\s+:\s+(.*)", cpu_info).group(1)
        cpu_cores = len(re.findall(r"processor\s+:", cpu_info))
        return f"{model_name} ({cpu_cores} cores)"
    except Exception as e:
        return f"CPU information unavailable: {e}"


def read_previous_fps(results_file):
    """Read the average FPS from a previous benchmark results file."""
    try:
        with open(results_file, "r") as f:
            content = f.read()
            # Extract average FPS using regular expression
            match = re.search(r"Average FPS: (\d+\.\d+)", content)
            if match:
                return float(match.group(1))
    except FileNotFoundError:
        # No previous results exist
        pass
    return 0.0  # Return 0 if file doesn't exist or FPS couldn't be extracted


def main():
    # Create materials and world
    materials = rs.Materials()
    world = rs.World("water")

    # Add liver mesh to world
    material_idx = materials.get_index("liver")
    mesh = rs.Mesh("mesh/Liver.obj", material_idx)
    world.add(mesh)

    # Create probe with initial pose
    initial_pose = rs.Pose(
        np.array([40.0, -110.0, -300.0], dtype=np.float32),  # position (x, y, z)
        np.array(
            [np.deg2rad(-84.0), np.deg2rad(22.0), np.deg2rad(0.0)], dtype=np.float32
        ),  # rotation (x, y, z
    )
    probe = rs.CurvilinearProbe(initial_pose, num_elements_x=256)

    # Create simulator
    simulator = rs.RaytracingUltrasoundSimulator(world, materials)

    # Configure simulation parameters
    sim_params = rs.SimParams()
    sim_params.conv_psf = True
    sim_params.buffer_size = 4096
    sim_params.t_far = 180.0
    sim_params.enable_cuda_timing = (
        False  # Set to True to enable CUDA profiling of processings steps
    )
    sim_params.b_mode_size = (
        500,
        500,
    )

    # Setup benchmark parameters
    N_frames = 200
    z_range = -300.0  # Range of z movement
    z_start = -400.0

    # Prepare to measure frame times
    frame_times = []

    print(f"Starting benchmark with {N_frames} frames...")

    for i in tqdm(range(N_frames), desc="Simulating frames"):
        # Calculate z position with a smooth oscillation to simulate realistic movement
        z = z_start + (z_range / 2) * np.sin(i * 0.1)

        # Create probe with updated pose
        position = np.array([40.0, -110.0, z], dtype=np.float32)
        rotation = np.array(
            [np.deg2rad(-84.0), np.deg2rad(22.0), np.deg2rad(0.0)], dtype=np.float32
        )
        probe = rs.CurvilinearProbe(rs.Pose(position=position, rotation=rotation))

        # Time the simulation
        start_time = time.time()
        _ = simulator.simulate(probe, sim_params)
        end_time = time.time()

        # Record frame time
        frame_time = end_time - start_time
        frame_times.append(frame_time)

    # Calculate statistics
    frame_times = np.array(frame_times)
    avg_frame_time = np.mean(frame_times)
    avg_fps = 1.0 / avg_frame_time
    min_fps = 1.0 / np.max(frame_times)
    max_fps = 1.0 / np.min(frame_times)

    # Create benchmark results report string (used for both console and file output)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    results_report = f"""Benchmark Results:
        Total frames: {N_frames}
        Average frame time: {avg_frame_time:.4f} seconds
        Average FPS: {avg_fps:.2f}
        Minimum FPS: {min_fps:.2f}
        Maximum FPS: {max_fps:.2f}
        Date: {timestamp}

        System Information:
        GPU: {get_gpu_info()}
        CPU: {get_cpu_info()}
        """

    # Report results to console
    print(f"\n{results_report}")

    # Define results file path
    results_file = "benchmark_results.txt"

    # Check if previous results exist and compare performance
    previous_fps = read_previous_fps(results_file)

    if avg_fps > previous_fps:
        # New results are better, save them
        with open(results_file, "w") as f:
            f.write(results_report)

        print(f"New high score! Results saved to {results_file}")
        print(f"Previous best: {previous_fps:.2f} FPS, New best: {avg_fps:.2f} FPS")
    else:
        if previous_fps > 0:
            print(
                f"Results not saved. Previous best ({previous_fps:.2f} FPS) outperforms current result ({avg_fps:.2f} FPS)"
            )
        else:
            print("No previous results found. Creating new benchmark file.")
            with open(results_file, "w") as f:
                f.write(results_report)
            print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
