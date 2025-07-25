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
import sys

# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# isort: off
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import raysim.cuda as rs
from tqdm import tqdm

output_dir = "liver_sweep"
os.makedirs(output_dir, exist_ok=True)

# Create materials and world
materials = rs.Materials()
world = rs.World("water")

# Add liver mesh to world
material_idx = materials.get_index("liver")
mesh = rs.Mesh("mesh/Liver.obj", material_idx)
world.add(mesh)

# Create probe with initial pose matching C++ implementation
initial_pose = rs.Pose(
    np.array([10, -145, -361.0], dtype=np.float32),  # position (x, y, z)
    np.array([-np.pi / 2, 0, 0], dtype=np.float32),
)  # rotation (x, y, z)
probe = rs.CurvilinearProbe(initial_pose)

# Create simulator
simulator = rs.RaytracingUltrasoundSimulator(world, materials)

# Configure simulation parameters
sim_params = rs.SimParams()
sim_params.conv_psf = True
sim_params.buffer_size = 4096
sim_params.t_far = 180.0
sim_params.enable_cuda_timing = True
sim_params.b_mode_size = (
    500,
    500,
)

# Setup sweep parameters
N_frames = 100
z_start = -30
z_end = 110
z_positions = np.linspace(z_start, z_end, N_frames)

# Image dynamic range
min_val = -60.0
max_val = 0.0

for i, z in tqdm(enumerate(z_positions), total=len(z_positions)):
    # Create probe with updated pose
    position = np.array([-30, -104, z], dtype=np.float32)
    rotation = np.array([-np.pi / 2, np.pi, 0], dtype=np.float32)
    probe = rs.CurvilinearProbe(rs.Pose(position=position, rotation=rotation))

    # Run simulation
    b_mode_image = simulator.simulate(probe, sim_params)

    normalized_image = np.clip((b_mode_image - min_val) / (max_val - min_val), 0, 1)

    # Get boundary values from the result dictionary
    min_x = simulator.get_min_x()
    max_x = simulator.get_max_x()
    min_z = simulator.get_min_z()
    max_z = simulator.get_max_z()

    # Display and save image with proper axes
    plt.figure(figsize=(10, 8))
    plt.imshow(
        normalized_image,
        cmap="gray",
        extent=[min_x, max_x, min_z, max_z],
        aspect="equal",
    )  # Note: depth axis is flipped
    plt.xlabel("Width (mm)")
    plt.ylabel("Depth (mm)")
    plt.colorbar(label="Intensity (normalized)")
    plt.title(
        f"B-mode Ultrasound Image of Liver: (x, y, z) = ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
    )
    plt.savefig(os.path.join(output_dir, f"frame_{i:03d}.png"))
    plt.show()
