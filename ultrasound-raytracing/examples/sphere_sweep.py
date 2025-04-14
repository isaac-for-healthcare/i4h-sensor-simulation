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

# Use non-interactive matplotlib backend to avoid Qt/XCB issues
# isort: off
import matplotlib
matplotlib.use('Agg')
# isort: on
import matplotlib.pyplot as plt
import numpy as np
import raysim.cuda as rs
from tqdm import tqdm


def main():
    # Create materials and world
    materials = rs.Materials()
    world = rs.World("liver")

    # Add spheres to world
    material_idx = materials.get_index("water")
    sphere1 = rs.Sphere(np.array([0, 0, -145], dtype=np.float32), 40, material_idx)
    sphere2 = rs.Sphere(np.array([0, 0, -225], dtype=np.float32), 40, material_idx)
    world.add(sphere1)
    world.add(sphere2)

    # Create output directory
    output_dir = 'ultrasound_sweep'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create simulator
    simulator = rs.RaytracingUltrasoundSimulator(world, materials)

    # Configure simulation parameters
    sim_params = rs.SimParams()
    sim_params.conv_psf = True
    sim_params.buffer_size = 4096
    sim_params.t_far = 180.0
    sim_params.enable_cuda_timing = True
    sim_params.b_mode_size = (500, 500)

    # Setup sweep parameters
    N_frames = 10
    x_start = 20
    x_end = -20
    x_positions = np.linspace(x_start, x_end, N_frames)


    # Image dynamic range
    min_val = -60.0
    max_val = 0.0
    for i, x in tqdm(enumerate(x_positions), total=len(x_positions)):
        # Create probe with updated pose
        position = np.array([x, 0, 0], dtype=np.float32)
        rotation = np.array([0, np.pi, 0], dtype=np.float32)
        probe = rs.UltrasoundProbe(rs.Pose(position=position, rotation=rotation))

        # Run simulation
        b_mode_image = simulator.simulate(probe, sim_params)

        normalized_image = np.clip((b_mode_image - min_val) / (max_val - min_val), 0, 1)

        min_x = simulator.get_min_x()
        max_x = simulator.get_max_x()
        min_z = simulator.get_min_z()
        max_z = simulator.get_max_z()

        # Display and save image with proper axes
        plt.figure(figsize=(10, 8))
        plt.imshow(normalized_image, cmap='gray',
                    extent=[min_x, max_x, min_z, max_z], aspect='equal')
        plt.title(f"B-mode Ultrasound Image\nPosition: ({x:.2f}, 0.00, 0.00), Rotation: (0.00°, 180.00°, 0.00°)")
        plt.xlabel('Width (mm)')
        plt.ylabel('Depth (mm)')
        plt.colorbar(label='Intensity (normalized)')

        # Save figure
        filename = os.path.join(output_dir, f"frame_{i:03d}.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()



if __name__ == "__main__":
    main()
