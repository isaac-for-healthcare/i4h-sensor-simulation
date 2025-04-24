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
import cv2

# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# isort: off
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import raysim.cuda as rs
from tqdm import tqdm

output_dir = "liver_sweep"
os.makedirs(output_dir, exist_ok=True)

# Create materials and world
materials = rs.Materials()

all_mesh_configs = [
    ("Tumor1.obj", "fat"),
    ("Tumor2.obj", "water"),
    ("Liver.obj", "liver"),
    ("Skin.obj", "fat"),
    # ("Bone.obj", "bone"),
    ("Vessels.obj", "water"),
    ("Gallbladder.obj", "water"),
    ("Spleen.obj", "liver"),
    # ("Heart.obj", "liver"),
    ("Stomach.obj", "water"),
    ("Pancreas.obj", "liver"),
    ("Small_bowel.obj", "water"),
    ("Colon.obj", "water"),
]

liver_mesh_configs = [
    ("Liver.obj", "liver"),
]

def prepare_world_with_mesh(mesh_config):
    world = rs.World("water")
    for mesh_file, material_name in mesh_config:
        material_idx = materials.get_index(material_name)
        mesh = rs.Mesh(f"mesh/{mesh_file}", material_idx)
        world.add(mesh)
    return world


# Create probe with initial pose matching C++ implementation
initial_pose = rs.Pose(
    np.array([10, -145, -361.0], dtype=np.float32),  # position (x, y, z)
    np.array([-np.pi/2, 0, 0], dtype=np.float32))   # rotation (x, y, z)
probe = rs.UltrasoundProbe(initial_pose)

# Create simulator
world_liver = prepare_world_with_mesh(liver_mesh_configs)
simulator_liver = rs.RaytracingUltrasoundSimulator(world_liver, materials)
world_all = prepare_world_with_mesh(all_mesh_configs)
simulator_all = rs.RaytracingUltrasoundSimulator(world_all, materials)

# Configure simulation parameters
sim_params = rs.SimParams()
sim_params.conv_psf = True
sim_params.buffer_size = 4096
sim_params.t_far = 180.0
sim_params.enable_cuda_timing = True
sim_params.b_mode_size = (512, 512,)
sim_params.boundary_value = float("-inf") # float("inf") to produce white background

scan_area_liver = simulator_liver.generate_scan_area(
    probe,
    sim_params.t_far,
    sim_params.b_mode_size,
    inside_value=float("inf"),
    outside_value=float("-inf")
)
scan_area_liver_uint8 = np.zeros(scan_area_liver.shape, dtype=np.uint8)
scan_area_liver_uint8[scan_area_liver == float("inf")] = 255

# cv2.imwrite(os.path.join(output_dir, f"scan_area.png"), scan_area_uint8)

# Setup sweep parameters
N_frames = 10
z_start = -30
z_end = 110
z_positions = np.linspace(z_start, z_end, N_frames)

# Image dynamic range
min_val = -60.0
max_val = 0.0

for i, z in tqdm(enumerate(z_positions), total=len(z_positions)):
    # Create probe with updated pose
    position = np.array([-30, -104, z], dtype=np.float32)
    rotation = np.array([-np.pi/2, np.pi, 0], dtype=np.float32)
    probe = rs.UltrasoundProbe(rs.Pose(position=position, rotation=rotation))

    # Run simulation
    b_mode_image = simulator_liver.simulate(probe, sim_params)


    normalized_image = np.clip((b_mode_image - min_val) / (max_val - min_val), 0, 1)

    normalized_image = (normalized_image * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, f"frame_{i:03d}.png"), normalized_image)

    b_mode_image_all = simulator_all.simulate(probe, sim_params)
    normalized_image_all = np.clip((b_mode_image_all - min_val) / (max_val - min_val), 0, 1)
    normalized_image_all = (normalized_image_all * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, f"frame_{i:03d}_all.png"), normalized_image_all)

    # get contour of normalized_image
    contours, _ = cv2.findContours(normalized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)

        rc_mask = np.zeros_like(normalized_image)
        cv2.drawContours(rc_mask, [main_contour], -1, (255), thickness=cv2.FILLED)

        # rc with scan area
        output_image = np.zeros_like(normalized_image)
        # for each pixel in output_image, if it is in scan area, set it to 255
        output_image[scan_area_liver_uint8 == 255] = 255
        # if it's also in rc_mask, set it to normalized_image value
        output_image[(rc_mask == 255) & (scan_area_liver_uint8 == 255)] = normalized_image[(rc_mask == 255) & (scan_area_liver_uint8 == 255)]

        # save filled_mask
        cv2.imwrite(os.path.join(output_dir, f"frame_{i:03d}_output.png"), output_image)
