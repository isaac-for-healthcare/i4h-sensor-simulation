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
import json
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import raysim.cuda as rs
from tqdm import tqdm
from monai.transforms import LoadImage


def calculate_bounding_box(segmentation_path):
    with open(segmentation_path, 'r') as file:
        points = json.load(file)

    h_coords = [point[1] for point in points]
    w_coords = [point[0] for point in points]

    min_h = min(h_coords)
    max_h = max(h_coords)
    min_w = min(w_coords)
    max_w = max(w_coords)

    return min_h, max_h, min_w, max_w, points

def get_pose_position(image, bounding_box):
    min_h, max_h, min_w, max_w = bounding_box
    spacing_w, spacing_h = np.abs(image.affine[0, 0]), np.abs(image.affine[1, 1])
    real_min_h, real_max_h = min_h * spacing_h, max_h * spacing_h
    real_min_w, real_max_w = min_w * spacing_w, max_w * spacing_w
    print(bounding_box)
    # assume position h is 20mm smaller than real_min_h
    position_h = real_min_h - 20
    # position_h = 0
    # assume position w is the median of real_min_w and real_max_w
    position_w = (real_min_w + real_max_w) / 2
    # assume position d is 50mm
    position_d = 50
    return position_h, position_w, position_d


rc_output_dir = "/localhome/local-vennw/code/rc_to_us_dataset/rc/train"
us_output_dir = "/localhome/local-vennw/code/rc_to_us_dataset/us/train"

os.makedirs(rc_output_dir, exist_ok=True)
os.makedirs(us_output_dir, exist_ok=True)


# Add liver mesh to world
image_type = "Normal"
data_dir = "/localhome/local-vennw/code/i4h-sensor-simulation/ultrasound-raytracing/examples/raytracing_to_us/zenodo_dataset/"

idx_list = os.listdir(os.path.join(data_dir, image_type, "fake_3d_masks"))
idx_list = [i for i in idx_list if i.endswith(".nii.gz")]
idx_list = [str(i.split(".")[0]) for i in idx_list]


for image_idx in idx_list:

    liver_seg_path = os.path.join(data_dir, image_type, "segmentation", "liver", f"{image_idx}.json")
    outline_seg_path = os.path.join(data_dir, image_type, "segmentation", "outline", f"{image_idx}.json")
    image_3d_path = os.path.join(data_dir, image_type, "fake_3d_masks", f"{image_idx}.nii.gz")
    image_2d_path = os.path.join(data_dir, image_type, "image", f"{image_idx}.jpg")
    mesh_dir = os.path.join(data_dir, image_type, "fake_3d_masks", image_idx, "obj")

    image = LoadImage()(image_3d_path)
    original_w, original_h = image.shape[1], image.shape[0]
    spacing_w, spacing_h = np.abs(image.affine[0, 0]), np.abs(image.affine[1, 1])

    min_h, max_h, min_w, max_w, points = calculate_bounding_box(liver_seg_path)
    min_h_outline, max_h_outline, min_w_outline, max_w_outline, points_outline = calculate_bounding_box(outline_seg_path)
    position_h, position_w, position_d = get_pose_position(image, (min_h, max_h, min_w, max_w))

    # Create materials and world
    materials = rs.Materials()
    world = rs.World("water")

    mesh_configs = [
        ("Liver.obj", "liver"),
        # ("Outline.obj", "fat"),
    ]
    for mesh_config in mesh_configs:
        material_idx = materials.get_index(mesh_config[1])
        mesh = rs.Mesh(os.path.join(mesh_dir, mesh_config[0]), material_idx)
        world.add(mesh)

    # Create simulator
    simulator = rs.RaytracingUltrasoundSimulator(world, materials)

    # Configure simulation parameters
    sim_params = rs.SimParams()
    sim_params.conv_psf = True
    sim_params.buffer_size = 4096
    sim_params.t_far = 180.0
    sim_params.enable_cuda_timing = True
    sim_params.b_mode_size = (500, 500,)

    # Image dynamic range
    min_val = -60.0
    max_val = 0.0

    image_2d = cv2.imread(image_2d_path)
    mask = np.zeros_like(image_2d)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], (0, 255, 0))

    outline_mask = np.zeros_like(image_2d)
    cv2.fillPoly(outline_mask, [np.array(points_outline, dtype=np.int32)], (1, 1, 1))
    image_2d = image_2d * outline_mask

    for i, variable in enumerate([1]):
        # Create probe with updated pose
        position = np.array([position_h, position_w * -1, position_d], dtype=np.float32)
        print(position)
        rotation = np.array([-np.pi/2, 0, -np.pi/2], dtype=np.float32)
        probe = rs.UltrasoundProbe(rs.Pose(position=position, rotation=rotation), radius=45)

        # Run simulation
        b_mode_image = simulator.simulate(probe, sim_params)
        b_mode_image = np.clip((b_mode_image - min_val) / (max_val - min_val), 0, 1)

        b_mode_image = (b_mode_image * 255).astype(np.uint8)

        # check imge mask range
        rc_min_h = np.where(b_mode_image != 0)[0].min()
        rc_max_h = np.where(b_mode_image != 0)[0].max()
        rc_min_w = np.where(b_mode_image != 0)[1].min()
        rc_max_w = np.where(b_mode_image != 0)[1].max()

        
        b_mode_image = b_mode_image[rc_min_h:rc_max_h, rc_min_w:rc_max_w]
        b_mode_image = cv2.resize(b_mode_image, (int(max_w) - int(min_w), int(max_h) - int(min_h)))
        b_mode_image = np.repeat(b_mode_image[:, :, np.newaxis], 3, axis=2)
        # add outline background to b_mode_image
        b_mode_background = np.zeros_like(image_2d)
        b_mode_background[int(min_h):int(max_h), int(min_w):int(max_w)] = b_mode_image
        # cv2.polylines(b_mode_background, [np.array(points_outline, dtype=np.int32)], isClosed=True, color=(255, 255, 255), thickness=1)

        cv2.imwrite(os.path.join(rc_output_dir, f"{image_type}_{image_idx}.png"), b_mode_background)

        cv2.imwrite(os.path.join(us_output_dir, f"{image_type}_{image_idx}.png"), image_2d)

        # # expand b_mode_background to (h, w, 3)
        # b_mode_background = np.repeat(b_mode_background[:, :, np.newaxis], 3, axis=2)
        # image_concat = np.concatenate([image_2d, mask, b_mode_background], axis=1)
        # print(image_2d.shape, mask.shape, b_mode_background.shape)
        # cv2.imwrite(os.path.join(concat_output_dir, f"{image_type}_{image_idx}.png"), image_concat)

