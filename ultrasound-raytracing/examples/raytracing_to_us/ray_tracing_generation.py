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

import cv2


def keep_largest_connected_component(np_img, threshold=1, mask=None, apply_background_mask=False):
    _, binary_img = cv2.threshold(np_img, threshold, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=4)

    if num_labels <= 1:  # Only background or no foreground components
        return np_img

    largest_component_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1 # Add 1 because we skipped the background

    if mask is not None and apply_background_mask:
        is_largest_component = (labels == largest_component_label)

        # Condition 2: Corresponding pixel in the mask is 255
        is_mask_allowed = (mask == 255)

        # Combine conditions: Use np_img value only if both are true
        use_np_img_condition = np.logical_and(is_largest_component, is_mask_allowed)

        # Initialize the output image with the mask values (this handles the 'otherwise' case)
        output_img = mask.copy()

        # Where the condition is true, update the output image with values from np_img
        output_img[use_np_img_condition] = np_img[use_np_img_condition]

        return output_img
    
    else:
        np_img[labels != largest_component_label] = 0

        return np_img



def calculate_bounding_box(segmentation_path):
    mask = cv2.imread(segmentation_path)

    target_color_green_bgr = np.array([0, 255, 0])
    target_color_white_bgr = np.array([255, 255, 255])

    liver_mask_boolean = np.all(mask == target_color_green_bgr, axis=2)
    liver_mask = liver_mask_boolean.astype(np.uint8) * 255

    # Create a boolean mask where pixels are white
    outline_mask_boolean = np.all(mask == target_color_white_bgr, axis=2)

    combined_mask_boolean = np.logical_or(liver_mask_boolean, outline_mask_boolean)

    combined_mask = combined_mask_boolean.astype(np.uint8) * 255

    non_zero_indices = np.argwhere(liver_mask > 0)
    min_coords = non_zero_indices.min(axis=0)
    max_coords = non_zero_indices.max(axis=0)

    min_h = int(min_coords[0])
    max_h = int(max_coords[0])
    min_w = int(min_coords[1])
    max_w = int(max_coords[1])

    return min_h, max_h, min_w, max_w, liver_mask, combined_mask

def get_pose_position(image, bounding_box):
    min_h, max_h, min_w, max_w = bounding_box
    spacing_w, spacing_h = np.abs(image.affine[0, 0]), np.abs(image.affine[1, 1])
    real_min_h, real_max_h = min_h * spacing_h, max_h * spacing_h
    real_min_w, real_max_w = min_w * spacing_w, max_w * spacing_w
    # assume position h is 30mm smaller than real_min_h
    position_h = real_min_h - 30
    # position_h = 0
    # assume position w is the median of real_min_w and real_max_w
    position_w = (real_min_w + real_max_w) / 2
    # assume position d is 50mm
    position_d = 50
    return position_h, position_w, position_d


rc_output_dir = "/localhome/local-vennw/code/tcia_us_dataset/rc/train"
us_output_dir = "/localhome/local-vennw/code/tcia_us_dataset/us/train"

os.makedirs(rc_output_dir, exist_ok=True)
os.makedirs(us_output_dir, exist_ok=True)


# Add liver mesh to world
data_dir = "/localhome/local-vennw/code/tcia_us_dataset"

idx_list = os.listdir(os.path.join(data_dir, "fake3d_masks_add_border"))
idx_list = [i for i in idx_list if i.endswith(".nii.gz")]
idx_list = [str(i.split(".")[0]) for i in idx_list]


for image_idx in idx_list:

    liver_seg_path = os.path.join(data_dir, "ray_tracing_masks", f"{image_idx}.png")
    image_3d_path = os.path.join(data_dir, "fake3d_masks_add_border", f"{image_idx}.nii.gz")
    image_2d_path = os.path.join(data_dir, "ray_tracing_images", f"{image_idx}.png")
    mesh_dir = os.path.join(data_dir, "fake3d_masks_add_border", image_idx, "obj")

    image = LoadImage()(image_3d_path)
    original_w, original_h = image.shape[1], image.shape[0]
    spacing_w, spacing_h = np.abs(image.affine[0, 0]), np.abs(image.affine[1, 1])

    min_h, max_h, min_w, max_w, liver_mask, mask = calculate_bounding_box(liver_seg_path)
    position_h, position_w, position_d = get_pose_position(image, (min_h, max_h, min_w, max_w))

    # Create materials and world
    materials = rs.Materials()
    world = rs.World("water")

    mesh_configs = [
        ("Liver.obj", "marker"),
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

    image_2d = cv2.imread(image_2d_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(os.path.join(us_output_dir, f"{image_idx}.png"), image_2d)

    # Create probe with updated pose
    position = np.array([position_h, position_w * -1, position_d], dtype=np.float32)
    rotation = np.array([-np.pi/2, 0, -np.pi/2], dtype=np.float32)
    probe = rs.UltrasoundProbe(rs.Pose(position=position, rotation=rotation), radius=120)

    # Run simulation
    b_mode_image = simulator.simulate(probe, sim_params)
    b_mode_image = np.clip((b_mode_image - min_val) / (max_val - min_val), 0, 1)

    b_mode_image = (b_mode_image * 255).astype(np.uint8)
    # b_mode_image = keep_largest_connected_component(b_mode_image)

    # # check imge mask range
    # rc_min_h = np.where(b_mode_image != 0)[0].min()
    # rc_max_h = np.where(b_mode_image != 0)[0].max()
    # rc_min_w = np.where(b_mode_image != 0)[1].min()
    # rc_max_w = np.where(b_mode_image != 0)[1].max()

    # b_mode_image = b_mode_image[rc_min_h:rc_max_h, rc_min_w:rc_max_w]
    # b_mode_image = cv2.resize(b_mode_image, (int(max_w) - int(min_w), int(max_h) - int(min_h)))

    # b_mode_image_mask_shape = np.zeros_like(mask)
    # b_mode_image_mask_shape[min_h:max_h, min_w:max_w] = b_mode_image

    # b_mode_image_mask_shape = keep_largest_connected_component(b_mode_image_mask_shape, mask=mask, apply_background_mask=True)

    cv2.imwrite(os.path.join(rc_output_dir, f"{image_idx}.png"), b_mode_image)
    break
