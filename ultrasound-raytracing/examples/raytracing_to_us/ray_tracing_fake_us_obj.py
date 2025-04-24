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
import trimesh
import cv2
import pyvista as pv
from scipy.spatial.transform import Rotation as R # Import SciPy Rotation


import cv2
import numpy as np
import os

import cv2
import numpy as np
import os


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


def find_main_contour_via_lcc(
    b_mode_image_uint8_raw, # Input: Raw B-mode image (uint8, 0-255)
    output_dir,             # Where to save visualization
    image_idx,              # Identifier for saving files
    manual_threshold=50,    # Fallback manual threshold value (NEEDS TUNING)
    ):
    """
    Finds the contour of the largest connected component after thresholding a B-mode image.

    Args:
        b_mode_image_uint8_raw (np.ndarray): Raw grayscale B-mode image (0-255, uint8).
        output_dir (str): Directory to save visualization image.
        image_idx (str): Identifier for naming the saved image.
        manual_threshold (int): Manual threshold value if Otsu is disabled or fails.
                                 Needs tuning per image/dataset.
        use_otsu (bool): If True, attempts Otsu's automatic thresholding first.

    Returns:
        np.ndarray: The contour (NumPy array of points) of the largest connected
                    component found. Returns None if no component is found or an error occurs.
    """
    print(f"Finding main contour via LCC for {image_idx}...")
    if b_mode_image_uint8_raw is None or b_mode_image_uint8_raw.ndim != 2:
        print("Error: Invalid input B-mode image.")
        return None

    # --- 1. Thresholding ---

    # Force manual threshold if use_otsu is False
    print(f"  Using manual threshold: {manual_threshold}")
    _, binary_image = cv2.threshold(
        b_mode_image_uint8_raw, manual_threshold, 255, cv2.THRESH_BINARY
    )
    threshold_value_used = manual_threshold

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=4)

    if num_labels <= 1: # Only background found
        print("  No foreground components found after thresholding.")
        return None

    # Find the label of the largest component (ignoring background label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) # Add 1 because we slice stats[1:]

    # Create a mask containing only the largest component
    lcc_mask = np.zeros_like(binary_image)
    lcc_mask[labels == largest_label] = 255

    # --- 3. Find Contour of the Largest Component Mask ---
    contours, hierarchy = cv2.findContours(
        lcc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        main_contour = max(contours, key=cv2.contourArea)

        filled_mask = np.zeros_like(b_mode_image_uint8_raw)
        cv2.drawContours(filled_mask, [main_contour], -1, (255), thickness=cv2.FILLED)

        return filled_mask
    else:
        return None


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
    liver_mesh_obj_path = os.path.join(mesh_dir, "Liver.obj")

    image = LoadImage()(image_3d_path)
    spacing_w, spacing_h = image.affine[0, 0], image.affine[1, 1]
    mesh = trimesh.load_mesh(os.path.join(mesh_dir, "Liver.obj"))
    centroid = mesh.centroid
    position_w, position_d = centroid[1], centroid[2]
    # for position_h, need to add an offset
    opening_angle = 73.0
    radius = 45
    bounds = mesh.bounds
    liver_height = bounds[1][0] - bounds[0][0]
    liver_width = bounds[1][1] - bounds[0][1]
    offset = 30
    position_h = bounds[0][0] - offset
    

    # Create materials and world
    materials = rs.Materials()
    world = rs.World("water")

    mesh_configs = [
        ("Liver.obj", "liver"),
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
    sim_params.t_far = liver_height + 50
    sim_params.enable_cuda_timing = True
    sim_params.b_mode_size = (500, 500,)

    # # assume opening angle is 73 degrees, we need to ensure t_far is large enough
    # min_t_far = liver_width / (2 * np.sin(opening_angle / 2.0 * np.pi / 180.0)) - radius
    # print(f"min_t_far: {min_t_far}")
    # Image dynamic range
    min_val = -60.0
    max_val = 0.0

    image_2d = cv2.imread(image_2d_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(os.path.join(us_output_dir, f"{image_idx}.png"), image_2d)

    # Create probe with updated pose
    position = np.array([position_h, position_w, position_d], dtype=np.float32)
    rotation = np.array([-np.pi/2, 0, -np.pi/2], dtype=np.float32)
    probe_pose_obj = rs.Pose(position=position, rotation=rotation)
    probe_sim = rs.UltrasoundProbe(probe_pose_obj, radius=radius, opening_angle=opening_angle)

    # Run simulation
    b_mode_image_raw_float = simulator.simulate(probe_sim, sim_params)

    b_mode_image_normalized = np.clip((b_mode_image_raw_float - min_val) / (max_val - min_val), 0, 1)

    b_mode_image_uint8_raw = (b_mode_image_normalized * 255).astype(np.uint8)

    # resize image to align the original us image's h, w ratio
    opening_angle_rad = opening_angle * np.pi / 180.0
    sim_physical_width = 2 * (radius + sim_params.t_far) * np.sin(opening_angle_rad / 2.0)
    sim_physical_depth = sim_params.t_far
    
    target_pixel_width = int(np.round(sim_physical_width / np.abs(spacing_w)))
    target_pixel_height = int(np.round(sim_physical_depth / np.abs(spacing_h)))
    b_mode_image_uint8_raw = cv2.resize(b_mode_image_uint8_raw, (target_pixel_width, target_pixel_height))
    
    filled_mask = find_main_contour_via_lcc(
        b_mode_image_uint8_raw=b_mode_image_uint8_raw,
        output_dir=rc_output_dir,
        image_idx=image_idx,
        manual_threshold=1,
    )

    rc_min_h = np.where(filled_mask != 0)[0].min()
    rc_max_h = np.where(filled_mask != 0)[0].max()
    rc_min_w = np.where(filled_mask != 0)[1].min()
    rc_max_w = np.where(filled_mask != 0)[1].max()

    min_h, max_h, min_w, max_w, us_liver_mask, us_mask = calculate_bounding_box(liver_seg_path)

    b_mode_image_crop = b_mode_image_uint8_raw[rc_min_h:rc_max_h, rc_min_w:rc_max_w]
    filled_mask_crop = filled_mask[rc_min_h:rc_max_h, rc_min_w:rc_max_w]

    b_mode_image_crop = cv2.resize(b_mode_image_crop, (int(max_w) - int(min_w), int(max_h) - int(min_h)))
    filled_mask_crop = cv2.resize(filled_mask_crop, (int(max_w) - int(min_w), int(max_h) - int(min_h)))

    us_shape_b_mode = np.zeros_like(us_mask)
    us_shape_b_mode[min_h:max_h, min_w:max_w] = b_mode_image_crop
    us_shape_b_mode_mask = np.zeros_like(us_mask)
    us_shape_b_mode_mask[min_h:max_h, min_w:max_w] = filled_mask_crop

    is_us_mask = (us_mask == 255)
    if_b_mode_liver = (us_shape_b_mode_mask == 255)
    add_b_mode_condition = np.logical_and(is_us_mask, if_b_mode_liver)

    us_mask[add_b_mode_condition] = us_shape_b_mode[add_b_mode_condition]

    cv2.imwrite(os.path.join(rc_output_dir, f"{image_idx}.png"), us_mask)
