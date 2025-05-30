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

import io
import os
import sys

# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import raysim.cuda as rs
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# Create materials and world
materials = rs.Materials()
world = rs.World("water")

material_idx = materials.get_index("fat")
liver_tumor = rs.Mesh("mesh/Tumor1.obj", material_idx)
world.add(liver_tumor)
material_idx = materials.get_index("water")
liver_cyst = rs.Mesh("mesh/Tumor2.obj", material_idx)
world.add(liver_cyst)
# Add liver mesh to world
material_idx = materials.get_index("liver")
liver_mesh = rs.Mesh("mesh/Liver.obj", material_idx)
world.add(liver_mesh)
material_idx = materials.get_index("fat")
skin_mesh = rs.Mesh("mesh/Skin.obj", material_idx)
world.add(skin_mesh)
material_idx = materials.get_index("bone")
bone_mesh = rs.Mesh("mesh/Bone.obj", material_idx)
world.add(bone_mesh)
material_idx = materials.get_index("water")
vessels_mesh = rs.Mesh("mesh/Vessels.obj", material_idx)
world.add(vessels_mesh)
material_idx = materials.get_index("water")
galbladder_mesh = rs.Mesh("mesh/Gallbladder.obj", material_idx)
world.add(galbladder_mesh)
material_idx = materials.get_index("liver")
spleen_mesh = rs.Mesh("mesh/Spleen.obj", material_idx)
world.add(spleen_mesh)
material_idx = materials.get_index("liver")
heart_mesh = rs.Mesh("mesh/Heart.obj", material_idx)
world.add(heart_mesh)
material_idx = materials.get_index("water")
stomach_mesh = rs.Mesh("mesh/Stomach.obj", material_idx)
world.add(stomach_mesh)
material_idx = materials.get_index("liver")
pancreas_mesh = rs.Mesh("mesh/Pancreas.obj", material_idx)
world.add(pancreas_mesh)
material_idx = materials.get_index("water")
small_intestine_mesh = rs.Mesh("mesh/Small_bowel.obj", material_idx)
world.add(small_intestine_mesh)
material_idx = materials.get_index("water")
large_intestine_mesh = rs.Mesh("mesh/Colon.obj", material_idx)
world.add(large_intestine_mesh)

# Initial poses for different probe types
initial_poses = {
    "curvilinear": rs.Pose(
        np.array([-14, -122, 72], dtype=np.float32),  # position (x, y, z)
        np.array([np.deg2rad(-90), np.deg2rad(180), np.deg2rad(0)], dtype=np.float32),
    ),
    "linear": rs.Pose(
        np.array([-14, -122, 72], dtype=np.float32),  # position (x, y, z)
        np.array([np.deg2rad(-90), np.deg2rad(180), np.deg2rad(0)], dtype=np.float32),
    ),
    "phased": rs.Pose(
        np.array([-14, -122, 72], dtype=np.float32),  # position (x, y, z)
        np.array([np.deg2rad(-90), np.deg2rad(180), np.deg2rad(0)], dtype=np.float32),
    ),
}

# Create probes with different geometries
probes = {
    "curvilinear": rs.CurvilinearProbe(  # Original curvilinear probe
        initial_poses["curvilinear"],
        num_elements_x=256,  # Number rays which represent elements
        sector_angle=73.0,  # Field of view in degrees
        radius=45.0,  # probe radius in mm
        frequency=5.0,  # probe frequency in MHz
        elevational_height=7.0,  # probe elevational aperture height in mm
        num_el_samples=10,  # number of samples in elevational direction (default is 1)
    ),
    "linear": rs.LinearArrayProbe(  # Linear array probe
        initial_poses["linear"],
        num_elements_x=256,  # Number of elements
        width=50.0,  # Width of the array in mm
        frequency=7.5,  # probe frequency in MHz
        elevational_height=5.0,  # probe elevational aperture height in mm
        num_el_samples=10,  # number of samples in elevational direction
    ),
    "phased": rs.PhasedArrayProbe(  # Phased array probe
        initial_poses["phased"],
        num_elements_x=128,  # Number of elements
        width=20.0,  # Width of the array in mm
        sector_angle=90.0,  # Full sector angle in degrees
        frequency=3.5,  # probe frequency in MHz
        elevational_height=5.0,  # probe elevational aperture height in mm
        num_el_samples=10,  # number of samples in elevational direction
    ),
}

# Current active probe
active_probe = "curvilinear"

# Create simulator
simulator = rs.RaytracingUltrasoundSimulator(world, materials)

# Configure simulation parameters
sim_params = rs.SimParams()
sim_params.conv_psf = True
sim_params.buffer_size = 4096
sim_params.t_far = 180.0
sim_params.enable_cuda_timing = True
sim_params.pepper_vertical_filter = False


@app.route("/")
def home():
    return send_file("templates/index.html")


@app.route("/get_probe_types", methods=["GET"])
def get_probe_types():
    # Return list of available probe types
    return jsonify(list(probes.keys()))


@app.route("/set_probe_type", methods=["POST"])
def set_probe_type():
    global active_probe
    probe_type = request.json["probe_type"]
    if probe_type in probes:
        active_probe = probe_type
        return {"status": "success", "probe_type": active_probe}
    else:
        return {"status": "error", "message": f"Unknown probe type: {probe_type}"}, 400


@app.route("/get_initial_pose", methods=["GET"])
def get_initial_pose():
    current_pose = probes[active_probe].get_pose()
    # Convert the pose to a list for JSON serialization
    pose_list = current_pose.position.tolist() + current_pose.rotation.tolist()
    return {"pose": pose_list, "probe_type": active_probe}


@app.route("/get_sim_params", methods=["GET"])
def get_sim_params():
    """Get current simulation parameters"""
    return {
        "pepper_vertical_filter": sim_params.pepper_vertical_filter,
        "enable_cuda_timing": sim_params.enable_cuda_timing,
        "write_debug_images": sim_params.write_debug_images
    }


@app.route("/set_sim_params", methods=["POST"])
def set_sim_params():
    """Update simulation parameters"""
    try:
        params = request.json

        if "pepper_vertical_filter" in params:
            sim_params.pepper_vertical_filter = bool(params["pepper_vertical_filter"])

        if "enable_cuda_timing" in params:
            sim_params.enable_cuda_timing = bool(params["enable_cuda_timing"])

        if "write_debug_images" in params:
            sim_params.write_debug_images = bool(params["write_debug_images"])

        return {
            "status": "success",
            "params": {
                "pepper_vertical_filter": sim_params.pepper_vertical_filter,
                "enable_cuda_timing": sim_params.enable_cuda_timing,
                "write_debug_images": sim_params.write_debug_images
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}, 400


@app.route("/simulate", methods=["POST"])
def simulate():
    pose_delta = request.json["pose_delta"]
    print(f"Applying delta: {pose_delta}")

    # Get current pose
    current_pose = probes[active_probe].get_pose()

    # Apply deltas to current pose
    position_delta = np.array(pose_delta[0:3], dtype=np.float32)
    rotation_delta = np.array(pose_delta[3:6], dtype=np.float32)

    # Calculate new pose by adding deltas
    new_position = current_pose.position + position_delta
    new_rotation = current_pose.rotation + rotation_delta

    # Set new pose
    new_pose = rs.Pose(new_position, new_rotation)
    probes[active_probe].set_pose(new_pose)

    b_mode_image = simulator.simulate(probes[active_probe], sim_params)

    # Apply normalization as in C++ code
    min_val = -60.0  # Matching C++ min_max.x
    max_val = 0.0  # Matching C++ min_max.y
    normalized_image = np.clip((b_mode_image - min_val) / (max_val - min_val), 0, 1)

    # Convert to 8-bit image for display
    img_uint8 = (normalized_image * 255).astype(np.uint8)

    # Convert to PIL Image
    pil_img = Image.fromarray(img_uint8)

    img_io = io.BytesIO()
    pil_img.save(img_io, "PNG")
    img_io.seek(0)

    # Return the image and the current probe type
    return send_file(img_io, mimetype="image/png")


if __name__ == "__main__":
    app.run(port=8000, debug=True)
