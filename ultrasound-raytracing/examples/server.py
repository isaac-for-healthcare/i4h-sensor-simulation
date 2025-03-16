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
from flask import Flask, request, send_file
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)


# Create materials and world
materials = rs.Materials()
world = rs.World("water")

origin = rs.Mesh("mesh/origin.obj", materials.get_index("fat"))
world.add(origin)
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



# Create probe with initial pose matching C++ implementation
initial_pose = rs.Pose(
    np.array([-8,-110,-336], dtype=np.float32),  # position (x, y, z)
    np.array([np.deg2rad(-84), np.deg2rad(22), np.deg2rad(0)], dtype=np.float32))   # rotation (x, y, z)
# Create ultrasound probe with elevational extent parameters
probe = rs.UltrasoundProbe(
    initial_pose,
    num_elements=256,            # Number rays which represent elements
    opening_angle=73.0,          # default value in degrees
    radius=45.0,                 # probe radius  in mm
    frequency=2.5,               # probe frequency in MHz
    elevational_height=7.0,      # probe elevational aperture height in mm
    num_el_samples=10            # number of samples in elevational direction (default is 1)
)
# Create simulator
simulator = rs.RaytracingUltrasoundSimulator(world, materials)

# Configure simulation parameters
sim_params = rs.SimParams()
sim_params.conv_psf = True
sim_params.buffer_size = 4096
sim_params.t_far = 180.0
sim_params.enable_cuda_timing = True

@app.route('/')
def home():
    return send_file('templates/index.html')

@app.route('/get_initial_pose', methods=['GET'])
def get_initial_pose():
    current_pose = probe.get_pose()
    # Convert the pose to a list for JSON serialization
    pose_list = current_pose.position.tolist() + current_pose.rotation.tolist()
    return {'pose': pose_list}


@app.route('/simulate', methods=['POST'])
def simulate():
    pose_delta = request.json['pose_delta']
    print(f"Applying delta: {pose_delta}")

    # Get current pose
    current_pose = probe.get_pose()

    # Apply deltas to current pose
    position_delta = np.array(pose_delta[0:3], dtype=np.float32)
    rotation_delta = np.array(pose_delta[3:6], dtype=np.float32)

    # Calculate new pose by adding deltas
    new_position = current_pose.position + position_delta
    new_rotation = current_pose.rotation + rotation_delta

    # Set new pose
    new_pose = rs.Pose(new_position, new_rotation)
    probe.set_pose(new_pose)

    b_mode_image = simulator.simulate(probe, sim_params)


    # Apply normalization as in C++ code
    min_val = -60.0  # Matching C++ min_max.x
    max_val = 0.0    # Matching C++ min_max.y
    normalized_image = np.clip((b_mode_image - min_val) / (max_val - min_val), 0, 1)

    # Convert to 8-bit image for display
    img_uint8 = (normalized_image * 255).astype(np.uint8)

    # Convert to PIL Image
    pil_img = Image.fromarray(img_uint8)

    img_io = io.BytesIO()
    pil_img.save(img_io, 'PNG')
    img_io.seek(0)

    # We could add a json response with the current pose, but for simplicity
    # we'll just keep the image response as is. The frontend can get the latest
    # pose from the /get_initial_pose endpoint if needed
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(port=8000, debug=True)
