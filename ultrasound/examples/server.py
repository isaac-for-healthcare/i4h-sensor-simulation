import io
import os
import sys

import numpy as np
from flask import Flask, request, send_file
from flask_cors import CORS
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import raysim.cuda as rs

app = Flask(__name__)
CORS(app)


# Create materials and world
materials = rs.Materials()
world = rs.World("water")

# Add liver mesh to world
material_idx = materials.get_index("liver")
mesh = rs.Mesh("mesh/Liver.obj", material_idx)
world.add(mesh)

# Create probe with initial pose matching C++ implementation
initial_pose = rs.Pose(
    np.array([-310.0, -420.0, 200.0], dtype=np.float32),  # position (x, y, z)
    np.array([np.pi, np.pi, np.pi/2], dtype=np.float32)   # rotation (y, ?, x) z-up by default
)
probe = rs.UltrasoundProbe(initial_pose)
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
    app.run(port=8600, debug=True)
