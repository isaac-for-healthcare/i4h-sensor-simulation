import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import raysim.cuda as rs


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
sim_params.b_mode_size = (500, 500,)

# Setup sweep parameters
N_frames = 10
z_start = 100.0
z_end = 200.0
z_positions = np.linspace(z_start, z_end, N_frames)

# Image dynamic range
min_val = -60.0
max_val = 0.0

for i, z in tqdm(enumerate(z_positions), total=len(z_positions)):
    # Create probe with updated pose
    position = np.array([-310.0, -420.0, z], dtype=np.float32)
    rotation = np.array([np.pi, np.pi, np.pi/2], dtype=np.float32)
    probe = rs.UltrasoundProbe(rs.Pose(position=position, rotation=rotation))

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
    plt.imshow(normalized_image, cmap='gray',
            extent=[min_x, max_x, min_z, max_z], aspect='auto')  # Note: depth axis is flipped
    plt.xlabel('Width (mm)')
    plt.ylabel('Depth (mm)')
    plt.colorbar(label='Intensity (normalized)')
    plt.title(f"B-mode Ultrasound Image of Liver: (x, y, z) = ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
    plt.savefig(f"liver_sweep_frame_{i:03d}.png")
    plt.show()
