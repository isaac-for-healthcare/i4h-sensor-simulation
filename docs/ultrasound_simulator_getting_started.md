# Ultrasound Simulator Getting Started Guide

This comprehensive tutorial will guide you through the ultrasound raytracing simulator using a progressive, hands-on approach. Each example builds on the previous ones, introducing new concepts step by step.

> 💡 **For deeper technical details**, refer to the [Ultrasound Simulator Technical Guide](ultrasound_simulator_technical_guide.md)

## Prerequisites and Installation

Before starting the tutorial, you need to properly build and install the ultrasound raytracing simulator.

> 📖 **Complete Installation Guide**: For detailed installation instructions including OptiX setup and mesh data, see the [main README](../ultrasound-raytracing/README.md)

---

## Chapter 1: Hello, Ultrasound!

**Goal**: Get your first ultrasound simulation running in under 5 minutes

Let's start with the simplest working ultrasound simulation - a single sphere in water using a linear probe. Linear probes are ideal for learning because they produce clean, rectangular images with parallel beams, making it easier to understand geometric interface reflections without the complexity of curved beam patterns.

> ⚠️ **Note**: The simulator requires at least one object in the world to function properly. Empty worlds cause OptiX acceleration structure errors.

### The Code

```python
import numpy as np
import matplotlib.pyplot as plt


import raysim.cuda as rs

# Step 1: Create materials registry
materials = rs.Materials()

# Step 2: Create a world with water background
world = rs.World("water")

# Step 3: Add a simple sphere (currently required for simulator to work)
material_id = materials.get_index("water")  # High contrast material
sphere = rs.Sphere(
    np.array([0, 0, 100], dtype=np.float32),  # Center: 50mm in front of probe
    20,                                       # Radius: 20mm
    material_id                               # Material ID
)
world.add(sphere)

# Step 4: Create a simple probe
pose = rs.Pose(
    position=[0., 0., 0.],   # Probe position in world
    rotation=[0., 0., 0.]    # Probe orientation
)
probe = rs.LinearArrayProbe(pose,
                       frequency=2, # MHz
                       )

# Step 5: Create the simulator
simulator = rs.RaytracingUltrasoundSimulator(world, materials)

# Step 6: Configure and run the simulation
sim_params = rs.SimParams()
sim_params.conv_psf = True
sim_params.b_mode_size = (1500, 1500)
image = simulator.simulate(probe, sim_params)

# Step 7: Display the result with proper scaling
# Normalize the image for display (typical ultrasound dynamic range)
min_val = -60.0  # dB
max_val = 0.0    # dB
normalized_image = np.clip((image - min_val) / (max_val - min_val), 0, 1)

min_x = simulator.get_min_x()
max_x = simulator.get_max_x()
min_z = simulator.get_min_z()
max_z = simulator.get_max_z()


plt.figure(figsize=(10, 8))
plt.imshow(
    normalized_image,
    cmap='gray',
    extent=[min_x, max_x, min_z, max_z],  # Use physical coordinates
    aspect='auto'
)
plt.title("Your First Ultrasound Simulation")
plt.xlabel("Width (mm)")
plt.ylabel("Depth (mm)")
plt.colorbar(label="Normalized Intensity")

# Add annotation showing the sphere location
plt.savefig("Getting_Started_Chapter_1.png")
plt.show()

print("Success! You've created your first ultrasound simulation!")
print(f"Image dimensions: {image.shape}")
print(f"Value range: {image.min():.2f} to {image.max():.2f}")
print(f"Sphere material: water (same as background - geometric interface)")
```

### Expected Output

You should see a bright, curved echo in the image! This might seem surprising since both the sphere and background are water (same material), but the sphere appears as a bright arc because:

1. **Geometric Interface**: The simulator detects the geometric boundary of the sphere surface, regardless of material properties
2. **Specular Reflection**: Any geometric surface generates specular reflections based on surface geometry and incident angles
3. **Physical vs. Simulation Behavior**: In reality, identical materials with perfect boundaries would produce no reflection since there would be no boundary by definition. In our ray-tracing implementation treats geometric boundaries as reflective surfaces to immetate imperfect boundaries created by material gaps and textures.

### What Just Happened?

Let's break down the simulation pipeline:

1. **Materials Registry** (`rs.Materials()`): This creates a database of tissue types with their acoustic properties like speed of sound, acoustic impedance, and attenuation.

2. **World Creation** (`rs.World("water")`): This sets up the 3D environment for simulation with water as the background medium.

3. **Object Addition**: We add a water sphere to the world:
   - **Material**: "water" (same as background) - this tests geometric vs. material-based reflections
   - **Center** `[0, 0, 100]`: Centered laterally, 100mm deep
   - **Radius** `20`: Moderate size for clear visualization

4. **Probe Setup**:
   - **Position** `[0, 0, 0]`: The probe is at the world origin
   - **Rotation** `[0, 0, 0]`: The probe points in the positive Z direction (deeper into the scene)
   - **LinearArrayProbe**: This creates a linear array probe that produces rectangular images with parallel beams
   - **Frequency** `2 MHz`: Lower frequency provides better penetration for deep structures

5. **Simulation**: The raytracing engine sends ultrasound rays from the probe into the world. When rays hit the sphere, and the hits are recorded in the RF buffer to indicate an interface

6. **Image Formation**: The simulator converts the echo data into a 2D grayscale image where brightness represents relative echo intensity.

### Understanding the Coordinate System

The ultrasound coordinate system follows these conventions:
- **X-axis (Lateral)**: Left-right across the probe face
- **Y-axis (Elevational)**: Slice thickness direction (perpendicular to image plane)
- **Z-axis (Axial)**: Depth direction, positive Z points away from the probe

> 📖 **Learn More**: For detailed coordinate system explanation, see [Technical Guide - Coordinate System](ultrasound_simulator_technical_guide.md#coordinate-system)

### Understanding Frequency Selection

Ultrasound frequency is a critical parameter that affects image quality:

| Frequency | Penetration | Resolution | Clinical Use |
|-----------|-------------|------------|--------------|
| 2-3 MHz | Excellent | Low | Deep abdominal imaging |
| 3-5 MHz | Good | Medium | General abdominal, cardiac |
| 5-7 MHz | Medium | Good | Superficial structures |
| 7-15 MHz | Poor | Excellent | Small parts, vascular |

We chose **2 MHz** for this tutorial because it provides good penetration for demonstrating interface reflections, is appropriate for linear probe applications, and balances penetration with resolution for educational purposes.

### Understanding dB Scale in Ultrasound

Ultrasound images use **decibel (dB) scaling** for display because:

1. **Huge Dynamic Range**: Raw ultrasound signals can span 6+ orders of magnitude
2. **Logarithmic Perception**: Human vision perceives intensity changes logarithmically
3. **Clinical Standard**: All clinical ultrasound systems use dB compression

**dB Conversion Formula**:
```python
image_db = 20 * log10(intensity / reference)
```

**Typical Display Range**:
- **0 dB**: Maximum intensity (brightest)
- **-60 dB**: Minimum displayed intensity (darkest)
- **Below -60 dB**: Noise floor (appears black)

This creates the characteristic ultrasound appearance where:
- Strong reflectors appear bright (near 0 dB)
- Weak echoes appear gray (-20 to -40 dB)
- No echoes appear dark (below -60 dB)

### Understanding Geometric vs. Material-Based Reflections

This example reveals an important aspect of the ray-tracing simulator:

**Physical Reality**: Identical materials with perfect boundaries would produce no reflection since there would be no boundary by definition.

**Simulation Behavior**: Our ray-tracing implementation treats geometric boundaries as reflective surfaces to imitate imperfect boundaries created by material gaps and textures.

**Practical Implications**: Geometric boundaries always create some reflection, representing realistic imperfections rather than idealized perfect material interfaces.

This chapter introduced the basic simulation pipeline: World → Objects → Probe → Simulator → Image. The simulator requires at least one object to create the OptiX acceleration structure, and any geometric boundary creates reflections regardless of material match. The simulator prioritizes surface geometry over pure acoustic physics, the rectangular field of view comes from the linear probe design, and lower frequencies penetrate deeper but with lower resolution.

---

## Chapter 2: Understanding Materials

**Goal**: Explore how different materials affect ultrasound appearance

Now that you've seen a water sphere in liver background, let's explore how different tissue types appear in ultrasound. We'll create a row of spheres with different materials to see their varying echo patterns.

### The Code

Building on Chapter 1, we'll replace the single sphere with multiple spheres of different materials:

```python
import numpy as np
import matplotlib.pyplot as plt

import raysim.cuda as rs

# Steps 1-2: Same as Chapter 1
materials = rs.Materials()
world = rs.World("water")

# NEW: Create a row of spheres with different materials
materials_to_test = ["water", "fat", "liver", "muscle", "bone"]
positions = [
    [-24, 0, 60], [-12, 0, 60], [0, 0, 60], [12, 0, 60], [24, 0, 60]
]

print("Creating spheres with different materials:")
for i, (material_name, pos) in enumerate(zip(materials_to_test, positions)):
    material_id = materials.get_index(material_name)
    sphere = rs.Sphere(
        np.array(pos, dtype=np.float32),  # Center position
        5,                                # Radius (5mm for compact arrangement)
        material_id                       # Material ID
    )
    world.add(sphere)
    print(f"  {i+1}. {material_name} sphere at {pos}")

# Steps 3-6: Same as Chapter 1
pose = rs.Pose(
    position=[0., 0., 0.],   # Probe position in world
    rotation=[0., 0., 0.]    # Probe orientation
)
probe = rs.LinearArrayProbe(pose,
                       frequency=2, # MHz
                       )

simulator = rs.RaytracingUltrasoundSimulator(world, materials)

# Configure simulation parameters (same as Chapter 1)
sim_params = rs.SimParams()
sim_params.conv_psf = True
sim_params.b_mode_size = (1500, 1500)
sim_params.t_far = 120.0
sim_params.buffer_size = 4096
image = simulator.simulate(probe, sim_params)

# Normalize the image for display (typical ultrasound dynamic range)
min_val = -60.0  # dB
max_val = 0.0    # dB
normalized_image = np.clip((image - min_val) / (max_val - min_val), 0, 1)

# Get physical coordinates
# We multiply a factor of 2 to account for a scan conversion.
min_x = simulator.get_min_x() * 2
max_x = simulator.get_max_x() * 2
min_z = simulator.get_min_z()
max_z = simulator.get_max_z()

# Display with enhanced visualization
plt.figure(figsize=(12, 8))
plt.imshow(
    normalized_image,
    cmap='gray',
    extent=[min_x, max_x, max_z, min_z],
    aspect='auto'
)
plt.title("Ultrasound Image: Material Comparison")
plt.xlabel("Width (mm)")
plt.ylabel("Depth (mm)")
plt.colorbar(label="Normalized Intensity")

# Add labels for each material
colors = ['blue', 'green', 'orange', 'red', 'purple']
sphere_positions = [-24, -12, 0, 12, 24]  # Physical positions in mm

# Calculate relative positions within the image extent
x_range = max_x - min_x
z_range = max_z - min_z

for i, (material_name, color, x_pos) in enumerate(zip(materials_to_test, colors, sphere_positions)):
    # Convert physical position to image coordinates
    x_img = min_x + (x_pos - min_x) * (x_range / x_range)  # This simplifies to just x_pos
    z_img = min_z + 0.9 * z_range  # Place text at 90% depth

    plt.axvline(x=x_pos, color=color, linestyle='--', alpha=0.7, linewidth=1)
    plt.text(x_pos, z_img, material_name, color=color, fontsize=9, rotation=0,
             ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.savefig("Getting_Started_Chapter_2_1.png")
plt.show()

print("Material comparison complete!")
print("Notice how different materials create different echo intensities:")
for material in materials_to_test:
    print(f"  - {material}: {'Very bright' if material == 'bone' else 'Bright' if material in ['muscle'] else 'Medium' if material in ['liver', 'fat'] else 'Dark'}")
```

### Expected Output

You should now see **five different spheres** across the image, each with different echo intensities! From left to right, you'll notice:

1. **Water sphere**: Nearly invisible (same as background)
2. **Fat sphere**: Dim echo (low impedance contrast)
3. **Liver sphere**: Medium echo with speckled texture
4. **Muscle sphere**: Brighter echo
5. **Bone sphere**: Very bright echo with a distorted reflection (highest impedance contrast)

### What's New?

The key changes are:

**Multiple Materials**:
```python
materials_to_test = ["water", "fat", "liver", "muscle", "bone"]
```
We test a spectrum of tissue types from lowest to highest acoustic impedance.

**Spatial Arrangement**:
```python
positions = [[-24, 0, 60], [-12, 0, 60], [0, 0, 60], [12, 0, 60], [24, 0, 60]]
```
We arrange the spheres in a row at 60mm depth, spaced 12mm apart laterally (center-to-center), with 5mm radius so they have 2mm clearance between them and all fit within the probe's 60mm field of view.

**Loop Creation**:
```python
for material_name, pos in zip(materials_to_test, positions):
    material_id = materials.get_index(material_name)
    sphere = rs.Sphere(np.array(pos, dtype=np.float32), 15, material_id)
    world.add(sphere)
```
This efficiently creates multiple objects with different properties.

### Understanding Material Properties

The echo brightness depends on **acoustic impedance contrast**:

| Material | Acoustic Impedance | Echo Appearance | Clinical Use |
|----------|-------------------|------------------|--------------|
| Water | 1.48 MRayl | Dark (minimal reflection) | Cysts, fluid collections |
| Fat | 1.38 MRayl | Dark-medium | Subcutaneous tissue |
| Liver | ~1.65 MRayl | Medium with speckle | Organ parenchyma |
| Muscle | 1.70 MRayl | Medium-bright | Muscular structures |
| Bone | 7.80 MRayl | Very bright | Skeletal structures |

> 📖 **Learn More**: For details on acoustic impedance and reflection, see [Technical Guide - Material Properties](ultrasound_simulator_technical_guide.md#41-material-definition-and-properties)

### Experiment: Try Different Positions

Modify the sphere position to see how it moves in the image:

```python
# Try these different positions (one at a time):
# sphere = rs.Sphere([0, 0, 30], 5, material_idx)    # Shallower (closer to probe)
# sphere = rs.Sphere([0, 0, 80], 5, material_idx)    # Deeper
# sphere = rs.Sphere([20, 0, 60], 5, material_idx)   # To the right
# sphere = rs.Sphere([-20, 0, 60], 5, material_idx)  # To the left
```

This chapter demonstrated how objects with different acoustic properties create visible echoes, with high-impedance materials creating stronger echoes than low-impedance materials. Object position `[x, y, z]` maps to image location `[lateral, depth]`, primarily the front surface of objects creates the visible echo, and dense objects block ultrasound, creating shadows behind them.

### One More Thing: Background Material Effects

The background material significantly affects contrast and visibility. Let's compare the same spheres in a liver background versus water background to see how tissue context changes the ultrasound appearance.

```python
import numpy as np
import matplotlib.pyplot as plt
import raysim.cuda as rs

# Create comparison: Water background vs Liver background
backgrounds = ["water", "liver"]
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for bg_idx, background_material in enumerate(backgrounds):
    # Setup world with different background
    materials = rs.Materials()
    world = rs.World(background_material)  # KEY CHANGE: background material

    # Create the same row of spheres
    materials_to_test = ["water", "fat", "liver", "muscle", "bone"]
    positions = [
        [-24, 0, 60], [-12, 0, 60], [0, 0, 60], [12, 0, 60], [24, 0, 60]
    ]

    print(f"\nCreating spheres in {background_material} background:")
    for i, (material_name, pos) in enumerate(zip(materials_to_test, positions)):
        material_id = materials.get_index(material_name)
        sphere = rs.Sphere(
            np.array(pos, dtype=np.float32),  # Center position
            5,                                # Radius (5mm)
            material_id                       # Material ID
        )
        world.add(sphere)
        print(f"  {i+1}. {material_name} sphere at {pos}")

    # Probe and simulation setup
    pose = rs.Pose(position=[0., 0., 0.], rotation=[0., 0., 0.])
    probe = rs.LinearArrayProbe(pose, frequency=2)
    simulator = rs.RaytracingUltrasoundSimulator(world, materials)

    # Configure simulation parameters
    sim_params = rs.SimParams()
    sim_params.conv_psf = True
    sim_params.b_mode_size = (1500, 1500)
    sim_params.t_far = 120.0
    sim_params.buffer_size = 4096
    image = simulator.simulate(probe, sim_params)

    # Normalize the image for display
    min_val = -60.0  # dB
    max_val = 0.0    # dB
    normalized_image = np.clip((image - min_val) / (max_val - min_val), 0, 1)

    # Get physical coordinates with scan conversion factor
    min_x = simulator.get_min_x() * 2
    max_x = simulator.get_max_x() * 2
    min_z = simulator.get_min_z()
    max_z = simulator.get_max_z()

    # Display
    axes[bg_idx].imshow(
        normalized_image,
        cmap='gray',
        extent=[min_x, max_x, max_z, min_z],
        aspect='auto'
    )
    axes[bg_idx].set_title(f"Background: {background_material.title()}")
    axes[bg_idx].set_xlabel("Width (mm)")
    axes[bg_idx].set_ylabel("Depth (mm)")

    # Add material labels
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    sphere_positions = [-24, -12, 0, 12, 24]
    z_range = max_z - min_z

    for i, (material_name, color, x_pos) in enumerate(zip(materials_to_test, colors, sphere_positions)):
        z_img = min_z + 0.9 * z_range  # Place text at 90% depth
        axes[bg_idx].axvline(x=x_pos, color=color, linestyle='--', alpha=0.7, linewidth=1)
        axes[bg_idx].text(x_pos, z_img, material_name, color=color, fontsize=8, rotation=0,
                         ha='center', va='center',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig("Getting_Started_Chapter_2_2.png")
plt.show()

print("\nBackground Material Effects:")
print("Water Background:")
print("  - Water sphere: Invisible (no contrast)")
print("  - Fat sphere: Visible but dim")
print("  - Liver sphere: Medium brightness")
print("  - Muscle sphere: Bright")
print("  - Bone sphere: Very bright")
print("\nLiver Background:")
print("  - Water sphere: Dark (hypoechoic)")
print("  - Fat sphere: Dark (hypoechoic)")
print("  - Liver sphere: Invisible (no contrast)")
print("  - Muscle sphere: Slightly bright")
print("  - Bone sphere: Very bright")
```

### Understanding Contrast and Echogenicity

This comparison reveals important ultrasound principles:

**Contrast Dependency**: Echo visibility depends on the **impedance difference** between object and background, not absolute impedance values. A liver sphere is less visible in liver background but very visible in water background.

**Clinical Terminology**:
- **Hyperechoic**: Brighter than surrounding tissue (e.g., bone in any background)
- **Hypoechoic**: Darker than surrounding tissue (e.g., water sphere in liver background)
- **Isoechoic**: Same brightness as surrounding tissue (e.g., liver sphere in liver background)
- **Anechoic**: No internal echoes (e.g., pure fluid collections)

### Full Field of View Visualization

The liver background also demonstrates the full field of view with realistic tissue texture. Notice how the liver background shows:
- **Speckle pattern**: Characteristic grainy texture from microscopic scatterers
- **Depth-dependent attenuation**: Gradual signal loss with depth
- **Realistic tissue appearance**: More clinically representative than water background

### Seeing the Complete Scan-Converted Image

One of the most important advantages of using a speckled background like liver is that it **reveals the full extent of the scan-converted image**. Here's why this matters:

**Water Background Limitations**:
- Uniform black appearance makes it impossible to see image boundaries from the black image frame
- You can't tell where the actual ultrasound field of view ends
- The rectangular image extent is hidden
- Difficult to understand the probe's actual coverage area

**Liver Background Advantages**:
- **Visible field boundaries**: The speckle pattern clearly shows where the ultrasound beam reaches
- **Scan conversion geometry**: You can see the actual shape of the linear array's rectangular field of view
- **Depth penetration limits**: Speckle fades with depth, showing realistic beam attenuation
- **Lateral beam edges**: The sides of the image show where the probe's width ends

**What You Can Observe**:
1. **Rectangular field shape**: Linear arrays create rectangular imaging regions (unlike sector probes)
2. **Acoustic Shadowing**: Our ray model displays how refraction and frequency dependent attenuation can lead to shadowing artifacts below the spheres. Their pronounced apperacne is characteristic of deterministic ray-tracying ultrasound simulators.
3. **Penetration depth**: How far the ultrasound effectively travels (set by `t_far`)
4. **Image boundaries**: Clear distinction between "inside the beam" and "outside the beam"

This background comparison shows how material contrast drives ultrasound visibility, clinical terminology describes echogenicity relative to surrounding tissue, realistic backgrounds provide better clinical context, and the same object can appear completely different depending on the surrounding medium.

---

## Chapter 3: Scan Parameters and Image Control

**Goal**: Master the simulation parameters that control image quality, depth, and appearance

Now that you understand materials, let's explore how different scan parameters affect your ultrasound images. We'll use the same water sphere phantom from Chapter 1 but vary the simulation parameters to see their effects.

### Understanding Key Scan Parameters

The `SimParams` object controls how the ultrasound simulation runs:

- **`t_far`**: Maximum scan depth (time-of-flight limit in mm)
- **`buffer_size`**: Memory allocation for echo data
- **`b_mode_size`**: Output image dimensions (width, height)
- **`conv_psf`**: Point Spread Function convolution (should always be `True` for realistic images; only disable for debugging)
- **Frequency**: Probe frequency (penetration vs resolution trade-off)

### Experiment 1: Scan Depth Control

Let's see how `t_far` affects what we can see:

```python
import numpy as np
import matplotlib.pyplot as plt
import raysim.cuda as rs

# Setup (same as Chapter 1)
materials = rs.Materials()
world = rs.World("water")

# Add our test sphere
material_id = materials.get_index("water")
sphere = rs.Sphere(
    np.array([0, 0, 100], dtype=np.float32),  # 100mm deep
    40,                                       # 40mm radius
    material_id
)
world.add(sphere)

# Probe setup
pose = rs.Pose(position=[0., 0., 0.], rotation=[0., 0., 0.])
probe = rs.LinearArrayProbe(pose, frequency=2)
simulator = rs.RaytracingUltrasoundSimulator(world, materials)

# Test different scan depths
scan_depths = [120, 180, 240, 360]  # Different t_far values
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, t_far in enumerate(scan_depths):
    # Configure simulation with different scan depth
    sim_params = rs.SimParams()
    sim_params.conv_psf = True
    sim_params.b_mode_size = (1500, 1500)
    sim_params.t_far = t_far  # KEY PARAMETER: scan depth
    sim_params.buffer_size = 4096

    # Run simulation
    image = simulator.simulate(probe, sim_params)

    # Normalize for display
    min_val = -60.0
    max_val = 0.0
    normalized_image = np.clip((image - min_val) / (max_val - min_val), 0, 1)

    # Get coordinates
    min_x = simulator.get_min_x()
    max_x = simulator.get_max_x()
    min_z = simulator.get_min_z()
    max_z = simulator.get_max_z()

    # Display
    axes[i].imshow(
        normalized_image,
        cmap='gray',
        extent=[min_x, max_x, max_z, min_z],
        aspect='auto'
    )
    axes[i].set_title(f"Scan Depth: {t_far}mm")
    axes[i].set_xlabel("Width (mm)")
    axes[i].set_ylabel("Depth (mm)")

    # Mark sphere location
    axes[i].axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Sphere center')
    axes[i].legend()

plt.tight_layout()
plt.savefig("Getting_Started_Chapter_3_1_Depth.png")
plt.show()

print("Scan depth comparison:")
print("- t_far = 120mm: Sphere partially cut off (too shallow)")
print("- t_far = 180mm: Sphere fully visible")
print("- t_far = 240mm: Deeper, aspect ratio adjusts")
print("- t_far = 360mm: Deepest, aspect ratio adjusts further")
```

### Experiment 2: Image Resolution Control

The `b_mode_size` parameter controls output image resolution. Higher values give images with more pixels but take longer to compute, while lower values will lead to aliasing artifacts:

```python
# Single example - try modifying these values yourself!
sim_params = rs.SimParams()
sim_params.conv_psf = True
sim_params.b_mode_size = (1500, 1500)  # Try (500,500), (1000,1000), or (2000,2000)
sim_params.t_far = 180.0
sim_params.buffer_size = 4096

image = simulator.simulate(probe, sim_params)
# ... rest of display code same as above ...
```

**Try these values**: `(500,500)`, `(1500,1500)`, `(2000,2000)`.

### Experiment 3: PSF Convolution Effect

The Point Spread Function (PSF) convolution is an essential part of ultrasound simulation that models the finite resolution of real transducers. It should always be used when simulating ultrasound and can only be turned off for debugging purposes.

> 📖 **Learn More**: For detailed explanation of PSF physics and implementation, see [Technical Guide - Point Spread Function](ultrasound_simulator_technical_guide.md#11-basic-ultrasound-physics)

**Try this**: Take any previous simulation and change `sim_params.conv_psf = False` to see the dramatic difference. You'll notice:
- **PSF OFF**: Sharp, geometric boundaries with unrealistic appearance and streaking artifacts from undersampled envelope detection.
- **PSF ON**: Realistic ultrasound axial and lateral resolution.


### Experiment 4: Frequency Effects

Probe frequency creates a fundamental trade-off between penetration and resolution. Let's demonstrate this with a liver background that shows both scattering texture and depth-dependent attenuation:

```python
import numpy as np
import matplotlib.pyplot as plt
import raysim.cuda as rs

# Create liver background with some test objects
materials = rs.Materials()
world = rs.World("liver")  # Scattering background to show frequency effects

# Add a few spheres at different depths to test penetration
sphere_materials = ["water", "muscle", "bone"]
sphere_depths = [40, 80, 120]  # Different depths to test penetration

for i, (material_name, depth) in enumerate(zip(sphere_materials, sphere_depths)):
    material_id = materials.get_index(material_name)
    sphere = rs.Sphere(
        np.array([0, 0, depth], dtype=np.float32),  # Center at different depths
        8,                                          # 8mm radius
        material_id
    )
    world.add(sphere)

# Test different frequencies
frequencies = [1, 2, 5, 10]  # MHz
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, freq in enumerate(frequencies):
    # Create probe with different frequency
    pose = rs.Pose(position=[0., 0., 0.], rotation=[0., 0., 0.])
    probe_freq = rs.LinearArrayProbe(pose, frequency=freq)  # KEY PARAMETER: frequency
    simulator = rs.RaytracingUltrasoundSimulator(world, materials)

    # Configure simulation
    sim_params = rs.SimParams()
    sim_params.conv_psf = True
    sim_params.b_mode_size = (1500, 1500)
    sim_params.t_far = 150.0  # Moderate depth to show penetration differences
    sim_params.buffer_size = 4096

    # Run simulation
    image = simulator.simulate(probe_freq, sim_params)

    # Normalize for display
    min_val = -60.0
    max_val = 0.0
    normalized_image = np.clip((image - min_val) / (max_val - min_val), 0, 1)

    # Get coordinates with scan conversion factor
    min_x = simulator.get_min_x() * 2
    max_x = simulator.get_max_x() * 2
    min_z = simulator.get_min_z()
    max_z = simulator.get_max_z()

    # Display
    axes[i].imshow(
        normalized_image,
        cmap='gray',
        extent=[min_x, max_x, max_z, min_z],
        aspect='auto'
    )
    axes[i].set_title(f"Frequency: {freq} MHz")
    axes[i].set_xlabel("Width (mm)")
    axes[i].set_ylabel("Depth (mm)")

    # Mark sphere locations
    for depth in sphere_depths:
        axes[i].axhline(y=depth, color='red', linestyle='--', alpha=0.3, linewidth=1)

plt.tight_layout()
plt.savefig("Getting_Started_Chapter_3_4_Frequency.png")
plt.show()

print("Frequency comparison in liver background:")
print("- 1 MHz: Excellent penetration, coarse speckle, poor resolution")
print("- 2 MHz: Good penetration, moderate speckle resolution, moderate resolution")
print("- 5 MHz: Worsening penetration, fine speckle, good resolution")
print("- 10 MHz: Poor penetration, very fine speckle, excellent resolution")
print("\nNotice how:")
print("- Higher frequencies show finer speckle texture")
print("- Lower frequencies penetrate deeper with less attenuation")
print("- Deep spheres become harder to see at higher frequencies")
```

### Understanding Parameter Trade-offs

**Scan Depth (`t_far`)**:
Defines total the axial ray length, and thereby the axial extent of the simulated ultrasound image.

**Image Resolution (`b_mode_size`)**:
Defines the spatial pixel resolution (grid size) of the resulting output image.

**PSF Convolution (`conv_psf`)**:
A debugging parameter to turn of the PSF convolution operation. Not recommended for simulation use.

**Frequency**:
- **Low (1-3 MHz)**: Deep penetration, poor resolution, abdominal imaging
- **Medium (3-7 MHz)**: Balanced, general purpose imaging
- **High (7-15 MHz)**: Superficial structures, excellent resolution

**Buffer Size (`buffer_size`)**:
Defines the number of axial samples per ray where hits are written and convolved, modeling RF measurements in time.

This chapter demonstrated how scan parameters control image appearance and quality. The `t_far` parameter controls maximum imaging depth, `b_mode_size` affects image resolution and computation time, PSF convolution creates realistic ultrasound texture, frequency balances penetration versus resolution, and proper parameter selection depends on your imaging application and computational constraints.

---

## Chapter 4: Probe Types and Imaging Geometry

**Goal**: Understand how different probe types create different imaging patterns and field geometries

So far we've used linear array probes, which create rectangular images with parallel beams. But ultrasound systems use different probe types for different clinical applications. Let's explore the three main probe types and see how they create different imaging geometries.

### Understanding Probe Types

The simulator supports three main probe types, each optimized for different clinical scenarios:

| Probe Type | Field Shape | Best For | Beam Pattern |
|------------|-------------|----------|--------------|
| **Linear Array** | Rectangular | Superficial structures, vascular | Parallel beams |
| **Curvilinear** | Sector (curved) | Abdominal imaging | Diverging beams from curved surface |
| **Phased Array** | Sector (straight) | Cardiac, intercostal | Electronically steered (diverging) beams from a small flat surface |

### Experiment: Probe Type Comparison

Let's create the same phantom with all three probe types to see their different imaging characteristics:

```python
import numpy as np
import matplotlib.pyplot as plt
import raysim.cuda as rs

# Create a test phantom with multiple spheres
materials = rs.Materials()
world = rs.World("liver")

# Add spheres at different positions to show field geometry
sphere_positions = [
    [-20, 0, 60],   # Left
    [0, 0, 60],     # Center
    [20, 0, 60],    # Right
    [0, 0, 100],    # Deep center
]

for i, pos in enumerate(sphere_positions):
    material_id = materials.get_index("water")
    sphere = rs.Sphere(
        np.array(pos, dtype=np.float32),
        8,  # 8mm radius
        material_id
    )
    world.add(sphere)

# Create simulator once
simulator = rs.RaytracingUltrasoundSimulator(world, materials)

# Configure simulation parameters
sim_params = rs.SimParams()
sim_params.conv_psf = True
sim_params.b_mode_size = (1500, 1500)
sim_params.t_far = 150.0
sim_params.buffer_size = 4096

# Create three different probes
linear_probe = rs.LinearArrayProbe(
    rs.Pose(position=[0., 0., 0.], rotation=[0., 0., 0.]),
    frequency=5.0,
    width=50.0,  # 50mm width
    num_elements_x=128
)

curvilinear_probe = rs.CurvilinearProbe(
    rs.Pose(position=[0., 0., 0.], rotation=[0., 0., 0.]),
    frequency=3.5,
    radius=60.0,  # 60mm radius of curvature
    sector_angle=60.0,  # 60 degree sector
    num_elements_x=128
)

phased_probe = rs.PhasedArrayProbe(
    rs.Pose(position=[0., 0., 0.], rotation=[0., 0., 0.]),
    frequency=2.5,
    width=20.0,  # 20mm footprint
    sector_angle=90.0,  # 90 degree sector
    num_elements_x=64
)

probes = [linear_probe, curvilinear_probe, phased_probe]
probe_names = ['Linear Array', 'Curvilinear', 'Phased Array']
descriptions = [
    'Rectangular field, parallel beams',
    'Curved sector field, wide coverage',
    'Sector field from small footprint'
]

# Create comparison plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, (probe, probe_name, description) in enumerate(zip(probes, probe_names, descriptions)):
    # Run simulation
    image = simulator.simulate(probe, sim_params)

    # Normalize for display
    min_val = -60.0
    max_val = 0.0
    normalized_image = np.clip((image - min_val) / (max_val - min_val), 0, 1)

    # Get coordinates (note: different probes may have different coordinate systems)
    min_x = simulator.get_min_x() * 2
    max_x = simulator.get_max_x() * 2
    min_z = simulator.get_min_z()
    max_z = simulator.get_max_z()

    aspect_ratio = 'auto' if probe_name != 'Curvilinear' else 'equal' # Curvilinear probe has a different aspect ratio in it's scan conversion

    # Curvilinear dimensions conditional - swap min/max for proper display
    if probe_name == 'Curvilinear':
        max_z, min_z = min_z, max_z

    # Display
    axes[i].imshow(
        normalized_image,
        cmap='gray',
        extent=[min_x, max_x, max_z, min_z],
        aspect=aspect_ratio
    )
    axes[i].set_title(f"{probe_name}\n{description}")
    axes[i].set_xlabel("Width (mm)")
    axes[i].set_ylabel("Depth (mm)")

plt.tight_layout()
plt.savefig("Getting_Started_Chapter_4_1_Probe_Types.png")
plt.show()
```

### Understanding Field Geometry Differences

**Linear Array Characteristics**:
- **Field Shape**: Rectangular, uniform width
- **Beam Pattern**: Parallel beams perpendicular to probe face
- **Resolution**: Consistent lateral resolution at all depths
- **Coverage**: Limited to probe width, no expansion with depth
- **Clinical Use**: Vascular, superficial structures, musculoskeletal

**Curvilinear Characteristics**:
- **Field Shape**: Sector expanding with depth
- **Beam Pattern**: Diverging beams from curved transducer surface
- **Resolution**: Good near-field, decreasing lateral resolution with depth
- **Coverage**: Wide field of view, excellent for large organs
- **Clinical Use**: Abdominal, obstetric, general imaging

**Phased Array Characteristics**:
- **Field Shape**: Sector from small footprint
- **Beam Pattern**: Electronically steered beams
- **Resolution**: Variable, depends on steering angle
- **Coverage**: Wide sector from minimal surface contact
- **Clinical Use**: Cardiac, intercostal windows

### Experiment: Probe Parameter Effects

Let's see how key probe parameters affect the imaging:

```python
# ... World generation code from 4.1
# Example: Linear Array Width Comparison
widths = [30, 50, 70]  # Different probe widths in mm

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, width in enumerate(widths):
    probe = rs.LinearArrayProbe(
        rs.Pose(position=[0., 0., 0.], rotation=[0., 0., 0.]),
        frequency=5.0,
        width=width,
        num_elements_x=128
    )

    image = simulator.simulate(probe, sim_params)
    normalized_image = np.clip((image + 60.0) / 60.0, 0, 1)

    min_x = simulator.get_min_x() * 2
    max_x = simulator.get_max_x() * 2
    min_z = simulator.get_min_z()
    max_z = simulator.get_max_z()

    axes[i].imshow(normalized_image, cmap='gray',
                   extent=[min_x, max_x, max_z, min_z], aspect='auto')
    axes[i].set_title(f"Width: {width}mm")
    axes[i].set_xlabel("Width (mm)")
    axes[i].set_ylabel("Depth (mm)")

plt.tight_layout()
plt.savefig("Getting_Started_Chapter_4_2_Width_Comparison.png")
plt.show()
```

### Clinical Applications Guide

**When to Use Linear Arrays**:
- Superficial structures (< 6cm depth)
- Vascular imaging (carotid, peripheral vessels)
- Musculoskeletal applications
- Small parts (thyroid, breast, testicles)
- When uniform resolution is needed

**When to Use Curvilinear**:
- Abdominal imaging (liver, kidneys, gallbladder)
- Obstetric and gynecologic imaging
- Deep structures requiring wide field of view
- General purpose imaging

**When to Use Phased Array**:
- Cardiac imaging (echocardiography)
- Intercostal imaging (limited acoustic windows)
- Pediatric applications (small contact area)
- When small footprint is essential

This chapter demonstrated how different probe types create fundamentally different imaging geometries. Linear arrays provide uniform rectangular coverage, curvilinear probes offer wide sector views for deep imaging, and phased arrays enable sector imaging from small footprints. The choice of probe type significantly affects field of view, resolution characteristics, and clinical applicability. Probe geometries can be parametriszed to match real-world probes in curvature, sector angle, freqeuency, scan depth and size.

---

## Chapter 5: Building Complex Scenes

**Goal**: Learn to create realistic, complex ultrasound phantoms using meshes and multiple objects

So far we've worked with simple spheres, but real ultrasound imaging involves complex anatomical structures. The simulator can load 3D mesh files to create realistic organ phantoms. Let's build a complex abdominal scene step by step.

### Loading 3D Mesh Objects

> **NOTE:** Be sure to download the mesh assets as described in [README.md](../README.md) before starting with this section.

The simulator can load standard 3D mesh formats (OBJ files) to represent complex anatomical structures:

```python
import numpy as np
import matplotlib.pyplot as plt
import raysim.cuda as rs

# Create materials and world
materials = rs.Materials()
world = rs.World("water")  # Start with water background

# Load a liver mesh
liver_material_id = materials.get_index("liver")
liver_mesh = rs.Mesh("mesh/Liver.obj", liver_material_id)
world.add(liver_mesh)

# Create simulator
simulator = rs.RaytracingUltrasoundSimulator(world, materials)

# Use curvilinear probe for abdominal imaging
# IMPORTANT: Position the probe above the liver mesh (similar to real clinical positioning)
# When using mesh objects, the probe must be positioned correctly relative to the mesh
pose = rs.Pose(
    position=[-30., -104., 40.],   # Position above liver
    rotation=[-np.pi/2, np.pi, 0.]  # Orient probe to face downward into liver
)
probe = rs.CurvilinearProbe(
    pose,
    frequency=3.5,  # Good for abdominal imaging
    radius=60.0,
    sector_angle=70.0,
    num_elements_x=256
)

# Configure simulation for detailed imaging
sim_params = rs.SimParams()
sim_params.conv_psf = True
sim_params.b_mode_size = (2000, 2000)  # High resolution
sim_params.t_far = 200.0  # Deep imaging
sim_params.buffer_size = 4096

# Run simulation
image = simulator.simulate(probe, sim_params)

# Display with clinical-style presentation
min_val = -60.0
max_val = 0.0
normalized_image = np.clip((image - min_val) / (max_val - min_val), 0, 1)

min_x = simulator.get_min_x() * 2
max_x = simulator.get_max_x() * 2
min_z = simulator.get_min_z()
max_z = simulator.get_max_z()

plt.figure(figsize=(12, 10))
plt.imshow(
    normalized_image,
    cmap='gray',
    extent=[min_x, max_x, max_z, min_z],
    aspect='auto'
)
plt.title("Liver Anatomy in Water Bath\nCurvilinear Probe - 3.5 MHz", fontsize=14)
plt.xlabel("Width (mm)")
plt.ylabel("Depth (mm)")


plt.colorbar(label="Normalized Intensity")
plt.savefig("Getting_Started_Chapter_5_1_Complex_Scene.png", dpi=150, bbox_inches='tight')
plt.show()
```

### Understanding Mesh vs Sphere Objects

The simulator supports two main types of objects:

| Object Type | Best For | Example Usage |
|-------------|----------|---------------|
| **Spheres** | Simple phantoms, point targets | Basic testing, learning fundamentals |
| **Mesh Objects** | Realistic anatomy | Organ simulation, clinical training |

> ⚠️ **Important**: Currently, mesh objects and sphere objects cannot be used together in the same world. Choose one type based on your simulation needs.

---

## Conclusion

Congratulations! You've completed the ultrasound simulator getting started tutorial. You now understand:

### Advanced Examples

For more complex simulations and advanced usage, see these example files:

**Sphere-based Phantoms**:
- `examples/sphere_sweep.py` - Demonstrates probe positioning and movement with sphere objects

**Mesh-based Phantoms**:
- `examples/liver_sweep.py` - Displays relative positioning and movement over mesh assets
- `examples/server.py` - Interactive web-based ultrasound simulator with multiple organ meshes

These examples show more sophisticated techniques including probe movement, multiple organ systems, and interactive interfaces.

Happy simulating!
