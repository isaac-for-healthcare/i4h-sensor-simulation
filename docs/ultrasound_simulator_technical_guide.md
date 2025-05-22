# Ultrasound Simulator Technical Guide

This guide explains the technical implementation and physical principles behind the GPU-accelerated raytracing ultrasound simulator.

## 1. Ultrasound Physics and Ray-Based Modeling

### 1.1 Basic Ultrasound Physics

Ultrasound imaging relies on high-frequency sound waves (typically 1-20 MHz) propagating through tissue. The fundamental physical principles include:

- **Wave Propagation**: Ultrasound travels as a mechanical pressure wave through tissue at speeds of approximately 1540 m/s (varying by tissue type)
- **Wavelength**: For medical ultrasound at 5 MHz, the wavelength is approximately 0.3 mm in soft tissue
- **Attenuation**: As ultrasound travels through tissue, its amplitude decreases exponentially with distance due to absorption and scattering
- **Reflection and Refraction**: At interfaces between tissues with different acoustic impedances, waves are partially reflected and refracted
- **Scattering**: Small tissue structure that are smaller than the wavelength of the transmitted pulse scatter ultrasound energy in multiple directions. When scattered signals return to the transducer, they create the characteristic speckle pattern

#### Point Spread Function

The Point Spread Function (PSF) represents how the imaging system responds to a point reflector and defines the theoretical resolution of the ultrasound image. In our ray-based simulator:

- **PSF Implementation**: We model the PSF as a Gaussian kernel modulated by a cosine function with parameters based on:
  - Probe frequency (axial resolution)
  - Aperture size (lateral resolution)
  - Elevational height (slice thickness)

- **Resolution Modeling**: The PSF captures the key resolution limitations of real ultrasound:
  - Axial resolution: Limited by pulse length, related to transducer frequency
  - Lateral resolution: Determined by beam width, related to aperture size and focal depth

### 1.2 From Waves to Rays

Ultrasound wave propagation occurs in two distinct regions: the near field (Fresnel zone), where waves from a transducer source are complex with multiple interference patterns and curved wavefronts, and the far field (Fraunhofer zone), where waves spread in a more predictable pattern with approximately planar wavefronts. The transition between these regions occurs at approximately D²/4λ, where D is the aperture diameter and λ is the wavelength. Ultrasound imaging typically creates images in the near-field, but converts curved near-field wavefronts to coherent, flat far-field plane wave wavefronts via beamforming.

Rays model the normal direction of wavefront propagation over time. For a homogeneous medium, refraction can be ignored and wave propagation can be modeled as straight rays perpendicular to the wavefront.

![rays](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Hamiltonian_Optics-Rays_and_Wavefronts.svg/250px-Hamiltonian_Optics-Rays_and_Wavefronts.svg.png)

While ultrasound imaging works primarily in the near-field, we can model the beamformed scanlines with parallel rays because:

1. **Beamforming Transformation**: Beamforming effectively transforms near-field waves into far-field-like behavior
2. **Primary Propagation Direction**: Each scanline has a well-defined primary direction of energy propagation
3. **Interface Physics**: Reflection and refraction at tissue interfaces follow predictable patterns similar to optical rays

This ray-based approach provides an excellent compromise between physical accuracy and computational efficiency for real-time simulation.

### 1.3 Ray-Tracing Approach

In our simulator, we model ultrasound propagation using deterministic ray tracing:

- **Ray Generation**: Rays are emitted from the transducer elements in patterns matching the probe geometry
- **Ray Propagation**: Rays travel in straight lines until encountering a tissue interface
- **Interface Interaction**: At interfaces, rays can split into reflected and refracted components using acoustic versions of optical laws:
  - Snell's law for refraction: sin(θ₁)/sin(θ₂) = c₁/c₂
  - Reflection coefficient: R = ((Z₂cos(θ₁) - Z₁cos(θ₂))/(Z₂cos(θ₁) + Z₁cos(θ₂)))²
- **Intensity Modeling**: Ray intensity diminishes with distance according to the Beer-Lambert law and tissue attenuation properties
- **Scattering**: Volume scattering is modeled using stochastic textures with tissue-specific properties

### 1.4 Limitations and Solutions

Ray-based models have inherent limitations in capturing certain wave phenomena:

| Wave Phenomenon | Ray Limitation | Solution |
|-----------------|----------------|--------------|
| Diffraction | Not directly modeled by rays | PSF convolution in post-processing |
| Interference | Not captured by individual rays | Not modeled (PSF convolution) |
| Phase information | Rays typically don't carry phase | Not modeled (amplitude-only simulation) |
| Near-field effects | Ray approximation less accurate near transducer | Focus on modeling far-field behavior |

Despite these limitations, ray-based models provide an excellent compromise between physical accuracy and computational efficiency, especially for real-time applications.

## 2. System Overview

The ultrasound simulator uses NVIDIA OptiX ray tracing and CUDA to deliver high-performance real-time ultrasound image generation. The system currently implements deterministic ray tracing techniques as described by Bürger et al. (2013).

### Key Components

- **Core Physics Engine**: Ray tracing implementation using OptiX [`csrc/cuda/optix_trace.cu`]
- **Ultrasound Simulation**: Main simulation pipeline [`csrc/core/raytracing_ultrasound_simulator.cpp`]
- **CUDA Algorithms**: Post-processing pipelines for RF data to B-mode conversion [`csrc/cuda/cuda_algorithms.cu`]
- **Python Interface**: Bindings for ease of use [`raysim/cuda/__init__.py`]

## 3. Simulation Pipeline

### 3.1 Ray Generation

The simulation starts by generating rays from the face of the ultrasound transducer. The x dimension describes the position latterally (azimuthally) along the transducer surface. The y dimensions defines the elevation direction (slice thickness). By definition, transducers image in the positive z direction. The ray generation varies by probe type to model the probe dependent transmit beam sequence:

- **Curvilinear Probe**:
  - **Physical Layout**: Elements are arranged along a convex curved surface with a fixed radius
  - **Implemented Beam Pattern**: Rays originate from points along the curved surface with diverging directions forming a sector image
  - **Implementation**: [`csrc/cuda/optix_trace.cu:275-293`]
  - **Mathematical Model**: Each ray starts at position `(R*sin(θ), y, R*(cos(θ)-1))` with direction normal to the curved surface

- **Linear Array Probe**:
  - **Physical Layout**: Elements are arranged in a straight line with uniform spacing
  - **Implemented Beam Pattern**: Rays originate in parallel from points along the straight line, creating a rectangular image
  - **Implementation**: [`csrc/cuda/optix_trace.cu:296-312`]
  - **Mathematical Model**: Each ray starts at position `(x, y, 0)` along the linear array with direction `(0, 0, 1)`

- **Phased Array Probe**:
  - **Physical Layout**: Elements are arranged in a small, straight line array
  - **Implemented Beam Pattern**: Rays originate from approximately the same point but with different angular directions, creating a sector image
  - **Implementation**: [`csrc/cuda/optix_trace.cu:315-339`]
  - **Mathematical Model**: All rays start at `(0, y, 0)` with directions based on steering angle `(sin(θ), 0, cos(θ))`

#### Elevational Ray Modeling

In addition to the primary lateral scanning plane, our simulator models the elevational (out-of-plane) dimension for all probe types:

- **Configuration Parameters**:
  - `num_el_samples`: Controls the number of rays/planes in the elevational direction (default=1)
  - `elevational_height`: Defines the physical height of the transducer elements in mm
  - These parameters are set when creating the probe object, for example: `rs.CurvilinearProbe(pose, num_el_samples=10, elevational_height=7.0)`

- **OptiX Launch Configuration**:
  - The `optixLaunch` call in `raytracing_ultrasound_simulator.cpp:284` sets the launch dimensions using:
    ```cpp
    optixLaunch(pipeline_.get(),
                stream,
                pipeline_params_.get_device_ptr(stream),
                pipeline_params_.get_size(),
                &shader_binding_table_,
                probe->get_num_elements(),
                probe->get_num_el_samples(),  // <-- elevational dimension
                /*depth=*/1);
    ```

- **Ray Generation Implementation**:
  - For all probe types, elevational sampling is handled in a common code path: [`csrc/cuda/optix_trace.cu:342-346`]
  - The position along the elevational axis is calculated as:
    ```cpp
    const float d_y = (static_cast<float>(idx.y) / static_cast<float>(dim.y)) - 0.5f;
    const float elevation = ray_gen_data->elevational_height * d_y;
    origin.y = elevation;
    ```
  - This distributes rays evenly across the elevational height of the transducer

- **Post-Processing of Elevational Data**:
  - When `num_el_samples > 1`, the resulting data from multiple elevational planes is:
    1. Convolved with an elevational PSF: [`csrc/core/raytracing_ultrasound_simulator.cpp:303-304`] following [1].
    2. Averaged across all elevational planes to produce a 2D image: [`csrc/core/raytracing_ultrasound_simulator.cpp:306-311`]
  - This models the elevation extent of 2D ultrasound transducer

### 3.2 Ray-Object Interaction

When rays intersect with objects in the scene, several physical phenomena are simulated based on acoustic principles:

- **Reflection and Refraction**: Using acoustic impedance differences and Snell's law
  - Implementation: [`csrc/cuda/optix_trace.cu:171-203`]
  - Physics principle: At tissue interfaces, ultrasound waves are partially reflected and refracted
  - Mathematical model:
    - Reflection coefficient: `R = ((Z₂*cos(θ) - Z₁)/(Z₂*cos(θ) + Z₁))²` where Z₁, Z₂ are acoustic impedances
    - Snell's law: `sin(θ₁)/sin(θ₂) = c₁/c₂` where c₁, c₂ are speeds of sound

- **Attenuation**: Frequency dependent attenuation of ultrasound energy using Beer-Lambert law
  - Implementation: [`csrc/cuda/optix_trace.cu:56-65`]
  - Physics principle: Ultrasound energy decreases exponentially with distance traveled through tissue
  - Mathematical model: `I = I₀ * 10^(-αfd/20)` where:
    - α is attenuation coefficient in dB/(cm⋅MHz)
    - f is frequency in MHz
    - d is distance in cm

- **Scattering**: Backscattering from tissues based on material properties
  - Implementation: [`csrc/cuda/optix_trace.cu:40-49`]
  - Physics principle: Sub-wavelength scattering leads to partially coherent signals along the transducer. This creates the characteristic speckle in b-mode images
  - Mathematical model: Scattering intensity proportional to `scatter_value * material->sigma_`

### 3.3 B-mode Image Generation

#### Raw Scanline Data Generation

During ray tracing, the simulator writes reflection and scattering amplitudes directly into scanlines before any post-processing occurs:

1. **Volumetric Scattering** [`csrc/cuda/optix_trace.cu:56-88`]:
   - As rays travel through tissue, each ray samples the medium at discrete steps
   - For each sample point:
     - Retrieves a scattering value from a 3D texture: `get_scattering_value(pos, material)`
     - Scales by material-specific scattering coefficient: `scatter_val.y * material->sigma_`
     - Applies distance-dependent attenuation: `get_intensity_at_distance(distance, material->attenuation_)`
     - Adds contribution to appropriate depth in scanline: `intensities[step] += scattering_value * intensity * attenuation`

2. **Specular Reflections** [`csrc/cuda/optix_trace.cu:434-505`]:
   - When a ray encounters a tissue interface:
     - Calculates reflection coefficient based on acoustic impedance: `R = ((Z₂*cos(θ) - Z₁)/(Z₂*cos(θ) + Z₁))²`
     - Computes specular reflection intensity based on material specularity and geometric factors
     - Writes directly to the corresponding time/depth index in scanline: `scanline[get_intensity_offset(ray.t_ancestors + t)] = 2.f * specular_reflection` where `2.f` is an arbitrary amplification for appearance.

3. **Ray Recursion for Multiple Reflections/Refractions**:
   - Secondary rays (both reflected and refracted) recursively trace through the scene
   - Each recursive ray carries a fraction of the original intensity based on reflection/transmission coefficients
   - Contributions are accumulated in the same scanline buffer
   - Recursion is limited by the maximum recursion deth parameter `params.max_depth` (typically 3-5)

The resulting scanline data represents maximum resolution RF signals that would be recorded by an ideal point transducer. This raw data contains sharp specular reflections at exact interface locations and scattered signals with ideal spatial resolution. These signals are not yet modulated. The PSF convolution in subsequent processing introduces the physical limitations of real ultrasound imaging systems.

#### RF Data to B-mode Processing

After ray-tracing has populated the scanlines with raw reflection data, the data undergoes a series of transformations that mirror the signal processing chain in clinical ultrasound systems:

1. **PSF Convolution**: Simulating Transducer Resolution Limits
   - Implementation: [`csrc/core/raytracing_ultrasound_simulator.cpp:290-315`]
   - Real transducers have finite bandwidth and aperture size, limiting their ability to resolve small structures
   - We model this by convolving raw data with a Gaussian-cosine kernel: the Gaussian component represents the latteral beam width, while the cosine modulation simulates the transmitted pulse waveform
   - This 3D convolution operates in axial, lateral, and elevational dimensions, modeling the limited resolution of real transducers

2. **Time Gain Compensation**: Apmplifying over propagation distance
   - Implementation: [`csrc/core/raytracing_ultrasound_simulator.cpp:318-332`]
   - Deeper tissues naturally return weaker echoes due to attenuation
   - The TGC applies a depth-dependent amplification using a piecewise linear curve
   - This allows for constant structure brightness as is with the TGC controls on clinical machines

3. **Envelope Detection**: Extracting the Signal Amplitude
   - Implementation: [`csrc/core/raytracing_ultrasound_simulator.cpp:335-340`]
   - The [Hilbert transform](https://en.wikipedia.org/wiki/Hilbert_transform) converts oscillating RF signals into [analytic representation](https://en.wikipedia.org/wiki/Analytic_signal)
   - Taking the absolute value of the analytic signal allows for the extraction of the signal envelope
   - This process reveals the reflection strength at each tissue interface

4. **Log Compression**: Managing Wide Dynamic Range
   - Implementation: [`csrc/core/raytracing_ultrasound_simulator.cpp:343-350`]
   - Ultrasound signals span a dynamic range too wide for displays (often 60-100 dB)
   - Logarithmic compression (20*log10) maps this range to displayable levels
   - This enhances subtle tissue details while preventing strong reflectors from overwhelming the image and brings the image into a representation the can be more easily perceived by the human eye.

5. **Scan Conversion**: Creating the 2D Display Image
   - Implementation: [`csrc/core/raytracing_ultrasound_simulator.cpp:353-380`]
   - Transforms data from acquisition geometry (scan lines) to display geometry (pixels)
   - Each probe type requires specific mapping:
     - Curvilinear: Polar to rectangular conversion with increasing scanline separation at depth
     - Linear: Uniform rectangular mapping with parallel scanlines
     - Phased array: Sector to rectangular conversion with scanlines diverging from origin

This processing chain transforms the idealized reflection data into images with the characteristic appearance and artifacts of clinical ultrasound.

## 4. Physics-Based Features

The simulator implements several physics-based features for realistic ultrasound imaging:

### 4.1 Tissue Modeling and Material System

The simulator uses a material system to define the acoustic properties of different tissues and their interaction with ultrasound. This section explains how materials are defined, assigned to objects, and utilized throughout the ray-tracing process.

#### Material Definition and Properties

Materials in the simulator are defined through the `Material` class [`csrc/core/material.cpp`], which encapsulates all acoustic properties relevant to ultrasound simulation:

- **Speed of Sound** (m/s): Defines the wave propagation velocity through the tissue (Note: velocity is not considered in the current model of the travel time of a wave)
  - Affects: Refraction angle, travel time, depth calculation
  - Typical values: Water (1480 m/s), Fat (1450 m/s), Muscle (1580 m/s), Bone (3500 m/s)

- **Acoustic Impedance** (MRayl): The product of density and speed of sound
  - Affects: Reflection coefficient at tissue interfaces
  - Typical values: Water (1.48 MRayl), Fat (1.38 MRayl), Muscle (1.70 MRayl), Bone (7.80 MRayl)

- **Attenuation Coefficient** (dB/cm/MHz): The rate at which ultrasound energy is absorbed
  - Affects: Amplitude decay with depth, shadowing
  - Typical values: Water (0.002 dB/cm/MHz), Fat (0.63 dB/cm/MHz), Muscle (1.3-3.3 dB/cm/MHz)

- **Scattering Properties** (`mu0_`, `sigma_`): Control how tissue scatters ultrasound
  - `mu0_`: Scattering density threshold (0-1)
  - `sigma_`: Scattering coefficient intensity
  - Together these parameters create the characteristic speckle pattern of different tissues

- **Specularity**: Controls the directional nature of reflections (0-1)
  - Higher values create more mirror-like reflections at tissue boundaries
  - Lower values produce more diffuse reflections

#### Material Registry and Management

The simulator includes a `Materials` class that serves as a registry for predefined tissue types. The implementation [`csrc/core/material.cpp:40-55`] initializes a collection of common tissue types (water, blood, fat, liver, muscle, bone) with their respective acoustic properties.

Each material is stored with a name identifier and automatically uploaded to GPU memory for efficient access during ray tracing. The `get_index()` method allows retrieval of material indices by name when setting up simulations.

#### Assigning Materials to Objects

The simulator supports two primary geometry types that can be assigned materials:

1. **Primitive Shapes** (e.g., Spheres): Created programmatically with material indices via the `Sphere` class [`csrc/core/hitable.cpp:36-54`]. The constructor takes a position, radius, and material index as parameters.

2. **Mesh Objects**: Loaded from standard 3D file formats (OBJ, STL, etc.) using the Assimp library [`csrc/core/hitable.cpp:82-140`]. The `Mesh` class constructor takes a file path and material index, handling the loading and preparation of the mesh for simulation.

When objects are added to the world through the `World::add()` method [`csrc/core/world.cpp:47-57`], they are associated with their assigned material index, which is later used during ray tracing to access the appropriate acoustic properties.

#### Integration with OptiX Ray Tracing

During the build process, each object in the scene is converted into OptiX-compatible geometry with material assignments:

1. **Scene Building**: The `World::build()` method [`csrc/core/world.cpp:59-74`] prepares the scene for ray tracing:
   - Geometric data (vertices, indices) is uploaded to GPU memory
   - An acceleration structure is created for efficient ray-object intersection
   - Each object's material index is included in its `HitGroupData`

2. **Material Data Access**: During ray tracing, material properties are accessed directly through a pointer lookup in the ray tracing kernel [`csrc/cuda/optix_trace.cu:386-387`].

3. **Material Property Usage**: Throughout the ray tracing kernel:
   - Impedance and speed of sound determine reflection and refraction behavior [`csrc/cuda/optix_trace.cu:180-193`]
   - Attenuation controls how quickly ray intensity decreases with distance [`csrc/cuda/optix_trace.cu:66-73`]
   - Scattering parameters influence the speckle pattern [`csrc/cuda/optix_trace.cu:43-50`]
   - Specularity affects the appearance of interfaces [`csrc/cuda/optix_trace.cu:229-248`]

This design allows the ray tracer to model complex acoustic interactions between multiple tissue types in a physically accurate way, while maintaining high performance through GPU acceleration.

#### Volumetric Scattering Implementation

To create realistic tissue textures, the simulator uses a dual-channel 3D texture that efficiently models sub-wavelength scattering. This approach creates the complex speckle patterns characteristic of ultrasound images without modeling individual scatterers.

**Texture Generation and Structure** [`csrc/core/world.cpp:30-44`]:
- A 256³ voxel texture with two channels is generated:
  - **Channel 0**: Uniform distribution [0,1] - controls scattering density
  - **Channel 1**: Normal distribution N(0,1) - controls scattering amplitude

**Spatial Efficiency** [`csrc/core/world.cpp:51`]:
- The texture uses `cudaAddressModeWrap` to repeat seamlessly in all dimensions
- A single 256³ texture represents an infinite volume with no boundary artifacts
- Hardware-accelerated trilinear filtering improves performance and quality
- The pattern repeats every 50mm (default resolution), but repetition remains visually undetectable due to random distributions, material variations, and PSF convolution

**Material-Texture Interaction** [`csrc/cuda/optix_trace.cu:43-51`]:
- During ray traversal, positions are converted to texture coordinates: `pos /= resolution_mm`
- The texture is sampled: `scatter_val = tex3D<float2>(params.scattering_texture, pos.x, pos.y, pos.z)`
- Material parameters control how texture values affect scattering:
  - `mu0_` acts as a threshold against Channel 0 (density):
    ```
    if (scatter_val.x <= material->mu0_) { return scatter_val.y * material->sigma_; }
    return 0.f;
    ```
  - `sigma_` scales the intensity from Channel 1 (amplitude)
  - Example: Liver (`mu0_=0.7`, `sigma_=0.3`) scatters at 70% of locations with moderate intensity, while fat (`mu0_=1.0`, `sigma_=1.0`) scatters everywhere with higher intensity

**Integration into Scanlines** [`csrc/cuda/optix_trace.cu:80-101`]:
- Scattering values accumulate into the scanlines during ray traversal:
  ```
  intensities[step] += get_scattering_value(pos, material) * intensity *
                       get_intensity_at_distance(distance, material->attenuation_);
  ```
- This produces spatially consistent scattering where the same world position yields the same base texture values
- Different materials at the same position produce different scattering responses due to their unique parameters
- The scattering values are affected by distance-dependent attenuation

This volumetric scattering approach achieves three key goals:
1. **Tissue-Specific Textures**: Each tissue type has its characteristic speckle pattern
2. **Computational Efficiency**: A small texture simulates an infinite volume
3. **Physical Realism**: Speckle emerges naturally from the simulation physics rather than being artificially added

The tissue modeling system's integration with the OptiX ray-tracing pipeline enables simulation of complex acoustic phenomena while maintaining interactive performance.

### 4.2 Ultrasound-Specific Artifacts

The simulator accurately reproduces key ultrasound artifacts that result from the underlying physics. The following artifacts emerge naturally from the simulation rather than being artificially added:

#### Acoustic Shadowing

![Acoustic Shadowing: Image showing dark region behind a highly attenuating structure (e.g., bone or air pocket)]()

**Physics Principle**: Occurs when strongly attenuating or reflecting structures block the ultrasound beam, preventing it from reaching deeper structures.

**Implementation**:
- Beer-Lambert law calculates intensity attenuation through materials
- Ray-by-ray accumulation of attenuation effects
- Visible in the simulation when imaging through materials with high attenuation coefficients (e.g., bone)

#### Reverberation

![Reverberation: Image showing multiple parallel bright lines at regular intervals below a strong reflector]()

**Physics Principle**: Sound waves bounce back and forth between highly reflective parallel interfaces, creating ghost images at regular intervals.

**Implementation**:
- Multiple reflection paths through recursive ray tracing (`params.max_depth` parameter)
- Particularly visible when imaging perpendicular to two strong reflectors
- Note: Currently partially implemented as reflection recursion is limited (controlled by `reflection_on` flag)

#### Refraction

![Refraction: Image showing distortion of structures beneath an interface with speed of sound mismatch]()

**Physics Principle**: Bending of ultrasound waves at tissue interfaces due to speed of sound differences.

**Implementation**:
- Application of Snell's law at material interfaces [`csrc/cuda/optix_trace.cu:177-193`]
- Ray direction changes according to the ratio of speed of sound between materials
- Creates positional distortions when imaging through tissues with varying speeds of sound

#### Speckle

![Speckle: Image showing characteristic grainy texture in a homogeneous region of tissue]()

**Physics Principle**: Interference patterns caused by constructive and destructive interaction of echoes from sub-resolution scatterers.

**Implementation**:
- Spatially varying 3D texture combined with material-specific scattering properties
- Creates consistent, tissue-specific texture patterns
- Varies realistically with imaging parameters like frequency and aperture size

These artifacts are key components in creating realistic ultrasound images for training and simulation purposes. By reproducing them accurately through physics-based modeling rather than post-processing effects, the simulator provides a more authentic representation of how real ultrasound systems interact with biological tissues.

## 5. Performance Optimization

### 5.1 GPU Acceleration

The simulator leverages GPU parallelism for real-time performance:

- **OptiX Ray Tracing**: Efficient ray-scene intersection
  - Acceleration structure for fast geometric queries
  - Parallel tracing of thousands of rays simultaneously

- **CUDA Kernel Optimizations**: Parallel processing of post-processing steps
  - Separable convolution kernels for PSF application
  - Efficient parallel scan conversion

- **Memory Management**: Efficient texture memory usage for scattering data
  - 3D textures with hardware interpolation for scattering patterns
  - Coalesced memory access patterns for scan line processing

Next, have a look at our [Quick Start Guide](ultrasound_simulator_quickstart.md)
## 7. References

1. Bürger et al. (2013) - "Real-time GPU-based ultrasound simulation using deformable mesh models"
2. Mattausch & Goksel (2016) - "Monte-Carlo Ray-Tracing for Realistic Ultrasound Training Simulation"
3. Law et al. (2016) - "Real-time simulation of B-mode ultrasound images for medical training"
4. Szabo, T. L. (2004) - "Diagnostic Ultrasound Imaging: Inside Out"
5. Prince, J. L., & Links, J. M. (2015) - "Medical Imaging Signals and Systems"
