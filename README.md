# i4h-sensor-simulation

## Prerequisites

### NVIDIA OptiX SDK

This project requires the NVIDIA OptiX SDK version 8.1. Follow these steps to set it up:

1. Download OptiX SDK 8.1 from the [NVIDIA Developer website](https://developer.nvidia.com/optix/downloads)
   - You'll need to create an NVIDIA developer account if you don't already have one
   - Make sure to download version 8.1 specifically

2. Extract the downloaded OptiX SDK archive

3. Place the extracted directory inside the `ultrasound/third_party/optix` directory, maintaining the following structure:
   ```
   ultrasound/third_party/
   └── optix
       └── NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64  # Name may vary based on the platform
           ├── include
           │   └── internal
           └── SDK
               ├── cuda
               └── sutil
   ```

4. Do not modify the existing `CMakeLists.txt` and `sampleConfig.h` files that are part of this project.

5. Verify that the OptiX headers are accessible at `ultrasound/third_party/optix/NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64/include/optix.h`

Note: OptiX SDK 8.1 requires an NVIDIA GPU with compatible drivers (R525 or newer).
