# Isaac for Healthcare - Sensor Simulation

This repository contains high-performance GPU-accelerated sensor simulation tools for healthcare applications, powered by NVIDIA technologies.

## Components

### Ultrasound Raytracing Simulator

![image](https://github.com/user-attachments/assets/5d9ae84b-8718-43e5-806d-146629c17ab7)


A high-performance GPU-accelerated ultrasound simulator using NVIDIA OptiX raytracing with Python bindings. This simulator enables real-time ultrasound simulation for training, research, and development purposes.

Key features:
- GPU acceleration with CUDA and NVIDIA OptiX
- Python interface for ease of use
- Real-time simulation capabilities

[Learn more about the Ultrasound Raytracing Simulator](./ultrasound-raytracing/README.md)

## Requirements

- NVIDIA GPU with CUDA support
- NVIDIA Driver 555+
- CUDA 12.6+
- CMake 3.24+
- Python 3.10+

See individual component READMEs for specific requirements.

## Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/isaac-for-healthcare/i4h-sensor-simulation.git
   cd i4h-sensor-simulation
   ```

2. Follow the setup instructions for the specific simulator you want to use:
   - [Ultrasound Raytracing Simulator](./ultrasound-raytracing/README.md)


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](./LICENSE) file for details.

## Security

Please see our [Security Policy](./SECURITY.md) for information on reporting security vulnerabilities.

## Support

For questions and support, please open an issue in the GitHub repository.
