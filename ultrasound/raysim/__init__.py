"""
Ray-Based Ultrasound Simulator

A GPU-accelerated ultrasound simulation package that uses ray-tracing for realistic acoustic behavior.
"""

# Import directly from the root module
from .ray_sim_python import Materials, Pose, RaytracingUltrasoundSimulator, SimParams, UltrasoundProbe, World

__version__ = "0.1.0"

__all__ = [
    "Materials",
    "Pose",
    "RaytracingUltrasoundSimulator",
    "SimParams",
    "UltrasoundProbe",
    "World"
]
