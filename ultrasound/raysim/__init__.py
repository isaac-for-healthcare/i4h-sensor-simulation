"""
Ray-Based Ultrasound Simulator

A GPU-accelerated ultrasound simulation package that uses ray-tracing for realistic acoustic behavior.
"""

# Import directly from the root module
from .ray_sim_python import (
    World,
    Materials,
    UltrasoundProbe as Probe,
    RaytracingUltrasoundSimulator as Simulation,
    SimParams as SimulationConfig,
    Pose
)

__version__ = "0.1.0"
