# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Fluoroscopy Simulator - High-Level API for X-ray Image Simulation.

This package provides a simple, object-oriented API for generating simulated
fluoroscopy (X-ray) images from CT volumes. It wraps the underlying Slang-based
GPU rendering pipeline with a clean interface.

Example:
    >>> from fluorosim import VolumePreprocessor, FluoroSimulator, SimulatorConfig
    >>>
    >>> # Step 1: Preprocess CT volume
    >>> preprocessor = VolumePreprocessor.from_dicom("/path/to/dicom/")
    >>> volume = preprocessor.preprocess(output_dir="/tmp/fluoro_cache")
    >>>
    >>> # Step 2: Generate fluoroscopy frames
    >>> config = SimulatorConfig()
    >>> simulator = FluoroSimulator(volume, config)
    >>> frame = simulator.render_frame(rotation=(0, 0, 0), translation=(0, 0, 0))
"""

from .config import (
    CarmGeometry,
    HuToMuMapping,
    MetricsSettings,
    OutputSettings,
    PreprocessingSettings,
    RealismSettings,
    SimulatorConfig,
    XrayPhysics,
)
from .preprocessor import VolumePreprocessor
from .simulator import CineSequence, FluoroSimulator, Frame, Pose, SimulatorMetrics
from .volume import PreprocessedVolume, VolumeMetadata

__all__ = [
    # Configuration
    "SimulatorConfig",
    "CarmGeometry",
    "XrayPhysics",
    "RealismSettings",
    "OutputSettings",
    "MetricsSettings",
    "PreprocessingSettings",
    "HuToMuMapping",
    # Volume
    "PreprocessedVolume",
    "VolumePreprocessor",
    # Simulator
    "FluoroSimulator",
    "Pose",
    "Frame",
    "CineSequence",
    "SimulatorMetrics",
    # Volume metadata
    "VolumeMetadata",
]

__version__ = "0.1.0"
