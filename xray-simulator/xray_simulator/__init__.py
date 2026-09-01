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

"""Xray Simulator - High-Level API for X-ray Image Simulation.

This package provides a simple, object-oriented API for generating simulated
fluoroscopy (X-ray) images from CT volumes. It wraps the underlying Slang-based
GPU rendering pipeline with a clean interface.

Example:
    >>> from xray_simulator import VolumePreprocessor, xray_simulator, SimulatorConfig
    >>>
    >>> # Step 1: Preprocess CT volume
    >>> preprocessor = VolumePreprocessor.from_dicom("/path/to/dicom/")
    >>> volume = preprocessor.preprocess(output_dir="/tmp/fluoro_cache")
    >>>
    >>> # Step 2: Generate fluoroscopy frames
    >>> config = SimulatorConfig()
    >>> simulator = xray_simulator(volume, config)
    >>> frame = simulator.render_frame(rotation=(0, 0, 0), translation=(0, 0, 0))
"""

from .config import (
    DISPLAY_PRESETS,
    CarmGeometry,
    DisplaySettings,
    HuToMuMapping,
    MetricsSettings,
    OutputSettings,
    PreprocessingSettings,
    RealismSettings,
    SimulatorConfig,
    XrayPhysics,
    resolve_display_settings,
)
from .display import apply_display, calibrate_display, transmission
from .geometry import (
    CANONICAL_FRAME,
    VIEWS,
    clinical_angles_to_rotation,
    euler_zxy_to_matrix,
    matrix_to_euler_zxy,
    project_point_to_detector,
    rotation_to_clinical_angles,
    view_frame_warning,
    view_matrix,
    view_rotation,
    volume_center_xyz_mm,
)
from .hu_mapping import hu_to_mu, hu_to_mu_curve
from .preprocessor import VolumePreprocessor
from .simulator import CineSequence, Frame, Pose, SimulatorMetrics, xray_simulator
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
    # Image appearance
    "DisplaySettings",
    "DISPLAY_PRESETS",
    "resolve_display_settings",
    "apply_display",
    "calibrate_display",
    "transmission",
    # HU → μ transfer function
    "hu_to_mu",
    "hu_to_mu_curve",
    # C-arm geometry and clinical views
    "CANONICAL_FRAME",
    "VIEWS",
    "view_matrix",
    "view_rotation",
    "clinical_angles_to_rotation",
    "rotation_to_clinical_angles",
    "euler_zxy_to_matrix",
    "matrix_to_euler_zxy",
    "project_point_to_detector",
    "volume_center_xyz_mm",
    "view_frame_warning",
    # Volume
    "PreprocessedVolume",
    "VolumePreprocessor",
    # Simulator
    "xray_simulator",
    "Pose",
    "Frame",
    "CineSequence",
    "SimulatorMetrics",
    # Volume metadata
    "VolumeMetadata",
]

__version__ = "0.1.0"
