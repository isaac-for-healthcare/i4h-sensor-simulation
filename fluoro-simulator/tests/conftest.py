# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytest configuration and shared fixtures for fluorosim tests."""

import numpy as np
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "slow: mark test as slow running")


@pytest.fixture
def sample_hu_volume():
    """Create a small synthetic HU volume for testing.

    Returns a (Z, Y, X) shaped volume with:
    - Background at -900 HU (air)
    - A bone-like cube at +800 HU in the center
    """
    hu = np.full((16, 32, 32), -900.0, dtype=np.float32)
    # Add a "bone" cube in the center
    hu[4:12, 8:24, 8:24] = 800.0
    return hu


@pytest.fixture
def sample_spacing():
    """Return typical CT spacing (Z, Y, X) in mm."""
    return (2.5, 0.5, 0.5)


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary directory for cache testing."""
    cache_dir = tmp_path / "fluorosim_cache"
    cache_dir.mkdir()
    return cache_dir
