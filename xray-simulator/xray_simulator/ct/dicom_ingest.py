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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CtVolume:
    """A CT volume in a consistent in-memory convention.

    - `hu_zyx`: CT in Hounsfield Units, shape (Z,Y,X)
    - `spacing_zyx_mm`: voxel spacing in millimeters (Z,Y,X)
    - `origin_xyz_mm`: physical origin in millimeters (X,Y,Z)
    - `direction`: 3x3 direction cosine matrix (row-major)
    """

    hu_zyx: np.ndarray
    spacing_zyx_mm: tuple[float, float, float] | None = None
    origin_xyz_mm: tuple[float, float, float] | None = None
    direction: tuple[float, ...] | None = None

    def to_json_dict(self) -> dict:
        d: dict = {
            "shape_zyx": list(self.hu_zyx.shape),
            "dtype": str(self.hu_zyx.dtype),
        }
        if self.spacing_zyx_mm is not None:
            d["spacing_zyx_mm"] = list(self.spacing_zyx_mm)
        if self.origin_xyz_mm is not None:
            d["origin_xyz_mm"] = list(self.origin_xyz_mm)
        if self.direction is not None:
            d["direction_row_major_3x3"] = list(self.direction)
        return d


def _ct(scan):
    affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ scan.ijk_to_ras_m
    affine[:3] *= 1000
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    return CtVolume(
        scan.values_kji, tuple(spacing[::-1]), tuple(affine[:3, 3]), tuple((affine[:3, :3] / spacing).ravel())
    )


def load_dicom_series_hu(dicom_dir: str | Path) -> CtVolume:
    """Compatibility loader; prefer VolumePreprocessor.from_dicom for full geometry."""
    from ..scan_volume import from_dicom

    return _ct(from_dicom(dicom_dir))


def load_nifti_hu(nifti_path: str | Path) -> CtVolume:
    from ..scan_volume import from_nifti

    return _ct(from_nifti(nifti_path))
