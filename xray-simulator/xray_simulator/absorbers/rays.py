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

"""The renderer's detector ray grid, rebuilt on the CPU so absorbers can share it.

An absorber composited onto a rendered frame has to integrate along exactly the rays the volume
was marched along, or it lands a fraction of a pixel away from the anatomy it is registered
against. Every ray here starts at the source and ends at a pixel centre, placed the way
``computePixelIntensity`` in ``diffdrr_slang.slang`` places them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import euler_zxy_to_matrix


@dataclass(frozen=True, slots=True)
class DetectorRays:
    """Source and pixel-centre grid of one projection, in world millimetres.

    Pixel ``(column, row)`` sits at ``first_pixel_mm + column * column_step_mm + row *
    row_step_mm``, which indexes ``image[row, column]`` in the rendered frame.

    Attributes:
        source_mm: X-ray source position, ``(3,)``.
        first_pixel_mm: Centre of pixel ``(0, 0)``, ``(3,)``.
        column_step_mm: Displacement from one column to the next, ``(3,)``.
        row_step_mm: Displacement from one row to the next, ``(3,)``.
        width: Detector columns.
        height: Detector rows.
    """

    source_mm: np.ndarray
    first_pixel_mm: np.ndarray
    column_step_mm: np.ndarray
    row_step_mm: np.ndarray
    width: int
    height: int

    def __post_init__(self) -> None:
        for name in ("source_mm", "first_pixel_mm", "column_step_mm", "row_step_mm"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite 3-vector, got {value}")
            object.__setattr__(self, name, value)
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError(f"detector must have positive size, got {self.width}x{self.height}")
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))

    @classmethod
    def from_axes(
        cls,
        detector_axes: np.ndarray,
        isocenter_mm: np.ndarray,
        *,
        source_to_detector_mm: float,
        source_to_isocenter_mm: float,
        width: int,
        height: int,
        pixel_spacing_mm: float,
        pixel_spacing_y_mm: float | None = None,
        detector_offset_xy_mm: tuple[float, float] = (0.0, 0.0),
    ) -> DetectorRays:
        """Build the grid from the detector's axes, for callers that track the C-arm as a frame.

        Args:
            detector_axes: ``(3, 3)`` whose columns are the detector's column direction, its row
                direction and the beam direction from source to detector.
            isocenter_mm: Point the beam axis passes through, ``source_to_isocenter_mm`` from
                the source.
            source_to_detector_mm: SDD.
            source_to_isocenter_mm: SID.
            width: Detector columns.
            height: Detector rows.
            pixel_spacing_mm: Horizontal pixel pitch.
            pixel_spacing_y_mm: Vertical pitch, or None for square pixels.
            detector_offset_xy_mm: Detector centre displacement from the beam axis along its
                column and row directions; see :attr:`~xray_simulator.config.CarmGeometry.detector_offset_xy_mm`.
        """
        axes = np.asarray(detector_axes, dtype=np.float64)
        if axes.shape != (3, 3):
            raise ValueError(f"detector_axes must be (3, 3), got {axes.shape}")
        column_axis, row_axis, beam = axes[:, 0], axes[:, 1], axes[:, 2]
        isocenter = np.asarray(isocenter_mm, dtype=np.float64)
        dx = float(pixel_spacing_mm)
        dy = dx if pixel_spacing_y_mm is None else float(pixel_spacing_y_mm)
        offset_x, offset_y = (float(value) for value in detector_offset_xy_mm)
        detector = isocenter + (float(source_to_detector_mm) - float(source_to_isocenter_mm)) * beam
        first_pixel = (
            detector
            + column_axis * ((0.5 - 0.5 * width) * dx + offset_x)
            + row_axis * ((0.5 - 0.5 * height) * dy + offset_y)
        )
        return cls(
            source_mm=isocenter - float(source_to_isocenter_mm) * beam,
            first_pixel_mm=first_pixel,
            column_step_mm=column_axis * dx,
            row_step_mm=row_axis * dy,
            width=width,
            height=height,
        )

    @classmethod
    def for_pose(
        cls,
        rotation: tuple[float, float, float],
        translation: tuple[float, float, float],
        isocenter_xyz_mm: tuple[float, float, float] | np.ndarray,
        *,
        source_to_detector_mm: float,
        source_to_isocenter_mm: float,
        width: int,
        height: int,
        pixel_spacing_mm: float,
        pixel_spacing_y_mm: float | None = None,
        detector_offset_xy_mm: tuple[float, float] = (0.0, 0.0),
    ) -> DetectorRays:
        """Build the grid for a renderer pose, in the same terms the renderer takes it.

        Args:
            rotation: Pose Euler angles ``(rx, ry, rz)`` in radians, ZXY as in the shader.
            translation: Pose translation in mm.
            isocenter_xyz_mm: World point the C-arm rotates about, i.e. the rendered volume's
                :func:`~xray_simulator.geometry.volume_center_xyz_mm`.
            source_to_detector_mm: SDD.
            source_to_isocenter_mm: SID.
            width: Detector columns.
            height: Detector rows.
            pixel_spacing_mm: Horizontal pixel pitch.
            pixel_spacing_y_mm: Vertical pitch, or None for square pixels.
            detector_offset_xy_mm: As in :meth:`from_axes`.
        """
        center = np.asarray(isocenter_xyz_mm, dtype=np.float64) + np.asarray(translation, dtype=np.float64)
        return cls.from_axes(
            euler_zxy_to_matrix(rotation),
            center,
            source_to_detector_mm=source_to_detector_mm,
            source_to_isocenter_mm=source_to_isocenter_mm,
            width=width,
            height=height,
            pixel_spacing_mm=pixel_spacing_mm,
            pixel_spacing_y_mm=pixel_spacing_y_mm,
            detector_offset_xy_mm=detector_offset_xy_mm,
        )

    def transformed(self, linear: np.ndarray, offset: np.ndarray) -> DetectorRays:
        """The same rays expressed in another frame, ``x' = linear @ x + offset``.

        For an absorber whose geometry is held in a frame other than the renderer's world, such
        as a volume's own millimetre grid. Pixel steps are directions and take only the linear part.
        """
        matrix = np.asarray(linear, dtype=np.float64)
        shift = np.asarray(offset, dtype=np.float64)
        return DetectorRays(
            source_mm=matrix @ self.source_mm + shift,
            first_pixel_mm=matrix @ self.first_pixel_mm + shift,
            column_step_mm=matrix @ self.column_step_mm,
            row_step_mm=matrix @ self.row_step_mm,
            width=self.width,
            height=self.height,
        )

    def pixel_centers_mm(self) -> np.ndarray:
        """Every pixel centre, ``(height, width, 3)``."""
        rows = np.arange(self.height, dtype=np.float64)[:, None, None]
        columns = np.arange(self.width, dtype=np.float64)[None, :, None]
        return self.first_pixel_mm + columns * self.column_step_mm + rows * self.row_step_mm


__all__ = ["DetectorRays"]
