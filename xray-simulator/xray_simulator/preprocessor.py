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

"""Volume preprocessing for the Xray Simulator.

This module provides the VolumePreprocessor class for loading and preprocessing
CT volumes from DICOM or NIfTI sources into a format suitable for rendering.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import HuToMuMapping, PreprocessingSettings
from .hu_mapping import hu_to_mu
from .volume import PreprocessedVolume, VolumeMetadata


class VolumePreprocessor:
    """Preprocessor for converting CT volumes to μ volumes for rendering.

    This class handles the complete preprocessing pipeline:
    1. Load CT from DICOM or NIfTI
    2. Normalize HU values
    3. Convert HU → μ (linear attenuation coefficients)
    4. Optionally save to disk for caching

    Example:
        >>> # From DICOM
        >>> preprocessor = VolumePreprocessor.from_dicom("/path/to/dicom/")
        >>> volume = preprocessor.preprocess(output_dir="/tmp/cache")
        >>>
        >>> # From NIfTI
        >>> preprocessor = VolumePreprocessor.from_nifti("/path/to/ct.nii.gz")
        >>> volume = preprocessor.preprocess()
    """

    def __init__(
        self,
        hu_volume: np.ndarray,
        spacing_zyx_mm: tuple[float, float, float],
        origin_xyz_mm: tuple[float, float, float] | None = None,
        source: str | None = None,
        settings: PreprocessingSettings | None = None,
        anatomical_frame: str | None = None,
    ):
        """Initialize preprocessor with a loaded HU volume.

        Args:
            hu_volume: 3D numpy array of HU values (Z, Y, X).
            spacing_zyx_mm: Voxel spacing in mm (Z, Y, X).
            origin_xyz_mm: Volume origin in mm (X, Y, Z).
            source: Source path for metadata.
            settings: Preprocessing settings.
            anatomical_frame: Patient frame the volume axes are already in, recorded in the
                metadata so the renderer can trust the anatomical view presets. This
                preprocessor reads volumes as they come off disk and never reorients them,
                so pass "LPS" only for data already in the canonical frame; leave it None
                and use the digital-twin preprocessing to have arbitrary acquisitions
                reoriented for you.
        """
        if hu_volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {hu_volume.shape}")

        self._hu_volume = hu_volume.astype(np.float32, copy=False)
        self._spacing_zyx_mm = spacing_zyx_mm
        self._origin_xyz_mm = origin_xyz_mm
        self._source = source
        self._settings = settings or PreprocessingSettings()
        self._anatomical_frame = anatomical_frame

    @classmethod
    def from_dicom(
        cls,
        dicom_dir: str | Path,
        settings: PreprocessingSettings | None = None,
        anatomical_frame: str | None = None,
    ) -> "VolumePreprocessor":
        """Create a preprocessor from a DICOM series directory.

        Args:
            dicom_dir: Path to directory containing DICOM files.
            settings: Preprocessing settings.
            anatomical_frame: Patient frame the series is already in, if known. The series
                is not reoriented; see the constructor.

        Returns:
            VolumePreprocessor instance ready for preprocessing.

        Raises:
            FileNotFoundError: If directory doesn't exist.
            RuntimeError: If SimpleITK is not installed or no DICOM found.
        """
        # Import the existing DICOM loader
        from xray_simulator.ct.dicom_ingest import load_dicom_series_hu

        dicom_dir = Path(dicom_dir)
        if not dicom_dir.exists():
            raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

        ct = load_dicom_series_hu(dicom_dir)

        return cls(
            hu_volume=ct.hu_zyx,
            spacing_zyx_mm=ct.spacing_zyx_mm or (1.0, 1.0, 1.0),
            origin_xyz_mm=ct.origin_xyz_mm,
            source=str(dicom_dir),
            settings=settings,
            anatomical_frame=anatomical_frame,
        )

    @classmethod
    def from_nifti(
        cls,
        nifti_path: str | Path,
        settings: PreprocessingSettings | None = None,
        anatomical_frame: str | None = None,
    ) -> "VolumePreprocessor":
        """Create a preprocessor from a NIfTI file.

        Args:
            nifti_path: Path to NIfTI file (.nii or .nii.gz).
            settings: Preprocessing settings.
            anatomical_frame: Patient frame the file is already in, if known. The volume is
                not reoriented; see the constructor.

        Returns:
            VolumePreprocessor instance ready for preprocessing.

        Raises:
            FileNotFoundError: If file doesn't exist.
            RuntimeError: If nibabel is not installed.
        """
        # Import the existing NIfTI loader
        from xray_simulator.ct.dicom_ingest import load_nifti_hu

        nifti_path = Path(nifti_path)
        if not nifti_path.exists():
            raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")

        ct = load_nifti_hu(nifti_path)

        return cls(
            hu_volume=ct.hu_zyx,
            spacing_zyx_mm=ct.spacing_zyx_mm or (1.0, 1.0, 1.0),
            origin_xyz_mm=ct.origin_xyz_mm,
            source=str(nifti_path),
            settings=settings,
            anatomical_frame=anatomical_frame,
        )

    @classmethod
    def from_numpy(
        cls,
        hu_volume: np.ndarray,
        spacing_zyx_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
        settings: PreprocessingSettings | None = None,
        anatomical_frame: str | None = None,
    ) -> "VolumePreprocessor":
        """Create a preprocessor from a numpy array.

        Args:
            hu_volume: 3D numpy array of HU values (Z, Y, X).
            spacing_zyx_mm: Voxel spacing in mm.
            settings: Preprocessing settings.
            anatomical_frame: Patient frame the array axes are in, if known.

        Returns:
            VolumePreprocessor instance.
        """
        return cls(
            hu_volume=hu_volume,
            spacing_zyx_mm=spacing_zyx_mm,
            origin_xyz_mm=None,
            source="numpy",
            settings=settings,
            anatomical_frame=anatomical_frame,
        )

    def with_hu_to_mu(self, mapping: HuToMuMapping) -> "VolumePreprocessor":
        """Return a preprocessor using a different HU → μ transfer function.

        The loaded HU volume is shared rather than copied, so this is the cheap way to
        sweep window/level settings and compare the resulting images without re-reading
        the CT from disk.

        Args:
            mapping: Replacement HU → μ mapping.

        Returns:
            New preprocessor over the same HU volume.

        Example:
            >>> base = VolumePreprocessor.from_nifti("ct.nii.gz")  # doctest: +SKIP
            >>> soft_tissue = base.with_hu_to_mu(
            ...     HuToMuMapping.from_window_level(window_center=100.0, window_width=800.0)
            ... ).preprocess()  # doctest: +SKIP
        """
        return VolumePreprocessor(
            hu_volume=self._hu_volume,
            spacing_zyx_mm=self._spacing_zyx_mm,
            origin_xyz_mm=self._origin_xyz_mm,
            source=self._source,
            settings=replace(self._settings, hu_to_mu=mapping),
            anatomical_frame=self._anatomical_frame,
        )

    @property
    def settings(self) -> PreprocessingSettings:
        """Return the preprocessing settings, including the HU → μ mapping."""
        return self._settings

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return volume shape (Z, Y, X)."""
        return self._hu_volume.shape

    @property
    def spacing_zyx_mm(self) -> tuple[float, float, float]:
        """Return voxel spacing in mm (Z, Y, X)."""
        return self._spacing_zyx_mm

    @property
    def hu_range(self) -> tuple[float, float]:
        """Return HU value range [min, max]."""
        return (float(self._hu_volume.min()), float(self._hu_volume.max()))

    def preprocess(
        self,
        output_dir: str | Path | None = None,
    ) -> PreprocessedVolume:
        """Run the preprocessing pipeline.

        This performs:
        1. HU clipping (if enabled)
        2. HU → μ conversion
        3. Optional save to disk

        Args:
            output_dir: If provided, save the preprocessed volume here.

        Returns:
            PreprocessedVolume ready for rendering.
        """
        settings = self._settings
        hu = self._hu_volume.copy()

        # Store original HU range
        hu_range = (float(hu.min()), float(hu.max()))

        # Step 1: Clip HU values
        if settings.clip_hu:
            hu = np.clip(hu, settings.hu_clip_min, settings.hu_clip_max)

        # Step 2: Convert HU → μ
        mu = self._hu_to_mu(hu, settings.hu_to_mu)
        mu_range = (float(mu.min()), float(mu.max()))

        # Create metadata
        metadata = VolumeMetadata(
            shape_zyx=mu.shape,
            spacing_zyx_mm=self._spacing_zyx_mm,
            origin_xyz_mm=self._origin_xyz_mm,
            hu_range=hu_range,
            mu_range=mu_range,
            source=self._source,
            hu_to_mu=settings.hu_to_mu.to_dict(),
            anatomical_frame=self._anatomical_frame,
        )

        # Create volume
        volume = PreprocessedVolume(mu, metadata)

        # Optionally save
        if output_dir is not None:
            volume.save(output_dir)
            print(f"[VolumePreprocessor] Saved to: {output_dir}")

        return volume

    def _hu_to_mu(self, hu: np.ndarray, cfg: HuToMuMapping) -> np.ndarray:
        """Convert Hounsfield Units to linear attenuation coefficient (μ).

        Applies the piecewise-linear transfer function defined by ``cfg``, clamped
        outside its outermost control points.

        Args:
            hu: HU volume array.
            cfg: Mapping configuration.

        Returns:
            μ volume as float32 array.
        """
        return hu_to_mu(hu, cfg)

    def __repr__(self) -> str:
        z, y, x = self.shape
        hu_min, hu_max = self.hu_range
        return (
            f"VolumePreprocessor(\n"
            f"  source={self._source!r},\n"
            f"  shape=({z}, {y}, {x}),\n"
            f"  spacing_mm={self._spacing_zyx_mm},\n"
            f"  hu_range=[{hu_min:.0f}, {hu_max:.0f}]\n"
            f")"
        )
