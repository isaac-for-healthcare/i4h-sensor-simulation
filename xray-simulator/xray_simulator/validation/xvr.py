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

"""Geometry adapter for the pinned xvr-data DeepFluoro/Ljubljana release.

The release uses DiffDRR's centered RAS volume frame, AP reorientation and
``reverse_x_axis=False``. Its pose maps the AP camera frame into centered RAS.
This module implements that convention without a runtime DiffDRR dependency.
See docs/validation-datasets.md for the pinned upstream sources and equations.

No image registration, attenuation calibration, or intensity normalization is
performed here. The optional loaders require nibabel, torch and pydicom.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config import CarmGeometry
from ..geometry import matrix_to_euler_zxy, project_point_to_detector
from ..simulator import Pose

XVR_DATA_REVISION = "a17273e3eadbd793bd861f3598a80ce4590c1124"
XVR_CODE_REVISION = "caa55cc8096294cf70a218126bf16008dee0dec7"
EXCLUDED_VIEWS = frozenset({("subject01", "003"), ("subject01", "050"), ("subject04", "002"), ("subject04", "004")})
_AP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
_IMAGE_AXES = np.diag([1.0, -1.0, 1.0])


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    a = np.array(value, dtype=np.float64, copy=True)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    return a


def _size(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(value) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class XvrVolumeFrame:
    """CT grid and its orthogonal transform to centered RAS, without resampling.

    The renderer accepts axis-aligned grids and proper Euler rotations. The xvr
    detector's column/row/beam basis is left-handed. We therefore use a
    left-handed grid-to-RAS transform too: reverse voxel axis Y if needed.
    Both reflections cancel in the camera rotation. Rendered rows/columns then
    already match the reference, without an image flip or spatial registration.

    The resulting renderer frame is deliberately NOT labeled LPS. Its origin is
    the volume *box corner*, half a voxel before the first sample center.
    """

    shape_xyz: tuple[int, int, int]
    affine_ras: np.ndarray
    spacing_xyz_mm: tuple[float, float, float]
    center_ras_mm: np.ndarray
    simulator_to_centered_ras: np.ndarray
    flip_y: bool

    @classmethod
    def from_affine(cls, shape_xyz, affine_ras) -> XvrVolumeFrame:
        if len(shape_xyz) != 3:
            raise ValueError("Expected a 3D volume")
        shape = tuple(_size(v, "volume dimension") for v in shape_xyz)
        affine = _array(affine_ras, (4, 4), "NIfTI affine")
        if not np.allclose(affine[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
            raise ValueError("Invalid affine bottom row")
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        if np.any(spacing <= 0):
            raise ValueError("Voxel spacing must be positive")
        direction = affine[:3, :3] / spacing
        if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5, rtol=0):
            raise ValueError("Sheared volume affine needs explicit resampling")
        center = affine[:3, :3] @ ((np.array(shape) - 1) / 2) + affine[:3, 3]
        flip_y = bool(np.linalg.det(direction) > 0)
        if flip_y:
            direction[:, 1] *= -1
        for a in (affine, center, direction):
            a.setflags(write=False)
        return cls(shape, affine, tuple(spacing), center, direction, flip_y)

    @property
    def spacing_zyx_mm(self) -> tuple[float, float, float]:
        return self.spacing_xyz_mm[::-1]

    @property
    def origin_xyz_mm(self) -> tuple[float, float, float]:
        return tuple(-0.5 * np.array(self.shape_xyz) * self.spacing_xyz_mm)

    def prepare_array(self, values_xyz: np.ndarray) -> np.ndarray:
        """Reindex NIfTI XYZ values into the renderer ZYX grid; preserve values."""
        values = np.asarray(values_xyz)
        if values.shape != self.shape_xyz:
            raise ValueError("Array shape does not match the volume frame")
        if self.flip_y:
            values = np.flip(values, axis=1)
        return np.ascontiguousarray(values.transpose(2, 1, 0))

    def points_to_simulator(self, points_ras, *, centered: bool = True) -> np.ndarray:
        """Convert centered dataset fiducials, or absolute NIfTI RAS coordinates."""
        points = np.asarray(points_ras, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("Points must be finite with final dimension 3")
        if not centered:
            points = points - self.center_ras_mm
        return points @ self.simulator_to_centered_ras

    def to_dict(self) -> dict:
        return {
            "shape_xyz": list(self.shape_xyz),
            "affine_ras": self.affine_ras.tolist(),
            "spacing_xyz_mm": list(self.spacing_xyz_mm),
            "center_ras_mm": self.center_ras_mm.tolist(),
            "simulator_to_centered_ras": self.simulator_to_centered_ras.tolist(),
            "flip_voxel_y": self.flip_y,
            "origin_xyz_mm": list(self.origin_xyz_mm),
            "anatomical_frame": None,
            "warp_applied": False,
        }


@dataclass(frozen=True)
class XvrCamera:
    """A calibrated view consumable directly by SimulatorConfig and render_frame."""

    geometry: CarmGeometry
    pose: Pose
    volume_frame: XvrVolumeFrame
    binning: int

    def project(self, centered_ras_points) -> np.ndarray:
        """Project dataset fiducials to zero-based (column, row) pixel centers."""
        points = self.volume_frame.points_to_simulator(centered_ras_points)
        if points.ndim != 2:
            raise ValueError("Expected points with shape (N, 3)")
        pixels = []
        for point in points:
            pixel = project_point_to_detector(point, self.pose.rotation, self.pose.translation, **asdict(self.geometry))
            pixels.append((np.nan, np.nan) if pixel is None else pixel)
        return np.asarray(pixels).reshape(-1, 2)

    def to_dict(self) -> dict:
        return {
            "geometry": asdict(self.geometry),
            "pose": self.pose.to_dict(),
            "binning": self.binning,
            "volume_frame": self.volume_frame.to_dict(),
            "convention": "xvr centered RAS; AP; reverse_x_axis=False",
            "sid_note": "SDD/2 is a pose parameterization choice, not measured patient distance",
            "supported_data_revision": XVR_DATA_REVISION,
            "reference_code_revision": XVR_CODE_REVISION,
        }


def adapt_xvr_camera(pose, intrinsics: Mapping, volume_frame: XvrVolumeFrame, *, binning: int = 1) -> XvrCamera:
    """Convert an xvr pose/intrinsics companion to simulator geometry and pose.

    ``pose`` must be the stored 4x4 matrix, not Euler angles or the original
    DeepFluoro HDF5 extrinsics. Intrinsic x0/y0 are the *stored* detector offsets
    in mm (DiffDRR's property getters use the opposite sign).
    Binning preserves physical FOV and the center of each b-by-b reference block.
    """
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape == (1, 4, 4):
        pose = pose[0]
    pose = _array(pose, (4, 4), "pose")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-7, rtol=0):
        raise ValueError("Invalid pose bottom row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0) or not np.isclose(
        np.linalg.det(rotation), 1, atol=1e-5, rtol=0
    ):
        raise ValueError("Pose must contain a proper rigid rotation")
    required = {"sdd", "delx", "dely", "x0", "y0", "height", "width"}
    if not required.issubset(intrinsics):
        raise ValueError(f"Missing intrinsics: {sorted(required - intrinsics.keys())}")
    values = {k: float(intrinsics[k]) for k in required}
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("Intrinsics must be finite")
    if any(values[k] <= 0 for k in ("sdd", "delx", "dely")):
        raise ValueError("SDD and pixel pitches must be positive")
    height, width = (_size(intrinsics[k], k) for k in ("height", "width"))
    binning = _size(binning, "binning")
    if height % binning or width % binning:
        raise ValueError("Binning must divide both detector dimensions exactly")
    # Remove float32 round-off in the stored rigid rotation, not genuine scale/shear.
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    world_to_sim = volume_frame.simulator_to_centered_ras.T
    camera_rotation = world_to_sim @ rotation @ _AP @ _IMAGE_AXES
    source = world_to_sim @ pose[:3, 3]
    sid = values["sdd"] / 2
    adapted_pose = Pose(
        rotation=matrix_to_euler_zxy(camera_rotation), translation=tuple(source + sid * camera_rotation[:, 2])
    )
    geometry = CarmGeometry(
        source_to_detector_mm=values["sdd"],
        source_to_isocenter_mm=sid,
        detector_width_px=width // binning,
        detector_height_px=height // binning,
        pixel_spacing_mm=values["delx"] * binning,
        pixel_spacing_y_mm=values["dely"] * binning,
        detector_offset_xy_mm=(values["x0"], -values["y0"]),
    )
    return XvrCamera(geometry, adapted_pose, volume_frame, binning)


def load_xvr_volume(filename: str | Path) -> tuple[np.ndarray, XvrVolumeFrame]:
    """Load NIfTI values into the adapter frame, without changing their signal domain.

    Values are HU for DeepFluoro CTs, but NOT automatically HU for Ljubljana
    angiography. The caller must explicitly choose the appropriate signal model.
    Orthogonal oblique affines are supported without interpolating the volume.
    """
    import nibabel as nib

    image = nib.load(filename)
    units = image.header.get_xyzt_units()[0]
    if units not in ("mm", "unknown"):
        raise ValueError("Expected NIfTI spatial units in millimeters")
    frame = XvrVolumeFrame.from_affine(image.shape, image.affine)
    values = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Volume contains nonfinite values")
    return frame.prepare_array(values), frame


@dataclass(frozen=True)
class XvrView:
    """Paths and calibration for one view in the downloaded xvr layout."""

    dataset: str
    subject: str
    view: str
    reference_path: Path
    calibration_path: Path
    pose: np.ndarray
    intrinsics: Mapping

    def camera(self, frame: XvrVolumeFrame, *, binning: int = 1) -> XvrCamera:
        return adapt_xvr_camera(self.pose, self.intrinsics, frame, binning=binning)

    def prepare_reference(self, raw_pixels: np.ndarray, *, binning: int = 1) -> np.ndarray:
        """Crop the release's collimator border and block-average; no tone fitting."""
        h, w = (int(self.intrinsics[k]) for k in ("height", "width"))
        border = 50 if self.dataset == "deepfluoro" else 0
        pixels = np.asarray(raw_pixels)
        if pixels.shape != (h + 2 * border, w + 2 * border):
            raise ValueError("Reference shape disagrees with the declared dataset crop/intrinsics")
        if border:
            pixels = pixels[border:-border, border:-border]
        binning = _size(binning, "binning")
        if h % binning or w % binning:
            raise ValueError("Binning must divide both reference dimensions exactly")
        pixels = np.asarray(pixels, dtype=np.float32)
        if not np.isfinite(pixels).all():
            raise ValueError("Reference contains nonfinite values")
        return pixels.reshape(h // binning, binning, w // binning, binning).mean(axis=(1, 3))

    def load_reference(self, *, binning: int = 1) -> np.ndarray:
        """Read single-frame MONOCHROME2 stored DICOM values; reject unknown processing."""
        import pydicom

        ds = pydicom.dcmread(self.reference_path)
        if ds.get("PhotometricInterpretation") != "MONOCHROME2":
            raise ValueError("Expected MONOCHROME2; explicitly prepare other polarities")
        if (
            int(ds.get("NumberOfFrames", 1)) != 1
            or float(ds.get("RescaleSlope", 1)) != 1
            or float(ds.get("RescaleIntercept", 0)) != 0
            or "ModalityLUTSequence" in ds
        ):
            raise ValueError("Unexpected DICOM signal transform; explicit preparation required")
        return self.prepare_reference(ds.pixel_array, binning=binning)


def load_xvr_view(
    subject_dir: str | Path, view: str, *, dataset: str = "deepfluoro", include_excluded: bool = False
) -> XvrView:
    """Load a trusted local xvr companion safely; known questionable poses fail closed."""
    import torch

    subject_dir = Path(subject_dir)
    if dataset not in ("deepfluoro", "ljubljana"):
        raise ValueError("dataset must be deepfluoro or ljubljana")
    if not view or Path(view).name != view or view in (".", ".."):
        raise ValueError("view must be a file stem")
    if dataset == "deepfluoro" and (subject_dir.name, view) in EXCLUDED_VIEWS and not include_excluded:
        raise ValueError("Upstream flags this view's ground-truth pose; excluded by default")
    path = subject_dir / "xrays" / f"{view}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    pose = payload["pose"].detach().cpu().numpy().copy()
    pose.setflags(write=False)
    return XvrView(dataset, subject_dir.name, view, path.with_suffix(".dcm"), path, pose, dict(payload["intrinsics"]))
