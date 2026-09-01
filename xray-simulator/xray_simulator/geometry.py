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

"""C-arm pose geometry: world axes, clinical view presets and detector projection.

World frame
-----------
The renderer works in millimeters. Array axis 2 maps to world X, axis 1 to world Y and
axis 0 to world Z, so for a volume in the canonical patient frame produced by the
digital-twin preprocessing (``anatomical_frame == "LPS"``):

* world **+X** points to the patient's **Left**,
* world **+Y** points to the patient's **Posterior** (back),
* world **+Z** points to the patient's **Superior** (head).

Volumes whose ``anatomical_frame`` is unset carry no such guarantee, and the view presets
below are meaningless for them.

Pose convention
---------------
A pose is Euler angles ``(rx, ry, rz)`` in radians plus a translation in mm, matching the
shader's ZXY convention ``R = Rz * Rx * Ry``. ``R`` maps the C-arm's local frame to world:

* local **+Z** is the beam direction, from source toward detector, so column 2 of ``R`` is
  the patient direction the beam travels in,
* local **+X** is the detector row direction, so column 0 is the patient direction that
  increasing image *column* index moves toward,
* local **+Y** is the detector column direction, so column 1 is the patient direction that
  increasing image *row* index moves toward. Images are stored row 0 first and displayed
  top-down, so column 1 pointing Inferior is what puts the head at the top of the image.

The identity pose ``(0, 0, 0)`` therefore sends the beam along world +Z, which for a
canonical volume is a caudocranial axial projection through the patient's long axis, not a
clinical view. It is kept as-is for backward compatibility with saved poses; use the view
presets or :func:`clinical_angles_to_rotation` to get radiographic views.

Clinical angles
---------------
:func:`clinical_angles_to_rotation` follows the cath-lab convention, anchored on the PA
frontal view (source below the table, detector above a supine patient):

* **primary** angle is LAO/RAO, positive toward LAO, a rotation about the patient's
  Superior axis that swings the detector toward the patient's left,
* **secondary** angle is CRAN/CAUD, positive toward cranial, a rotation about the
  patient's Left axis that tilts the detector toward the head.

Both are extrinsic rotations in the patient frame, composed as
``R = Rz(primary) * Rx(-secondary) * R_pa``, with the primary rotation applied last.
"""

from __future__ import annotations

import numpy as np

CANONICAL_FRAME = "LPS"

# Unit patient directions in the world frame described above.
LEFT = np.array([1.0, 0.0, 0.0])
RIGHT = -LEFT
POSTERIOR = np.array([0.0, 1.0, 0.0])
ANTERIOR = -POSTERIOR
SUPERIOR = np.array([0.0, 0.0, 1.0])
INFERIOR = -SUPERIOR

# Each view is (beam direction, patient direction of increasing image column).
# The row direction is derived, and comes out Inferior for all of them, which is what
# places the head at the top of the image.
_VIEWS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    # Beam enters the front and exits the back; patient left on the viewer's right.
    "ap": (POSTERIOR, LEFT),
    # Beam enters the back and exits the front, as in a cath lab with the tube under the
    # table. Seen from behind, so patient left is on the viewer's left.
    "pa": (ANTERIOR, RIGHT),
    # Named by the side the detector is on, following radiographic convention.
    "lateral_left": (LEFT, ANTERIOR),
    "lateral_right": (RIGHT, POSTERIOR),
}

VIEWS: tuple[str, ...] = tuple(_VIEWS)

# Below this |cos| the ZXY decomposition is at a gimbal lock and only (rz - ry) is
# determined. Every view preset sits exactly there, since they all use rx = -90 degrees.
_GIMBAL_EPS = 1e-7


def euler_zxy_to_matrix(rotation: tuple[float, float, float]) -> np.ndarray:
    """Build the rotation matrix for Euler angles, mirroring the shader.

    Args:
        rotation: Euler angles ``(rx, ry, rz)`` in radians, ZXY convention.

    Returns:
        3x3 rotation matrix mapping the C-arm local frame to world.
    """
    rx, ry, rz = (float(a) for a in rotation)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    return np.array(
        [
            [cz * cy - sz * sx * sy, -sz * cx, cz * sy + sz * sx * cy],
            [sz * cy + cz * sx * sy, cz * cx, sz * sy - cz * sx * cy],
            [-cx * sy, sx, cx * cy],
        ]
    )


def matrix_to_euler_zxy(matrix: np.ndarray) -> tuple[float, float, float]:
    """Recover Euler angles from a rotation matrix.

    At a gimbal lock (``rx = +/-90 degrees``, which is where every view preset sits) only
    the combination of ``ry`` and ``rz`` is determined; ``ry`` is resolved to zero.

    Args:
        matrix: 3x3 rotation matrix.

    Returns:
        Euler angles ``(rx, ry, rz)`` in radians, ZXY convention.
    """
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    sx = float(np.clip(matrix[2, 1], -1.0, 1.0))
    rx = float(np.arcsin(sx))
    cx = float(np.cos(rx))

    if abs(cx) < _GIMBAL_EPS:
        return (rx, 0.0, float(np.arctan2(matrix[1, 0], matrix[0, 0])))

    rz = float(np.arctan2(-matrix[0, 1], matrix[1, 1]))
    ry = float(np.arctan2(-matrix[2, 0], matrix[2, 2]))
    return (rx, ry, rz)


def view_matrix(view: str) -> np.ndarray:
    """Return the rotation matrix for a named radiographic view.

    Args:
        view: One of :data:`VIEWS` (``"ap"``, ``"pa"``, ``"lateral_left"``,
            ``"lateral_right"``).

    Returns:
        3x3 rotation matrix mapping the C-arm local frame to world.

    Raises:
        ValueError: If the view name is not recognized.
    """
    key = view.strip().lower()
    if key not in _VIEWS:
        raise ValueError(f"Unknown view {view!r}; expected one of {VIEWS}")

    beam, image_column = _VIEWS[key]
    image_row = np.cross(beam, image_column)
    return np.stack([image_column, image_row, beam], axis=1)


def view_rotation(view: str) -> tuple[float, float, float]:
    """Return the Euler angles for a named radiographic view.

    Args:
        view: One of :data:`VIEWS`.

    Returns:
        Euler angles ``(rx, ry, rz)`` in radians.
    """
    return matrix_to_euler_zxy(view_matrix(view))


def clinical_angles_to_rotation(
    primary_deg: float = 0.0,
    secondary_deg: float = 0.0,
) -> tuple[float, float, float]:
    """Convert cath-lab C-arm angles to renderer Euler angles.

    Args:
        primary_deg: LAO/RAO angle in degrees, positive toward LAO (detector swinging
            toward the patient's left).
        secondary_deg: CRAN/CAUD angle in degrees, positive toward cranial (detector
            tilting toward the head).

    Returns:
        Euler angles ``(rx, ry, rz)`` in radians for the requested view.
    """
    primary = np.radians(float(primary_deg))
    secondary = np.radians(float(secondary_deg))

    rotation_z = euler_zxy_to_matrix((0.0, 0.0, primary))
    rotation_x = euler_zxy_to_matrix((-secondary, 0.0, 0.0))
    return matrix_to_euler_zxy(rotation_z @ rotation_x @ view_matrix("pa"))


def rotation_to_clinical_angles(rotation: tuple[float, float, float]) -> tuple[float, float]:
    """Report a pose as cath-lab C-arm angles.

    Inverse of :func:`clinical_angles_to_rotation`. Poses that are not reachable by
    primary and secondary angulation from the PA view (a rolled detector, for instance)
    have no exact description this way, and the returned angles are the closest primary
    and secondary pair.

    Args:
        rotation: Euler angles ``(rx, ry, rz)`` in radians.

    Returns:
        Tuple of (primary LAO/RAO degrees, secondary CRAN/CAUD degrees).
    """
    residual = euler_zxy_to_matrix(rotation) @ view_matrix("pa").T
    primary = np.arctan2(residual[1, 0], residual[0, 0])
    secondary = -np.arctan2(residual[2, 1], residual[2, 2])
    return (float(np.degrees(primary)), float(np.degrees(secondary)))


def view_frame_warning(view: str | None, anatomical_frame: str | None) -> str | None:
    """Check that an anatomically labeled view is safe to use on a given volume.

    A view label such as "AP" only means something if the volume axes are known to be in
    the canonical patient frame. Volumes preprocessed before that frame was recorded carry
    no label and may be in any orientation the scanner produced.

    Args:
        view: View label from the pose, or None for a raw pose carrying no anatomical claim.
        anatomical_frame: Frame recorded in the volume metadata, or None if unknown.

    Returns:
        Message describing the mismatch, or None if there is nothing to warn about.
    """
    if view is None or anatomical_frame == CANONICAL_FRAME:
        return None

    if anatomical_frame is None:
        detail = (
            "the volume metadata does not record an anatomical frame, so its axes may be "
            "in any orientation the scanner produced"
        )
    else:
        detail = f"the volume is in the {anatomical_frame!r} frame"

    return (
        f"Rendering the {view!r} view but {detail}. The view label may not match the "
        f"anatomy in the image. Re-run preprocessing so the volume is reoriented to the "
        f"canonical {CANONICAL_FRAME} frame."
    )


def volume_center_xyz_mm(
    shape_zyx: tuple[int, int, int],
    spacing_zyx_mm: tuple[float, float, float],
    origin_xyz_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return the world position the C-arm rotates about.

    The renderer places the isocenter at the center of the volume bounding box, and a
    pose translation displaces it from there. To aim the isocenter at a patient-space
    point ``p``, pass ``translation = p - volume_center_xyz_mm(...)``.

    Args:
        shape_zyx: Volume shape.
        spacing_zyx_mm: Voxel spacing in mm matching the volume axes.
        origin_xyz_mm: World position of voxel ``[0, 0, 0]``.

    Returns:
        Center of the volume bounding box in world mm, ``(x, y, z)``.
    """
    z, y, x = shape_zyx
    sz, sy, sx = spacing_zyx_mm
    extent = np.array([x * sx, y * sy, z * sz])
    return np.asarray(origin_xyz_mm, dtype=np.float64) + 0.5 * extent


def project_point_to_detector(
    point_xyz_mm: tuple[float, float, float] | np.ndarray,
    rotation: tuple[float, float, float],
    translation: tuple[float, float, float],
    source_to_detector_mm: float,
    source_to_isocenter_mm: float,
    detector_width_px: int,
    detector_height_px: int,
    pixel_spacing_mm: float,
    isocenter_xyz_mm: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
) -> tuple[float, float] | None:
    """Project a world point onto the detector, mirroring the shader's cone-beam geometry.

    Useful for labeling where a known structure lands in the image, and for checking pose
    conventions on the CPU without a GPU.

    Args:
        point_xyz_mm: World position to project.
        rotation: Pose Euler angles ``(rx, ry, rz)`` in radians.
        translation: Pose translation in mm.
        source_to_detector_mm: SDD.
        source_to_isocenter_mm: SID.
        detector_width_px: Detector width in pixels.
        detector_height_px: Detector height in pixels.
        pixel_spacing_mm: Detector pixel pitch in mm.
        isocenter_xyz_mm: World position the C-arm rotates about, i.e.
            :func:`volume_center_xyz_mm` for the volume being rendered.

    Returns:
        Continuous ``(column, row)`` pixel coordinates matching ``image[row, column]``, or
        None if the point is at or behind the source plane and cannot be projected.
    """
    rotation_matrix = euler_zxy_to_matrix(rotation)
    center = np.asarray(isocenter_xyz_mm, dtype=np.float64) + np.asarray(translation, dtype=np.float64)
    local = rotation_matrix.T @ (np.asarray(point_xyz_mm, dtype=np.float64) - center)

    depth_from_source = local[2] + source_to_isocenter_mm
    if depth_from_source <= 0.0:
        return None

    scale = source_to_detector_mm / depth_from_source
    column = scale * local[0] / pixel_spacing_mm + 0.5 * detector_width_px - 0.5
    row = scale * local[1] / pixel_spacing_mm + 0.5 * detector_height_px - 0.5
    return (float(column), float(row))
