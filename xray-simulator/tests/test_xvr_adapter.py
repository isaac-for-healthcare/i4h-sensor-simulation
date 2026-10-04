# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Independent camera equations and voxel-index checks; no patient data needed."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from xray_simulator.config import CarmGeometry
from xray_simulator.geometry import euler_zxy_to_matrix, project_point_to_detector
from xray_simulator.validation.xvr import XvrView, XvrVolumeFrame, adapt_xvr_camera, load_xvr_view, load_xvr_volume


def intrinsics(**updates):
    return dict(sdd=1000.0, delx=0.4, dely=0.6, x0=12.0, y0=-7.0, height=120, width=160) | updates


def frame(handedness=1, oblique=False):
    affine = np.diag([0.8, 1.2, handedness * 2.0, 1.0])
    if oblique:
        affine[:3, :3] = Rotation.from_euler("xyz", [21, -17, 9], degrees=True).as_matrix() @ affine[:3, :3]
    affine[:3, 3] = [127.0, -68.0, 314.0]
    return XvrVolumeFrame.from_affine((7, 9, 11), affine)


@pytest.mark.parametrize("handedness", [-1, 1])
@pytest.mark.parametrize("oblique", [False, True])
def test_voxel_centers_and_values_survive_reindexing(handedness, oblique):
    f = frame(handedness, oblique)
    original = np.arange(7 * 9 * 11).reshape(7, 9, 11)
    prepared = f.prepare_array(original)
    for ijk in ([0, 0, 0], [6, 8, 10], [2, 7, 3]):
        index = np.array(ijk)
        target = index.copy()
        if f.flip_y:
            target[1] = 8 - target[1]
        center = np.array(f.origin_xyz_mm) + (target + 0.5) * f.spacing_xyz_mm
        ras = f.affine_ras[:3, :3] @ index + f.affine_ras[:3, 3]
        np.testing.assert_allclose(f.points_to_simulator(ras, centered=False), center, atol=1e-12)
        assert prepared[tuple(target[::-1])] == original[tuple(index)]
    np.testing.assert_allclose(np.linalg.det(f.simulator_to_centered_ras), -1, atol=1e-12)


@pytest.mark.parametrize("binning", [1, 2, 4])
@pytest.mark.parametrize("handedness", [-1, 1])
def test_projection_against_independent_pinhole_matrix(binning, handedness):
    f = frame(handedness, oblique=True)
    p = np.eye(4)
    p[:3, :3] = Rotation.from_euler("zyx", [30, -10, 72], degrees=True).as_matrix()
    p[:3, 3] = [20, -700, 80]
    intr = intrinsics()
    camera = adapt_xvr_camera(p, intr, f, binning=binning)
    # Conventional camera coordinates, then an independently assembled 3x4 projection.
    ap = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    cam_to_ras = np.eye(4)
    cam_to_ras[:3, :3] = p[:3, :3] @ ap
    cam_to_ras[:3, 3] = p[:3, 3]
    dx, dy = intr["delx"] * binning, intr["dely"] * binning
    k = np.array(
        [
            [intr["sdd"] / dx, 0, (intr["width"] / binning - 1) / 2 - intr["x0"] / dx],
            [0, -intr["sdd"] / dy, (intr["height"] / binning - 1) / 2 + intr["y0"] / dy],
            [0, 0, 1],
        ]
    )
    points_camera = np.random.default_rng(42).uniform([-25, -20, 350], [25, 20, 850], (30, 3))
    points_ras = points_camera @ cam_to_ras[:3, :3].T + cam_to_ras[:3, 3]
    projection = k @ np.linalg.inv(cam_to_ras)[:3]
    homogeneous = np.column_stack([points_ras, np.ones(len(points_ras))]) @ projection.T
    expected = homogeneous[:, :2] / homogeneous[:, 2, None]
    np.testing.assert_allclose(camera.project(points_ras), expected, atol=1e-9)
    r = euler_zxy_to_matrix(camera.pose.rotation)
    source = np.array(camera.pose.translation) - camera.geometry.source_to_isocenter_mm * r[:, 2]
    np.testing.assert_allclose(source, f.points_to_simulator(p[:3, 3]), atol=1e-10)


def test_binning_preserves_block_centers():
    f, p = frame(), np.eye(4)
    p[1, 3] = 700
    points = np.array([[0, 0, 0], [15, 5, -20]])
    full = adapt_xvr_camera(p, intrinsics(), f).project(points)
    binned = adapt_xvr_camera(p, intrinsics(), f, binning=4).project(points)
    np.testing.assert_allclose(binned, (full + 0.5) / 4 - 0.5)


def test_literal_off_center_rectangular_detector():
    g = CarmGeometry(1000, 500, 100, 80, 0.5, 0.25, (10, -5))
    # At isocenter magnification is 2: x=5 -> column 49.5; y=-2.5 -> row 39.5.
    assert project_point_to_detector((5, -2.5, 0), (0, 0, 0), (0, 0, 0), **asdict(g)) == (49.5, 39.5)
    assert g.detector_size_mm == (50.0, 20.0)
    assert project_point_to_detector((0, 0, 0), (0, 0, 0), (0, 0, 0), **asdict(g)) == (29.5, 59.5)


def test_default_detector_compatibility():
    g = CarmGeometry()
    pixel = project_point_to_detector((12.375, 0, 0), (0, 0, 0), (0, 0, 0), **asdict(g))
    assert pixel == (305.0, 255.5)
    assert g.detector_size_mm == (256.0, 256.0)


@pytest.mark.parametrize(
    "key,value", [("sdd", -1), ("delx", 0), ("dely", np.nan), ("x0", np.inf), ("width", 12.5), ("height", True)]
)
def test_reject_invalid_intrinsics(key, value):
    with pytest.raises(ValueError):
        adapt_xvr_camera(np.eye(4), intrinsics(**{key: value}), frame())


@pytest.mark.parametrize("binning", [0, -2, 3, 1.5, True])
def test_reject_invalid_binning(binning):
    with pytest.raises(ValueError):
        adapt_xvr_camera(np.eye(4), intrinsics(), frame(), binning=binning)


def test_reject_reflected_pose_and_sheared_affine():
    pose = np.eye(4)
    pose[0, 0] = -1
    with pytest.raises(ValueError, match="proper rigid"):
        adapt_xvr_camera(pose, intrinsics(), frame())
    affine = np.eye(4)
    affine[0, 1] = 0.1
    with pytest.raises(ValueError, match="Sheared"):
        XvrVolumeFrame.from_affine((5, 5, 5), affine)


def test_reference_crop_and_block_average():
    view = XvrView(
        "deepfluoro", "subject01", "000", Path("ref.dcm"), Path("pose.pt"), np.eye(4), intrinsics(height=4, width=8)
    )
    raw = np.arange(104 * 108, dtype=float).reshape(104, 108)
    result = view.prepare_reference(raw, binning=2)
    assert result.shape == (2, 4)
    assert result[0, 0] == raw[50:52, 50:52].mean()
    assert result[-1, -1] == raw[52:54, 56:58].mean()
    with pytest.raises(ValueError, match="shape"):
        view.prepare_reference(raw[1:])


def test_optional_file_loaders(tmp_path):
    nib = pytest.importorskip("nibabel")
    torch = pytest.importorskip("torch")
    volume = np.arange(7 * 9 * 11, dtype=np.float32).reshape(7, 9, 11)
    nib.save(nib.Nifti1Image(volume, np.diag([-1.0, -1.0, 1.0, 1.0])), tmp_path / "volume.nii.gz")
    values, f = load_xvr_volume(tmp_path / "volume.nii.gz")
    np.testing.assert_array_equal(values, f.prepare_array(volume))
    subject = tmp_path / "subject01"
    (subject / "xrays").mkdir(parents=True)
    torch.save({"pose": torch.eye(4)[None], "intrinsics": intrinsics()}, subject / "xrays/000.pt")
    view = load_xvr_view(subject, "000")
    assert view.camera(f).geometry.detector_width_px == 160
    with pytest.raises(ValueError, match="excluded"):
        load_xvr_view(subject, "003")
    with pytest.raises(ValueError, match="stem"):
        load_xvr_view(subject, "../000")


@pytest.mark.gpu
def test_gpu_detector_offset_and_pixel_pitch():
    from xray_simulator import SimulatorConfig, xray_simulator
    from xray_simulator.config import XrayPhysics
    from xray_simulator.volume import PreprocessedVolume, VolumeMetadata

    # A compact symmetric attenuation bead. Projection shifts are literal values:
    # detector offset (+4, -6) / pitch (1, 2) => image shift (-4, +3).
    z, y, x = np.mgrid[:24, :24, :24] - 11.5
    mu = np.exp(-(x * x + y * y + z * z) / 2).astype(np.float32) * 0.05
    volume = PreprocessedVolume(mu, VolumeMetadata(mu.shape, (1.0, 1.0, 1.0), (-12.0, -12.0, -12.0)))
    geometry = CarmGeometry(1000, 500, 64, 48, 1.0, 2.0, (4.0, -6.0))
    config = SimulatorConfig(geometry=geometry, physics=XrayPhysics(step_mm=0.1)).with_output(keep_intensity=True)
    simulator = xray_simulator(volume, config)
    a = -np.log(simulator.render_frame().intensity)
    yy, xx = np.indices(a.shape)
    centroid = np.array([(xx * a).sum(), (yy * a).sum()]) / a.sum()
    np.testing.assert_allclose(centroid, [27.5, 26.5], atol=0.05)
