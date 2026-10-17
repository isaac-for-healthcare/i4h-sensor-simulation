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

"""The Slang DiffDRR wrappers hand their configuration to the renderer as ``cfg``.

``SlangDiffDRRRenderer`` takes ``origin_xyz_mm`` third and ``cfg`` fourth. A
wrapper passing its config positionally put it in the origin slot, so every
``TorchSlangDiffDRR`` and ``render_diffdrr_slang`` call failed with
``'SlangDiffDRRConfig' object is not iterable``, the default config included.
"""

import numpy as np
import pytest
from xray_simulator.rendering import diffdrr_slang_renderer as diffdrr

MU = np.zeros((4, 4, 4), np.float32)
SPACING = (1.0, 1.0, 1.0)


class _RecordingRenderer:
    """Stands in for ``SlangDiffDRRRenderer``, which needs Slang and a GPU."""

    calls: list = []

    def __init__(self, *args, **kwargs):
        type(self).calls.append((args, kwargs))

    def render(self, rotation, translation):
        return np.zeros((2, 2), np.float32)


@pytest.fixture
def recorded(monkeypatch):
    _RecordingRenderer.calls = []
    monkeypatch.setattr(diffdrr, "SlangDiffDRRRenderer", _RecordingRenderer)
    return _RecordingRenderer.calls


def _cfg_and_origin(call):
    """What the real renderer would bind to ``cfg`` and ``origin_xyz_mm``."""
    args, kwargs = call
    bound = dict(zip(("mu_volume", "spacing_zyx_mm", "origin_xyz_mm", "cfg"), args), **kwargs)
    return bound.get("cfg"), bound.get("origin_xyz_mm", (0.0, 0.0, 0.0))


def test_render_diffdrr_slang_passes_its_config_as_cfg(recorded):
    cfg = diffdrr.SlangDiffDRRConfig(det_width_px=64, det_height_px=32)
    diffdrr.render_diffdrr_slang(MU, SPACING, cfg=cfg)
    assert _cfg_and_origin(recorded[0]) == (cfg, (0.0, 0.0, 0.0))


@pytest.mark.parametrize("explicit", [True, False], ids=["explicit-cfg", "default-cfg"])
def test_torch_slang_diffdrr_passes_its_config_as_cfg(recorded, explicit):
    pytest.importorskip("torch")
    cfg = diffdrr.SlangDiffDRRConfig(det_width_px=64, det_height_px=32)
    module = diffdrr.TorchSlangDiffDRR(MU, SPACING, cfg) if explicit else diffdrr.TorchSlangDiffDRR(MU, SPACING)
    passed_cfg, origin = _cfg_and_origin(recorded[0])
    assert isinstance(passed_cfg, diffdrr.SlangDiffDRRConfig)
    assert origin == (0.0, 0.0, 0.0)
    assert passed_cfg is module._cfg
    if explicit:
        assert passed_cfg is cfg


@pytest.mark.gpu
def test_torch_slang_diffdrr_yields_pose_gradients():
    """The documented registration snippet: build with a config, render, backpropagate."""
    torch = pytest.importorskip("torch")
    if not diffdrr.SLANG_AVAILABLE:
        pytest.skip("slangpy is not installed")
    z, y, x = np.mgrid[:32, :32, :32].astype(np.float32)
    mu = np.full((32, 32, 32), 0.002, np.float32)
    mu[(z - 16) ** 2 + (y - 14) ** 2 + (x - 18) ** 2 < 7**2] = 0.05
    cfg = diffdrr.SlangDiffDRRConfig(det_width_px=64, det_height_px=48)
    drr = diffdrr.TorchSlangDiffDRR(mu, (2.0, 2.0, 2.0), cfg)

    with torch.no_grad():
        target = drr(torch.tensor([0.0, 0.0, 0.05]), torch.tensor([4.0, -3.0, 0.0]))
    rotation = torch.zeros(3, requires_grad=True)
    translation = torch.zeros(3, requires_grad=True)
    image = drr(rotation, translation)
    (image - target).pow(2).mean().backward()

    assert image.shape == (48, 64)
    for grad in (rotation.grad, translation.grad):
        assert grad.shape == (3,)
        assert torch.isfinite(grad).all()
    assert translation.grad.abs().sum() > 0


ROTATIONS = np.array([[0.0, 0.0, 0.0], [0.3, -0.2, 0.1], [-0.4, 0.25, 1.2]], np.float32)
TRANSLATIONS = np.array([[0.0, 0.0, 0.0], [5.0, -3.0, 2.0], [-4.0, 6.0, -1.0]], np.float32)


def _phantom_renderer(device_type="cuda", **cfg_kwargs):
    """A sphere in a faint box, anisotropic voxels so the three axes are distinguishable."""
    if not diffdrr.SLANG_AVAILABLE:
        pytest.skip("slangpy is not installed")
    z, y, x = np.mgrid[:48, :40, :32].astype(np.float32)
    mu = np.full((48, 40, 32), 0.002, np.float32)
    mu[(z - 24) ** 2 + (y - 18) ** 2 + (x - 17) ** 2 < 9**2] = 0.05
    cfg = diffdrr.SlangDiffDRRConfig(det_width_px=64, det_height_px=48, device_type=device_type, **cfg_kwargs)
    try:
        return diffdrr.SlangDiffDRRRenderer(mu, (2.0, 1.5, 1.0), cfg=cfg)
    except Exception as exc:  # noqa: BLE001 - a missing backend is a skip, not a failure
        pytest.skip(f"Slang {device_type} device unavailable: {exc}")


@pytest.mark.gpu
@pytest.mark.parametrize("device_type", ["cuda", "vulkan"])
@pytest.mark.parametrize("normalize", [False, True], ids=["raw", "normalized"])
def test_a_batch_renders_each_pose_as_render_does(device_type, normalize):
    renderer = _phantom_renderer(device_type, normalize=normalize, invert=normalize)

    batch = renderer.render_batch(ROTATIONS, TRANSLATIONS)

    assert batch.shape == (3, 48, 64) and batch.dtype == np.float32
    for index in range(3):
        np.testing.assert_allclose(batch[index], renderer.render(ROTATIONS[index], TRANSLATIONS[index]), atol=1e-6)
    assert not np.allclose(batch[0], batch[1])


@pytest.mark.gpu
def test_a_torch_batch_stays_on_the_gpu_and_survives_the_next_call():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available to torch")
    renderer = _phantom_renderer("cuda", normalize=False, invert=False)

    first = renderer.render_batch(ROTATIONS, TRANSLATIONS, as_torch=True)
    expected = first.cpu().numpy().copy()
    renderer.render_batch(ROTATIONS[::-1].copy(), TRANSLATIONS[::-1].copy(), as_torch=True)

    assert first.is_cuda and first.shape == (3, 48, 64) and first.dtype == torch.float32
    np.testing.assert_array_equal(first.cpu().numpy(), expected)
    np.testing.assert_allclose(expected, renderer.render_batch(ROTATIONS, TRANSLATIONS), atol=1e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("device_type", ["cuda", "vulkan"])
def test_hardware_sampling_matches_the_software_line_integrals(device_type):
    """One filtered fetch per step instead of eight point fetches gives the same attenuation."""
    software = _phantom_renderer(device_type, normalize=False, invert=False)
    hardware = _phantom_renderer(device_type, normalize=False, invert=False, hardware_sampling=True)

    reference = -np.log(software.render_batch(ROTATIONS, TRANSLATIONS))
    filtered = -np.log(hardware.render_batch(ROTATIONS, TRANSLATIONS))

    assert reference.max() > 0.5  # the sphere is in view, so the comparison has signal
    assert np.abs(filtered - reference).max() < 5e-3 * reference.max()
    np.testing.assert_allclose(hardware.render(ROTATIONS[1], TRANSLATIONS[1]), np.exp(-filtered[1]), atol=1e-6)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("rotations", "translations"),
    [(np.zeros((2, 3)), np.zeros((3, 3))), (np.zeros(3), np.zeros(3)), (np.zeros((0, 3)), np.zeros((0, 3)))],
    ids=["mismatched", "unbatched", "empty"],
)
def test_render_batch_rejects_malformed_poses(rotations, translations):
    renderer = _phantom_renderer()
    with pytest.raises(ValueError):
        renderer.render_batch(rotations, translations)
