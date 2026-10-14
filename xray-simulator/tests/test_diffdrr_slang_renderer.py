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
