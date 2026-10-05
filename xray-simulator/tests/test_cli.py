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

"""Config-file launcher uses the same validated settings as the Python API."""

import json
import subprocess
import sys

import numpy as np
import pytest
import yaml
from PIL import Image
from xray_simulator import HuToMuMapping, SimulatorConfig
from xray_simulator.cli import main
from xray_simulator.simulator import xray_simulator


@pytest.fixture
def cpu_transport(monkeypatch):
    calls = []

    def initialize(simulator):
        # Replace GPU transport only; configuration, HU mapping, effects and export
        # execute the production implementation.
        class Renderer:
            def render(self, rotation, translation):
                calls.append((tuple(rotation), tuple(translation)))
                a = simulator.volume.mu_volume.sum(axis=0) * simulator.volume.spacing_zyx_mm[0]
                return (simulator.config.physics.i0 * np.exp(-a)).astype(np.float32)

        simulator._renderer = Renderer()

    monkeypatch.setattr(xray_simulator, "_init_renderer", initialize)
    return calls


def preset(tmp_path, name="preset", suffix=".json", update=None):
    data = SimulatorConfig().with_output(format="npy").to_dict()
    if update:
        update(data)
    path = tmp_path / (name + suffix)
    path.write_text(json.dumps(data) if suffix == ".json" else yaml.safe_dump(data))
    return path


def render(tmp_path, name, update=None, *options, suffix=".json"):
    path = preset(tmp_path, name, suffix, update)
    output = tmp_path / name
    assert main(["render", "--config", str(path), "--synthetic", "--output", str(output), *options]) == 0
    return np.load(output / "frame_0000.npy"), json.loads((output / "run.json").read_text())


def modify(section, **fields):
    def update(data):
        node = data
        for key in section.split("."):
            node = node[key]
        node.update(fields)

    return update


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_dryrun_never_loads_data_or_initializes_renderer(tmp_path, monkeypatch, capsys, suffix):
    def forbidden(*args, **kwargs):
        pytest.fail("dryrun must not load voxels or touch the GPU")

    monkeypatch.setattr("xray_simulator.cli._load_volume", forbidden)
    monkeypatch.setattr(xray_simulator, "_init_renderer", forbidden)
    path = preset(tmp_path, suffix=suffix, update=modify("post_processing.display", polarity="diagnostic", gamma=1.7))
    output = tmp_path / "preview"
    assert main(["render", "--config", str(path), "--synthetic", "--output", str(output), "--dryrun"]) == 0
    plan = json.loads(capsys.readouterr().out)
    expected = SimulatorConfig.from_preset(path).with_output(output_dir=str(output)).to_dict()
    assert plan["preset"] == expected
    assert not output.exists()


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_piecewise_transfer_is_applied_and_saved(tmp_path, suffix):
    output = tmp_path / "cache"
    mapping = HuToMuMapping(control_points=((-1000, 0), (0, 0.01), (1000, 0.05)))
    config = SimulatorConfig().with_preprocessing(clip_hu=False, hu_to_mu=mapping)
    path = config.save_preset(tmp_path / ("preset" + suffix))
    assert main(["preprocess", "--config", str(path), "--synthetic", "--output", str(output)]) == 0
    np.testing.assert_allclose(np.unique(np.load(output / "mu_volume.npy")), [0, 0.0116, 0.046], atol=1e-7)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["hu_to_mu"]["control_points"] == [[-1000.0, 0.0], [0.0, 0.01], [1000.0, 0.05]]
    assert metadata["anatomical_frame"] == "LPS"


def test_hu_window_and_clipping_reach_preprocessor(tmp_path):
    def update(data):
        data["preprocessing"] = {
            "hu_clip_min": 0,
            "hu_clip_max": 100,
            "hu_to_mu": {"window_center": 100, "window_width": 200, "mu_min": 0.001, "mu_max": 0.021},
        }

    path = preset(tmp_path, update=update)
    output = tmp_path / "cache"
    main(["preprocess", "--config", str(path), "--synthetic", "--output", str(output)])
    np.testing.assert_allclose(np.unique(np.load(output / "mu_volume.npy")), [0.001, 0.005, 0.011], atol=1e-7)


def test_json_yaml_have_identical_rendered_results(tmp_path, cpu_transport):
    changes = modify(
        "post_processing.realism",
        enabled=True,
        gain=0.9,
        bias=0.01,
        poisson_photons=500,
        gaussian_sigma=0.01,
        blur_sigma_px=0.7,
        seed=19,
    )
    first, _ = render(tmp_path, "json", changes)
    second, _ = render(tmp_path, "yaml", changes, suffix=".yaml")
    np.testing.assert_array_equal(first, second)


def test_polarity_gamma_and_scaling(tmp_path, cpu_transport):
    fluoro, _ = render(tmp_path, "fluoro")
    xray, _ = render(tmp_path, "xray", modify("post_processing.display", polarity="diagnostic"))
    np.testing.assert_allclose(xray, 1 - fluoro, atol=1e-7)
    gamma, _ = render(tmp_path, "gamma", modify("post_processing.display", gamma=2))
    np.testing.assert_allclose(gamma, np.sqrt(fluoro), atol=1e-7)
    transmission, _ = render(tmp_path, "transmission", modify("post_processing.display", scaling="transmission"))
    window, _ = render(tmp_path, "window", modify("post_processing.display", scaling="window", window=[0.4, 0.9]))
    np.testing.assert_allclose(window, np.clip((transmission - 0.4) / 0.5, 0, 1), atol=1e-7)


def test_hu_settings_reach_rendering(tmp_path, cpu_transport):
    baseline, _ = render(tmp_path, "base")
    stronger, _ = render(tmp_path, "stronger", modify("preprocessing.hu_to_mu", mu_max=0.04))
    assert np.mean(stronger) < np.mean(baseline)


def test_gain_bias_i0_and_intensity_export(tmp_path, cpu_transport):
    def update(data):
        data["beam"]["i0"] = 2
        data["post_processing"]["display"]["scaling"] = "transmission"
        data["post_processing"]["realism"].update(enabled=True, gain=0, bias=0.25)
        data["output"]["keep_intensity"] = True

    display, plan = render(tmp_path, "effects", update)
    np.testing.assert_allclose(display, 0.125)
    np.testing.assert_allclose(np.load(tmp_path / "effects/intensity_0000.npy"), 0.25)
    assert plan["preset"]["post_processing"]["realism"]["enabled"] is True


def test_seeded_cine_repeats_run_but_not_each_frame(tmp_path, cpu_transport):
    def update(data):
        data["post_processing"]["realism"].update(enabled=True, seed=17, poisson_photons=500, gaussian_sigma=0.01)
        data["output"]["keep_intensity"] = True

    render(tmp_path, "first", update, "--frames", "3")
    render(tmp_path, "second", update, "--frames", "3")
    first = [np.load(tmp_path / "first" / f"intensity_{i:04d}.npy") for i in range(3)]
    for i in range(3):
        np.testing.assert_array_equal(first[i], np.load(tmp_path / "second" / f"intensity_{i:04d}.npy"))
    assert not np.array_equal(first[0], first[1])


def test_calibration_is_frozen_and_recorded(tmp_path, cpu_transport):
    _, plan = render(tmp_path, "calibrated", None, "--calibrate-display", "1", "99", "--frames", "3")
    assert len(cpu_transport) == 4
    assert plan["preset"]["post_processing"]["display"]["log_window"] != [0.0, 6.0]
    np.testing.assert_array_equal(
        np.load(tmp_path / "calibrated/frame_0000.npy"), np.load(tmp_path / "calibrated/frame_0002.npy")
    )
    SimulatorConfig.from_dict(plan["preset"])


def test_cached_volume_retains_mapping(tmp_path, cpu_transport):
    path = preset(tmp_path)
    cache = tmp_path / "cache"
    main(["preprocess", "--config", str(path), "--synthetic", "--output", str(cache)])
    main(["render", "--config", str(path), "--cache", str(cache), "--view", "ap", "--output", str(tmp_path / "valid")])
    path = preset(tmp_path, "custom", update=modify("preprocessing.hu_to_mu", mu_max=0.05))
    with pytest.raises(SystemExit) as exc:
        main(["render", "--config", str(path), "--cache", str(cache), "--output", str(tmp_path / "bad")])
    assert exc.value.code == 2
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("fmt", ["npy", "npz", "png"])
def test_preset_output_directory_format_and_single_writer(tmp_path, cpu_transport, fmt):
    output = tmp_path / "frames"
    path = preset(tmp_path, update=modify("output", output_dir=str(output), format=fmt, save_to_disk=True))
    main(["render", "--config", str(path), "--synthetic"])
    assert sorted(p.name for p in output.iterdir()) == [f"frame_0000.{fmt}", "run.json"]
    if fmt == "npy":
        image = np.load(output / "frame_0000.npy")
    elif fmt == "npz":
        with np.load(output / "frame_0000.npz") as archive:
            image = archive["image"]
    else:
        with Image.open(output / "frame_0000.png") as png:
            assert png.mode == "L"
            image = np.array(png)
    assert image.shape == (64, 64)


@pytest.mark.parametrize(
    "options",
    [
        ["--gamma", "1.2"],
        ["--gain", "2"],
        ["--hu-clip", "0", "100"],
        ["--frames", "0"],
        ["--fps", "nan"],
        ["--calibrate-display", "90", "10"],
    ],
)
def test_invalid_options_fail_before_rendering(tmp_path, monkeypatch, options):
    monkeypatch.setattr(xray_simulator, "_init_renderer", lambda *a: pytest.fail("GPU must not be initialized"))
    path = preset(tmp_path)
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit) as exc:
        main(["render", "--config", str(path), "--synthetic", "--output", str(output), *options])
    assert exc.value.code == 2
    assert not output.exists()


def test_invalid_preset_fails_before_loading_volume(tmp_path, monkeypatch):
    monkeypatch.setattr("xray_simulator.cli._load_volume", lambda *a: pytest.fail("must validate first"))
    path = preset(tmp_path, update=modify("beam", i0=-1))
    with pytest.raises(SystemExit) as exc:
        main(["render", "--config", str(path), "--synthetic", "--output", str(tmp_path / "out")])
    assert exc.value.code == 2


def test_existing_outputs_are_preserved(tmp_path):
    path = preset(tmp_path)
    sentinel = tmp_path / "existing.txt"
    sentinel.write_text("keep")
    with pytest.raises(SystemExit):
        main(["preprocess", "--config", str(path), "--synthetic", "--output", str(tmp_path)])
    assert sentinel.read_text() == "keep"


def test_module_help_lists_config_file():
    result = subprocess.run(
        [sys.executable, "-m", "xray_simulator", "render", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--config" in result.stdout and "--gain" not in result.stdout
