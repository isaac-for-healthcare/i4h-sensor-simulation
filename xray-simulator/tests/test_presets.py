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

"""Preset validation, portable serialization, and CPU-only simulator integration."""

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from jsonschema import Draft202012Validator
from xray_simulator import (
    CarmGeometry,
    DisplaySettings,
    HuToMuMapping,
    MetricsSettings,
    OutputSettings,
    RealismSettings,
    SimulatorConfig,
    VolumePreprocessor,
    XrayPhysics,
    get_preset_schema,
    xray_simulator,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/presets"


def minimal():
    return {"schema_version": 1, "modality": "xray", "beam": {}, "geometry": {}, "detector": {}, "post_processing": {}}


def test_versioned_defaults_and_no_mutation():
    data = minimal()
    original = deepcopy(data)
    config = SimulatorConfig.from_dict(data)
    assert config == SimulatorConfig()
    assert data == original
    serialized = config.to_dict()
    assert serialized["beam"] == {"i0": 1.0, "step_mm": 0.5}
    assert serialized["post_processing"]["display"]["log_window"] == [0.0, 6.0]
    serialized["geometry"]["source_to_detector_mm"] = 1
    assert config.geometry.source_to_detector_mm == 1020.0


def test_nondefault_config_file_roundtrip(tmp_path):
    config = SimulatorConfig(
        geometry=CarmGeometry(1200.0, 750.0, 640, 480, 0.2),
        physics=XrayPhysics(step_mm=0.8, i0=3.0),
        display=DisplaySettings(
            polarity="diagnostic", scaling="window", window=(0.1, 0.7), log_window=(1.0, 4.0), gamma=1.2
        ),
        realism=RealismSettings(True, 1.1, -0.1, 3000.0, 0.01, 0.5, 123),
        output=OutputSettings(True, "relative-frames", "npz", True, True),
        metrics=MetricsSettings(True, False, False, False),
    )
    path = config.save_preset(tmp_path / "custom.json")
    assert path.is_absolute()
    assert SimulatorConfig.from_preset(path) == config
    document = json.loads(path.read_text())
    assert document["detector"] == {"width_px": 640, "height_px": 480, "pixel_spacing_mm": 0.2}
    assert document["output"]["output_dir"] == "relative-frames"
    assert SimulatorConfig.from_dict(document).to_dict() == document


@pytest.mark.parametrize("name,polarity", [("fluoroscopy", "fluoro"), ("radiograph", "diagnostic")])
def test_examples_match_public_schema(name, polarity):
    schema = get_preset_schema()
    Draft202012Validator.check_schema(schema)
    data = json.loads((EXAMPLES / f"{name}.json").read_text())
    Draft202012Validator(schema).validate(data)
    config = SimulatorConfig.from_preset(EXAMPLES / f"{name}.json")
    assert config.display.polarity == polarity
    assert SimulatorConfig.from_dict(config.to_dict()) == config
    schema["properties"].clear()
    assert "beam" in get_preset_schema()["properties"]


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema_version",), 2),
        (("schema_version",), True),
        (("modality",), "ultrasound"),
        (("backend",), "warp"),
        (("beam", "i0"), 0),
        (("beam", "step_mm"), -1),
        (("beam", "i0"), "1.0"),
        (("beam", "i0"), True),
        (("beam", "kvp"), 80),
        (("detector", "width_px"), 1.5),
        (("detector", "height_px"), False),
        (("detector", "height_px"), 0),
        (("detector", "pixel_spacing_mm"), 0),
        (("geometry", "source_to_isocenter_mm"), 2000),
        (("geometry", "source_to_detector_mm"), 510),
        (("post_processing", "display", "window"), [0.4, 0.4]),
        (("post_processing", "display", "window"), [-0.1, 0.4]),
        (("post_processing", "display", "log_window"), [4, 1]),
        (("post_processing", "display", "log_window"), [0, 1, 2]),
        (("post_processing", "display", "gamma"), 0),
        (("post_processing", "display", "polarity"), "xray"),
        (("post_processing", "realism", "seed"), -1),
        (("post_processing", "realism", "enabled"), "false"),
        (("post_processing", "realism", "blur_sigma_px"), -1),
        (("post_processing", "realism", "gaussian_sigma"), -1),
        (("post_processing", "realism", "poisson_photons"), -1),
        (("post_processing", "realism", "gain"), -1),
        (("geometry", "pixel_spacing_mm"), 0.2),
        (("output", "format"), "jpeg"),
        (("metrics", "track_fps"), 1),
    ],
)
def test_invalid_fields_have_actionable_errors(path, value):
    data = minimal()
    node = data
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value
    with pytest.raises(ValueError, match=path[0]):
        SimulatorConfig.from_dict(data)


@pytest.mark.parametrize("field", list(minimal()))
def test_required_sections(field):
    data = minimal()
    del data[field]
    with pytest.raises(ValueError, match=field):
        SimulatorConfig.from_dict(data)


@pytest.mark.parametrize("value", [None, [], "preset", 1])
def test_root_must_be_an_object(value):
    with pytest.raises(ValueError, match="preset"):
        SimulatorConfig.from_dict(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_are_rejected_on_load_and_save(value, tmp_path):
    data = minimal()
    data["beam"]["i0"] = value
    with pytest.raises(ValueError, match="finite"):
        SimulatorConfig.from_dict(data)
    config = SimulatorConfig(physics=XrayPhysics(i0=value))
    target = tmp_path / "preset.json"
    target.write_text("previous content")
    with pytest.raises(ValueError, match="finite"):
        config.save_preset(target)
    assert target.read_text() == "previous content"


@pytest.mark.parametrize(
    "contents,match",
    [
        ('{"schema_version":1,"schema_version":2}', "duplicate"),
        ('{"beam":{"i0":1,"i0":2}}', "duplicate"),
        ('{"schema_version":', "Invalid preset"),
        (json.dumps(minimal()).replace('"beam": {}', '"beam": {"i0": 1e999}'), "finite"),
    ],
)
def test_invalid_json_file(contents, match, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(contents)
    with pytest.raises(ValueError, match=match):
        SimulatorConfig.from_preset(path)


def test_file_errors_and_output_paths(tmp_path):
    with pytest.raises(ValueError, match=".json"):
        SimulatorConfig.from_preset(tmp_path / "preset.toml")
    with pytest.raises(FileNotFoundError):
        SimulatorConfig.from_preset(tmp_path / "absent.json")
    config = SimulatorConfig(output=OutputSettings(save_to_disk=True, output_dir=tmp_path / "frames"))
    path = config.save_preset(tmp_path / "preset.json")
    assert SimulatorConfig.from_preset(path).output.output_dir == str(tmp_path / "frames")
    with pytest.raises(ValueError, match="output"):
        SimulatorConfig(output=OutputSettings(save_to_disk=True)).to_dict()
    # An invalid cross-field combination cannot replace an existing preset.
    original = path.read_bytes()
    with pytest.raises(ValueError, match="geometry"):
        config.with_geometry(source_to_detector_mm=100).save_preset(path)
    assert path.read_bytes() == original


def test_legacy_display_is_saved_as_effective_settings():
    config = SimulatorConfig(physics=XrayPhysics(normalize=True, invert=True))
    with pytest.warns(DeprecationWarning):
        data = config.to_dict()
    restored = SimulatorConfig.from_dict(data)
    assert restored.display == DisplaySettings.preset("legacy")
    assert restored.physics == XrayPhysics()
    assert "normalize" not in data["beam"] and "invert" not in data["beam"]


def test_preset_load_needs_no_gpu_modules():
    code = """
import importlib.abc
import sys
class NoGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'slangpy', 'torch', 'warp'}:
            raise AssertionError('GPU dependency imported: ' + fullname)
sys.meta_path.insert(0, NoGPU())
from xray_simulator import SimulatorConfig
assert SimulatorConfig.from_preset(sys.argv[1]).display.polarity == 'fluoro'
"""
    subprocess.run([sys.executable, "-I", "-c", code, str(EXAMPLES / "fluoroscopy.json")], check=True)


def test_preset_reaches_renderer_and_postprocessing(monkeypatch, tmp_path):
    """Use the real simulator pipeline with only GPU ray marching replaced."""
    renderer_module = ModuleType("xray_simulator.rendering.diffdrr_slang_renderer")
    initialized = []

    class Renderer:
        def __init__(self, **kwargs):
            self.config = kwargs["cfg"]
            initialized.append(self.config)

        def render(self, rotation, translation):
            cfg = self.config
            return (
                np.linspace(0.02, 0.95, cfg.det_height_px * cfg.det_width_px, dtype=np.float32).reshape(
                    cfg.det_height_px, cfg.det_width_px
                )
                * cfg.i0
            )

    renderer_module.SlangDiffDRRConfig = SimpleNamespace
    renderer_module.SlangDiffDRRRenderer = Renderer
    renderer_module.render_diffdrr_slang = None  # Unused re-export in rendering.__init__.
    monkeypatch.setitem(sys.modules, renderer_module.__name__, renderer_module)
    expected = SimulatorConfig(
        geometry=CarmGeometry(1100, 700, 12, 8, 0.3),
        physics=XrayPhysics(i0=2, step_mm=0.75),
        display=DisplaySettings(polarity="diagnostic", log_window=(0.2, 4), gamma=1.3),
        realism=RealismSettings(
            enabled=True, gain=0.9, bias=0.01, poisson_photons=5000, gaussian_sigma=0.005, blur_sigma_px=0.5, seed=73
        ),
        output=OutputSettings(keep_intensity=True),
    )
    loaded = SimulatorConfig.from_preset(expected.save_preset(tmp_path / "render.json"))
    volume = VolumePreprocessor.from_numpy(np.zeros((3, 4, 5)), spacing_zyx_mm=(1, 1, 1)).preprocess()
    reference_frame = xray_simulator(volume, expected).render_frame()
    loaded_frame = xray_simulator(volume, loaded).render_frame()
    assert initialized[0] == initialized[1]
    assert initialized[1].det_width_px == 12 and initialized[1].i0 == 2
    np.testing.assert_array_equal(loaded_frame.image, reference_frame.image)
    np.testing.assert_array_equal(loaded_frame.intensity, reference_frame.intensity)


@pytest.mark.parametrize("extension", [".json", ".yaml", ".yml", ".YAML"])
def test_preprocessing_and_yaml_roundtrip(extension, tmp_path):
    mapping = HuToMuMapping(control_points=((-1000, 0), (0, 0.01), (1000, 0.05)))
    config = SimulatorConfig().with_preprocessing(clip_hu=False, hu_to_mu=mapping)
    path = config.save_preset(tmp_path / ("preset" + extension))
    assert SimulatorConfig.from_preset(path) == config
    # Other helpers must retain preprocessing settings.
    assert config.with_geometry(detector_width_px=128).preprocessing == config.preprocessing


@pytest.mark.parametrize(
    "text,match",
    [
        ("schema_version: 1\nschema_version: 2", "duplicate"),
        ("beam:\n  i0: 1\n  i0: 2", "duplicate"),
        ("1: value", "keys must be strings"),
        ("modality: [xray", "Invalid preset"),
        ("!!python/object/apply:builtins.str [unsafe]", "Invalid preset"),
        ("---\nmodality: xray\n---\nmodality: xray", "Invalid preset"),
        ("beam: &beam\n  recursive: *beam", "finite JSON"),
    ],
)
def test_invalid_yaml_is_rejected(text, match, tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        SimulatorConfig.from_preset(path)


@pytest.mark.parametrize(
    "mapping",
    [
        {"hu_min": 100, "hu_max": 0},
        {"mu_max": -1},
        {"window_width": 0, "window_center": 100},
        {"window_center": 100},
        {"window_center": 100, "window_width": 200, "hu_min": 0},
        {"control_points": [[0, 0], [0, 1]]},
        {"control_points": [[0, 0], [1, -1]]},
        {"control_points": [[0, 0], [1, 1]], "mu_max": 1},
        {"unknown": 1},
    ],
)
def test_invalid_hu_mapping(mapping):
    data = minimal()
    data["preprocessing"] = {"hu_to_mu": mapping}
    with pytest.raises(ValueError, match="preprocessing"):
        SimulatorConfig.from_dict(data)


def test_hu_window_level_and_clip_validation():
    data = minimal()
    data["preprocessing"] = {"hu_to_mu": {"window_center": 100, "window_width": 200}}
    config = SimulatorConfig.from_dict(data)
    assert config.preprocessing.hu_to_mu.hu_min == 0
    assert config.preprocessing.hu_to_mu.hu_max == 200
    assert SimulatorConfig.from_dict(config.to_dict()) == config
    data["preprocessing"].update(hu_clip_min=10, hu_clip_max=0)
    with pytest.raises(ValueError, match="hu_clip_min"):
        SimulatorConfig.from_dict(data)


def test_yaml_and_json_examples_are_equivalent():
    for name in ("fluoroscopy", "radiograph"):
        yaml_path = EXAMPLES / f"{name}.yaml"
        assert SimulatorConfig.from_preset(yaml_path) == SimulatorConfig.from_preset(EXAMPLES / f"{name}.json")
