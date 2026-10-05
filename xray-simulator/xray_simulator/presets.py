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

"""Versioned JSON/YAML presets backing SimulatorConfig's serialization API.

The packaged JSON Schema defines fields, constraints, and v1 defaults. Loading
never imports or initializes a GPU backend. Cross-field geometry and display
constraints are checked after structural validation.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, cast

import yaml

from .config import (
    CarmGeometry,
    DisplaySettings,
    HuToMuMapping,
    MetricsSettings,
    OutputSettings,
    PreprocessingSettings,
    RealismSettings,
    SimulatorConfig,
    XrayPhysics,
    resolve_display_settings,
)

if TYPE_CHECKING:
    from jsonschema import Draft202012Validator


def get_preset_schema() -> dict[str, Any]:
    """Return a fresh copy of the bundled JSON Schema for preset version 1.

    The schema is available in installed wheels as well as source checkouts.
    Its defaults are applied by SimulatorConfig.from_dict/from_preset; generic
    JSON Schema validators validate structure without filling in defaults.
    """
    resource = files("xray_simulator").joinpath("schemas/preset-v1.schema.json")
    return cast(dict[str, Any], json.loads(resource.read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    from jsonschema import Draft202012Validator

    return Draft202012Validator(get_preset_schema())


def _fill_defaults(data: dict[str, Any], schema: dict[str, Any]) -> None:
    """Fill a validated private dictionary using the versioned schema defaults."""
    for name, definition in schema.get("properties", {}).items():
        if name not in data and "default" in definition:
            data[name] = deepcopy(definition["default"])
        if name in data and definition.get("type") == "object":
            _fill_defaults(data[name], definition)


def _validated(data: dict[str, Any]) -> dict[str, Any]:
    # JSON round-trip also detaches caller data, normalizes tuples to arrays, and
    # rejects NaN/Infinity and objects which cannot be represented in a preset.
    try:
        document = json.loads(json.dumps(data, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"preset must contain only finite JSON values: {exc}") from exc
    validator = _validator()
    error = next(validator.iter_errors(document), None)
    if error is not None:
        location = ".".join(map(str, error.absolute_path)) or "preset"
        raise ValueError(f"{location}: {error.message}")
    _fill_defaults(document, validator.schema)
    return cast(dict[str, Any], document)


def config_from_dict(data: dict[str, Any]) -> SimulatorConfig:
    """Implementation of SimulatorConfig.from_dict; see that method's public contract."""
    document = _validated(data)
    geometry, detector = document["geometry"], document["detector"]
    if geometry["source_to_isocenter_mm"] >= geometry["source_to_detector_mm"]:
        raise ValueError("geometry.source_to_isocenter_mm must be smaller than source_to_detector_mm")
    post = document["post_processing"]
    try:
        display = DisplaySettings(**post["display"])
    except ValueError as exc:
        raise ValueError(f"post_processing.display: {exc}") from exc
    realism = post["realism"]
    if realism["seed"] is not None:
        realism["seed"] = int(realism["seed"])
    preprocessing = document["preprocessing"]
    if preprocessing["hu_clip_min"] >= preprocessing["hu_clip_max"]:
        raise ValueError("preprocessing.hu_clip_min must be smaller than hu_clip_max")
    try:
        mapping = HuToMuMapping.from_dict(preprocessing["hu_to_mu"])
    except ValueError as exc:
        raise ValueError(f"preprocessing.hu_to_mu: {exc}") from exc
    return SimulatorConfig(
        geometry=CarmGeometry(
            **geometry,
            detector_width_px=int(detector["width_px"]),
            detector_height_px=int(detector["height_px"]),
            pixel_spacing_mm=detector["pixel_spacing_mm"],
        ),
        physics=XrayPhysics(**document["beam"]),
        display=display,
        realism=RealismSettings(**realism),
        output=OutputSettings(**document["output"]),
        metrics=MetricsSettings(**document["metrics"]),
        backend=document["backend"],
        preprocessing=PreprocessingSettings(**{**preprocessing, "hu_to_mu": mapping}),
    )


def config_to_dict(config: SimulatorConfig) -> dict[str, Any]:
    """Implementation of SimulatorConfig.to_dict, including legacy display resolution."""
    display = resolve_display_settings(config.physics, config.display)
    output = asdict(config.output)
    if isinstance(output["output_dir"], Path):
        output["output_dir"] = str(output["output_dir"])
    preprocessing = asdict(config.preprocessing)
    mapping = config.preprocessing.hu_to_mu
    preprocessing["hu_to_mu"] = (
        {"control_points": mapping.control_points} if mapping.control_points is not None else mapping.to_dict()
    )
    data = {
        "schema_version": 1,
        "modality": "xray",
        "beam": {"i0": config.physics.i0, "step_mm": config.physics.step_mm},
        "geometry": {
            "source_to_detector_mm": config.geometry.source_to_detector_mm,
            "source_to_isocenter_mm": config.geometry.source_to_isocenter_mm,
        },
        "detector": {
            "width_px": config.geometry.detector_width_px,
            "height_px": config.geometry.detector_height_px,
            "pixel_spacing_mm": config.geometry.pixel_spacing_mm,
        },
        "post_processing": {"realism": asdict(config.realism), "display": asdict(display)},
        "backend": config.backend,
        "output": output,
        "metrics": asdict(config.metrics),
        "preprocessing": preprocessing,
    }
    # Validate direct dataclass instances too, before any file can be overwritten.
    config_from_dict(data)
    return cast(dict[str, Any], json.loads(json.dumps(data, allow_nan=False)))


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


class _UniqueKeyLoader(yaml.SafeLoader):
    """Read data-only YAML and reject duplicate/non-string mapping keys."""

    def construct_mapping(self, node, deep=False):
        pairs = []
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("YAML preset keys must be strings")
            pairs.append((key, self.construct_object(value_node, deep=deep)))
        return _unique_keys(pairs)


def _preset_path(path: str | Path) -> Path:
    result = Path(path).expanduser().absolute()
    if result.suffix.lower() not in (".json", ".yaml", ".yml"):
        raise ValueError("Preset files must use the .json, .yaml, or .yml extension")
    return result


def load_preset(path: str | Path) -> SimulatorConfig:
    """Read JSON or safe YAML, rejecting duplicate keys before schema validation."""
    path = _preset_path(path)
    text = path.read_text(encoding="utf-8")
    try:
        document = (
            json.loads(text, object_pairs_hook=_unique_keys)
            if path.suffix.lower() == ".json"
            else yaml.load(text, Loader=_UniqueKeyLoader)
        )
        return config_from_dict(document)
    except (ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid preset {path}: {exc}") from exc


def save_preset(config: SimulatorConfig, path: str | Path) -> Path:
    """Implementation of SimulatorConfig.save_preset; publish only a complete document."""
    path = _preset_path(path)
    document = config_to_dict(config)
    text = (
        json.dumps(document, indent=2, allow_nan=False) + "\n"
        if path.suffix.lower() == ".json"
        else yaml.safe_dump(document, sort_keys=False)
    )
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
