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

"""Analytical metric checks and manifest validation; no clinical data or GPU."""

import json

import numpy as np
import pytest
from PIL import Image
from xray_simulator.validation import attenuation_from_intensity, evaluate_pair
from xray_simulator.validation.__main__ import main, run, summarize


@pytest.fixture
def image():
    y, x = np.mgrid[:48, :48]
    return 0.1 + 0.5 * np.exp(-((x - 19) ** 2 + (y - 23) ** 2) / 80) + 0.1 * np.sin(x / 3) ** 2


def compare(reference, rendered, *, domain="attenuation", mask=None, **kwargs):
    return evaluate_pair(
        reference, rendered, np.ones(reference.shape, dtype=bool) if mask is None else mask,
        domain=domain, data_range=1.0, histogram_range=(0.0, 1.0), **kwargs,
    )


def test_identical_pair(image):
    result = compare(image, image)
    for mode in ("as_configured", "shape_only"):
        scores = result[mode]
        assert scores["ncc"] == pytest.approx(1)
        assert scores["gradient_ncc"] == pytest.approx(1)
        assert scores["ssim"] == pytest.approx(1)
        assert scores["rmse"] == pytest.approx(0, abs=1e-14)
        assert scores["wasserstein"] == pytest.approx(0, abs=1e-14)
    assert result["affine_fit"]["gain"] == pytest.approx(1)
    assert sum(row["pixels"] for row in result["radial_profiles"]) == image.size


def test_known_affine_fit_preserves_unfitted_error(image):
    result = compare(image, 0.5 * image + 0.15)
    assert result["affine_fit"] == pytest.approx({"gain": 2.0, "bias": -0.3})
    assert result["as_configured"]["rmse"] > 0.02
    assert result["shape_only"]["rmse"] < 1e-14
    assert result["shape_only"]["ssim"] == pytest.approx(1)


def test_fit_cannot_absorb_spatial_misregistration(image):
    aligned = compare(image, image)
    shifted = compare(image, np.roll(image, 7, axis=1))
    assert shifted["as_configured"]["gradient_ncc"] < aligned["as_configured"]["gradient_ncc"] - 0.2
    assert shifted["shape_only"]["rmse"] > 0.03


def test_polarity_is_not_silently_corrected(image):
    result = compare(image, 1 - image)
    assert result["as_configured"]["ncc"] == pytest.approx(-1)
    assert result["as_configured"]["gradient_ncc"] == pytest.approx(1)
    assert result["affine_fit"]["gain"] == 0
    assert result["warnings"]
    assert result["shape_only"]["rmse"] > 0


def test_display_does_not_get_attenuation_fit(image):
    result = compare(image, image * 0.5, domain="display")
    assert result["shape_only"] is None
    assert result["affine_fit"] is None
    assert result["radial_profiles"] is None
    assert result["units"] == "normalized grayscale"


def test_proxy_is_not_labeled_physical_attenuation(image):
    assert compare(image, image, domain="attenuation_proxy")["units"] == "arbitrary proxy units"


def test_constant_images_have_undefined_correlations():
    constant = np.ones((32, 32))
    result = compare(constant, constant)
    assert result["as_configured"]["ncc"] is None
    assert result["as_configured"]["gradient_ncc"] is None
    assert result["shape_only"] is None
    assert result["as_configured"]["rmse"] == 0
    json.dumps(result, allow_nan=False)


def test_mask_excludes_sobel_and_ssim_neighborhoods(image):
    mask = np.zeros_like(image, dtype=bool)
    mask[7:41, 7:41] = True
    changed = image.copy()
    changed[~mask] = 1000
    result = compare(image, changed, mask=mask)
    assert result["as_configured"]["gradient_ncc"] == pytest.approx(1)
    assert result["as_configured"]["ssim"] == pytest.approx(1, abs=1e-7)
    assert result["gradient_mask_pixels"] == 32 * 32
    assert result["ssim_mask_pixels"] == 24 * 24


def test_mi_known_binary_joint_distribution():
    y, x = np.indices((32, 32))
    first, independent = (x % 2).astype(float), (y % 2).astype(float)
    assert compare(first, first, bins=2)["as_configured"]["mi_nats"] == pytest.approx(np.log(2))
    assert compare(first, independent, bins=2)["as_configured"]["mi_nats"] == pytest.approx(0)


def test_histogram_keeps_and_reports_out_of_range_mass(image):
    result = compare(image, image + 2)
    assert result["as_configured"]["histogram_clipped_rendered_fraction"] == 1
    assert result["as_configured"]["mi_nats"] == pytest.approx(0)
    assert result["as_configured"]["wasserstein"] == pytest.approx(2)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_rejected(image, bad):
    altered = image.copy()
    altered[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        compare(image, altered)


def test_shape_and_mask_errors(image):
    with pytest.raises(ValueError, match="identical shapes"):
        compare(image, image[:-1])
    with pytest.raises(ValueError, match="0/1"):
        compare(image, image, mask=np.full_like(image, 255))
    mask = np.zeros_like(image)
    mask[12:20, 12:20] = 1
    with pytest.raises(ValueError, match="11x11"):
        compare(image, image, mask=mask)


@pytest.mark.parametrize("data_range,histogram_range,domain", [
    (0, (0, 1), "attenuation"), (np.inf, (0, 1), "attenuation"),
    (1, (1, 1), "attenuation"), (1, (0, np.nan), "attenuation"),
    (2, (0, 1), "display"), (1, (0, 2), "display"), (1, (0, 1), "unknown"),
])
def test_invalid_scales(image, data_range, histogram_range, domain):
    with pytest.raises(ValueError):
        evaluate_pair(image, image, np.ones_like(image), domain=domain,
                      data_range=data_range, histogram_range=histogram_range)


def test_display_outside_unit_interval_rejected(image):
    with pytest.raises(ValueError, match="scaled"):
        compare(image, image + 2, domain="display")


def test_beer_lambert_inverse_and_explicit_floor(image):
    assert np.allclose(attenuation_from_intensity(3 * np.exp(-image), 3), image)
    assert attenuation_from_intensity(np.zeros((2, 2)), 1, transmission_floor=1e-5)[0, 0] == pytest.approx(-np.log(1e-5))
    assert attenuation_from_intensity(np.full((2, 2), 2), 1)[0, 0] == pytest.approx(-np.log(2))
    with pytest.raises(ValueError):
        attenuation_from_intensity(-image, 1)
    with pytest.raises(ValueError):
        attenuation_from_intensity(image, 0)


def make_manifest(tmp_path, image, *, domain="attenuation"):
    np.save(tmp_path / "reference.npy", image)
    np.save(tmp_path / "rendered.npy", image * 0.5 + 0.15)
    np.save(tmp_path / "mask.npy", np.ones_like(image, dtype=bool))
    np.save(tmp_path / "misposed.npy", np.roll(image, 7, axis=1))
    manifest = {"schema_version": 1, "domain": domain, "data_range": 1,
                "histogram_range": [0, 1], "pairs": [
                    {"dataset": "analytic-fixture", "subject": "phantom", "view": "AP",
                     "reference": "reference.npy", "rendered": "rendered.npy", "mask": "mask.npy",
                     "misposed": "misposed.npy"}]}
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_cli_preview_report_and_overwrite_protection(tmp_path, image):
    path, _ = make_manifest(tmp_path, image)
    output = tmp_path / "results/report.json"
    assert main([str(path), "--output", str(output), "--dryrun"]) == 0
    assert not output.parent.exists()
    assert main([str(path), "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert len(report["pairs"]) == 1
    assert report["pairs"][0]["gradient_ncc_gap"] > 0.2
    assert len(report["pairs"][0]["input_sha256"]["reference"]) == 64
    assert report["subjects"][0]["as_configured"]["ncc"]["ci95"] is None
    before = output.read_bytes()
    with pytest.raises(SystemExit) as exc:
        main([str(path), "--output", str(output)])
    assert exc.value.code == 2
    assert output.read_bytes() == before


def test_duplicate_pair_rejected(tmp_path, image):
    path, manifest = make_manifest(tmp_path, image)
    manifest["pairs"].append(manifest["pairs"][0])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Duplicate"):
        run(path)


def test_cli_failure_leaves_no_report(tmp_path, image):
    path, _ = make_manifest(tmp_path, image)
    np.save(tmp_path / "rendered.npy", image[:-1])
    output = tmp_path / "results/report.json"
    with pytest.raises(SystemExit):
        main([str(path), "--output", str(output)])
    assert not output.exists()


def test_png_scale_and_overlay_rejection(tmp_path, image):
    path, manifest = make_manifest(tmp_path, image, domain="display")
    pixels = np.rint(image * 255).astype(np.uint8)
    rgb = np.repeat(pixels[:, :, None], 3, axis=2)
    png = tmp_path / "reference.png"
    Image.fromarray(rgb).save(png)
    manifest["pairs"][0]["reference"] = png.name
    path.write_text(json.dumps(manifest))
    assert run(path)["pairs"][0]["scores"]["shape_only"] is None
    rgb[0, 0, 1] = 255
    Image.fromarray(rgb).save(png)
    with pytest.raises(ValueError, match="Colored"):
        run(path)
    manifest["domain"] = "attenuation"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="PNG cannot establish"):
        run(path)


def test_bootstrap_groups_whole_acquisitions_and_keeps_subjects_separate():
    records = []
    for subject, value in (("one", 0.2), ("two", 0.8)):
        for acquisition in ("cine-A", "cine-B"):
            for _ in range(3):
                records.append({"dataset": "fixture", "subject": subject, "acquisition": acquisition,
                                "scores": {"as_configured": {"ncc": value}, "shape_only": None}})
    first = summarize(records, seed=9, bootstrap_samples=100)
    assert first == summarize(records, seed=9, bootstrap_samples=100)
    assert len(first) == 2
    for item, expected in zip(first, (0.2, 0.8)):
        assert item["views"] == 6
        assert item["as_configured"]["ncc"]["ci95"] == pytest.approx([expected, expected])
    for record in records:
        record["acquisition"] = "one-correlated-cine"
    assert summarize(records)[0]["as_configured"]["ncc"]["ci95"] is None


def test_palette_png_cannot_hide_colored_overlays(tmp_path, image):
    path, manifest = make_manifest(tmp_path, image, domain="display")
    png = tmp_path / "palette.png"
    palette_image = Image.fromarray(np.zeros(image.shape, dtype=np.uint8), mode="P")
    palette_image.putpalette([255, 0, 0] + [0] * 765)
    palette_image.save(png)
    manifest["pairs"][0]["reference"] = png.name
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Colored"):
        run(path)


def test_correlated_frames_do_not_narrow_acquisition_interval():
    # Per-frame resampling would almost certainly return [0.8, 0.8]. Whole
    # acquisitions can both be drawn from A, so the interval must include 0.2.
    records = [
        {"dataset": "fixture", "subject": "one", "acquisition": acquisition,
         "scores": {"as_configured": {"ncc": value}, "shape_only": None}}
        for acquisition, count, value in (("A", 50, 0.2), ("B", 150, 0.8))
        for _ in range(count)
    ]
    result = summarize(records, seed=3, bootstrap_samples=500)
    assert result[0]["as_configured"]["ncc"]["ci95"] == pytest.approx([0.2, 0.8])
