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

"""Tests for the piecewise-linear HU → μ transfer function."""

import numpy as np
import pytest
from xray_simulator import HuToMuMapping, hu_to_mu, hu_to_mu_curve


class TestTwoPointRamp:
    """The default two-point ramp: clamped below hu_min and above hu_max."""

    def test_clamped_below_ramp(self):
        mapping = HuToMuMapping(hu_min=-200.0, hu_max=1000.0, mu_min=0.0, mu_max=0.02)
        mu = hu_to_mu(np.array([-5000.0, -1000.0, -200.0]), mapping)
        np.testing.assert_allclose(mu, 0.0)

    def test_clamped_above_ramp(self):
        mapping = HuToMuMapping(hu_min=-200.0, hu_max=1000.0, mu_min=0.0, mu_max=0.02)
        mu = hu_to_mu(np.array([1000.0, 3000.0, 10000.0]), mapping)
        np.testing.assert_allclose(mu, 0.02, rtol=1e-6)

    def test_midpoint_is_half_of_mu_max(self):
        mapping = HuToMuMapping(hu_min=-200.0, hu_max=1000.0, mu_min=0.0, mu_max=0.02)
        mu = hu_to_mu(np.array([mapping.window_center]), mapping)
        np.testing.assert_allclose(mu, 0.01, rtol=1e-6)

    def test_slope_matches_endpoints(self):
        mapping = HuToMuMapping(hu_min=0.0, hu_max=1000.0, mu_min=0.0, mu_max=0.02)
        assert mapping.slope == pytest.approx(2e-5)
        mu = hu_to_mu(np.array([250.0, 500.0, 750.0]), mapping)
        np.testing.assert_allclose(mu, [0.005, 0.01, 0.015], rtol=1e-6)

    def test_nonzero_mu_min_offsets_the_ramp(self):
        mapping = HuToMuMapping(hu_min=0.0, hu_max=1000.0, mu_min=0.004, mu_max=0.02)
        mu = hu_to_mu(np.array([-500.0, 0.0, 500.0, 1000.0]), mapping)
        np.testing.assert_allclose(mu, [0.004, 0.004, 0.012, 0.02], rtol=1e-6)

    def test_monotonic_non_decreasing(self):
        mapping = HuToMuMapping()
        _, mu = hu_to_mu_curve(mapping, num=1024)
        assert np.all(np.diff(mu) >= 0.0)

    def test_output_is_float32_and_shape_preserving(self):
        hu = np.zeros((3, 4, 5), dtype=np.int16)
        mu = hu_to_mu(hu, HuToMuMapping())
        assert mu.dtype == np.float32
        assert mu.shape == (3, 4, 5)

    def test_default_mapping_used_when_omitted(self):
        hu = np.array([0.0, 1000.0])
        np.testing.assert_allclose(hu_to_mu(hu), hu_to_mu(hu, HuToMuMapping()))


class TestWindowLevel:
    """Window/level parameterization of the ramp."""

    def test_from_window_level_sets_endpoints(self):
        mapping = HuToMuMapping.from_window_level(window_center=100.0, window_width=800.0)
        assert (mapping.hu_min, mapping.hu_max) == (-300.0, 500.0)

    def test_window_center_and_width_round_trip(self):
        mapping = HuToMuMapping.from_window_level(window_center=250.0, window_width=1500.0)
        assert mapping.window_center == pytest.approx(250.0)
        assert mapping.window_width == pytest.approx(1500.0)

    def test_with_window_level_keeps_mu_range(self):
        mapping = HuToMuMapping(hu_min=-1000.0, hu_max=3000.0, mu_max=0.02)
        narrowed = mapping.with_window_level(window_center=0.0, window_width=500.0)
        assert (narrowed.hu_min, narrowed.hu_max) == (-250.0, 250.0)
        assert narrowed.mu_max == pytest.approx(0.02)

    def test_narrower_window_is_steeper(self):
        wide = HuToMuMapping.from_window_level(window_center=0.0, window_width=4000.0)
        narrow = wide.with_window_level(window_width=400.0)
        assert narrow.slope > wide.slope

    def test_shifted_moves_ramp_without_changing_width(self):
        mapping = HuToMuMapping.from_window_level(window_center=0.0, window_width=1000.0)
        moved = mapping.shifted(300.0)
        assert moved.window_center == pytest.approx(300.0)
        assert moved.window_width == pytest.approx(mapping.window_width)

    def test_scaled_changes_gradient_only(self):
        mapping = HuToMuMapping(hu_min=0.0, hu_max=1000.0, mu_max=0.02)
        steeper = mapping.scaled(1.5)
        assert steeper.mu_max == pytest.approx(0.03)
        assert steeper.window_width == pytest.approx(mapping.window_width)
        assert steeper.slope == pytest.approx(1.5 * mapping.slope)

    def test_scaled_preserves_intermediate_knots(self):
        mapping = HuToMuMapping(control_points=((-1000.0, 0.0), (0.0, 0.004), (1000.0, 0.02)))
        steeper = mapping.scaled(2.0)
        assert steeper.mu_knots == pytest.approx((0.0, 0.008, 0.04))
        assert steeper.hu_knots == mapping.hu_knots

    def test_with_window_level_rescales_intermediate_knots(self):
        mapping = HuToMuMapping(control_points=((0.0, 0.0), (500.0, 0.005), (1000.0, 0.02)))
        rescaled = mapping.with_window_level(window_center=0.0, window_width=2000.0)
        assert rescaled.hu_knots == pytest.approx((-1000.0, 0.0, 1000.0))
        assert rescaled.mu_knots == pytest.approx(mapping.mu_knots)


class TestMultiPointCurve:
    """Extra control points give independent slopes per HU band."""

    def test_knot_values_are_reproduced_exactly(self):
        points = ((-1000.0, 0.0), (0.0, 0.004), (300.0, 0.012), (1500.0, 0.02))
        mapping = HuToMuMapping(control_points=points)
        mu = hu_to_mu(np.array([hu for hu, _ in points]), mapping)
        np.testing.assert_allclose(mu, [mu_ for _, mu_ in points], rtol=1e-6)

    def test_bands_have_different_slopes(self):
        mapping = HuToMuMapping(control_points=((0.0, 0.0), (100.0, 0.010), (1000.0, 0.012)))
        soft = hu_to_mu(np.array([50.0]), mapping)[0]
        bone = hu_to_mu(np.array([550.0]), mapping)[0]
        assert soft == pytest.approx(0.005, rel=1e-5)
        assert bone == pytest.approx(0.011, rel=1e-5)

    def test_endpoints_synced_to_scalar_fields(self):
        mapping = HuToMuMapping(control_points=((-500.0, 0.001), (0.0, 0.004), (900.0, 0.03)))
        assert (mapping.hu_min, mapping.hu_max) == (-500.0, 900.0)
        assert (mapping.mu_min, mapping.mu_max) == (0.001, 0.03)

    def test_clamped_outside_outer_knots(self):
        mapping = HuToMuMapping(control_points=((0.0, 0.002), (100.0, 0.01), (1000.0, 0.02)))
        mu = hu_to_mu(np.array([-4000.0, 6000.0]), mapping)
        np.testing.assert_allclose(mu, [0.002, 0.02], rtol=1e-6)

    def test_list_of_lists_normalized_to_tuples(self):
        mapping = HuToMuMapping(control_points=[[-1000, 0], [1000, 0.02]])
        assert mapping.control_points == ((-1000.0, 0.0), (1000.0, 0.02))


class TestValidation:
    """Invalid curves are rejected at construction time."""

    def test_inverted_ramp_rejected(self):
        with pytest.raises(ValueError, match="hu_max must be greater than hu_min"):
            HuToMuMapping(hu_min=1000.0, hu_max=-1000.0)

    def test_degenerate_ramp_rejected(self):
        with pytest.raises(ValueError, match="hu_max must be greater than hu_min"):
            HuToMuMapping(hu_min=0.0, hu_max=0.0)

    def test_negative_mu_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            HuToMuMapping(mu_max=-0.02)

    def test_single_control_point_rejected(self):
        with pytest.raises(ValueError, match="at least 2 knots"):
            HuToMuMapping(control_points=((0.0, 0.01),))

    def test_non_increasing_control_points_rejected(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            HuToMuMapping(control_points=((1000.0, 0.0), (0.0, 0.02)))

    def test_duplicate_hu_knots_rejected(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            HuToMuMapping(control_points=((0.0, 0.0), (0.0, 0.02)))

    def test_negative_mu_knot_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            HuToMuMapping(control_points=((0.0, 0.0), (1000.0, -0.01)))

    def test_non_positive_window_width_rejected(self):
        with pytest.raises(ValueError, match="window_width must be positive"):
            HuToMuMapping.from_window_level(window_center=0.0, window_width=0.0)

    def test_negative_scale_factor_rejected(self):
        with pytest.raises(ValueError, match="factor must be non-negative"):
            HuToMuMapping().scaled(-1.0)


class TestSerialization:
    """Round-trip through the dict form used by YAML configs and volume metadata."""

    def test_ramp_round_trip(self):
        mapping = HuToMuMapping(hu_min=-500.0, hu_max=1500.0, mu_min=0.001, mu_max=0.03)
        assert HuToMuMapping.from_dict(mapping.to_dict()) == mapping

    def test_control_points_round_trip(self):
        mapping = HuToMuMapping(control_points=((-1000.0, 0.0), (0.0, 0.004), (1000.0, 0.02)))
        assert HuToMuMapping.from_dict(mapping.to_dict()) == mapping

    def test_ramp_dict_omits_control_points(self):
        assert "control_points" not in HuToMuMapping().to_dict()

    def test_from_dict_accepts_window_level(self):
        mapping = HuToMuMapping.from_dict({"window_center": 100.0, "window_width": 800.0, "mu_max": 0.02})
        assert (mapping.hu_min, mapping.hu_max, mapping.mu_max) == (-300.0, 500.0, 0.02)

    def test_from_dict_falls_back_to_defaults(self):
        assert HuToMuMapping.from_dict({}) == HuToMuMapping()


class TestCurveSampling:
    """Curve sampling used for plotting and inspection."""

    def test_sample_count_and_dtype(self):
        hu, mu = hu_to_mu_curve(HuToMuMapping(), num=64)
        assert hu.shape == mu.shape == (64,)
        assert mu.dtype == np.float32

    def test_default_range_shows_clamped_tails(self):
        mapping = HuToMuMapping(hu_min=0.0, hu_max=1000.0, mu_max=0.02)
        hu, mu = hu_to_mu_curve(mapping, num=256)
        assert hu[0] < mapping.hu_min and hu[-1] > mapping.hu_max
        assert mu[0] == pytest.approx(0.0)
        assert mu[-1] == pytest.approx(0.02, rel=1e-5)

    def test_explicit_range_respected(self):
        hu, _ = hu_to_mu_curve(HuToMuMapping(), hu_range=(-100.0, 100.0), num=16)
        assert hu[0] == pytest.approx(-100.0)
        assert hu[-1] == pytest.approx(100.0)
