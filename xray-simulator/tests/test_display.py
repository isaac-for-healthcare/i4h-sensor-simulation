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

"""Tests for the intensity-to-display mapping, presets and deprecated flags."""

from __future__ import annotations

import numpy as np
import pytest
from xray_simulator.config import (
    DISPLAY_PRESETS,
    DisplaySettings,
    SimulatorConfig,
    XrayPhysics,
    resolve_display_settings,
)
from xray_simulator.display import apply_display, calibrate_display, transmission
from xray_simulator.rendering.realism import RealismConfig, apply_realism

I0 = 2.0


@pytest.fixture
def intensity() -> np.ndarray:
    """Intensity image: air at i0, soft tissue mid, dense structure heavily attenuated."""
    img = np.full((8, 8), I0, dtype=np.float32)
    img[2:6, 2:6] = I0 * 0.5
    img[3:5, 3:5] = I0 * 0.05
    return img


def _brightest(img: np.ndarray) -> float:
    return float(img.max())


class TestTransmission:
    def test_air_transmits_fully(self, intensity):
        assert transmission(intensity, I0).max() == pytest.approx(1.0)

    def test_dense_region_transmits_least(self, intensity):
        t = transmission(intensity, I0)
        assert t[4, 4] == pytest.approx(0.05, abs=1e-6)

    def test_rejects_non_positive_i0(self, intensity):
        with pytest.raises(ValueError, match="i0 must be positive"):
            transmission(intensity, 0.0)


class TestPolarity:
    def test_fluoro_draws_dense_structures_dark(self, intensity):
        img = apply_display(intensity, DisplaySettings(polarity="fluoro"), i0=I0)
        assert img[4, 4] < img[2, 2] < img[0, 0]

    def test_diagnostic_draws_dense_structures_bright(self, intensity):
        img = apply_display(intensity, DisplaySettings(polarity="diagnostic"), i0=I0)
        assert img[4, 4] > img[2, 2] > img[0, 0]

    def test_polarities_are_complementary(self, intensity):
        fluoro = apply_display(intensity, DisplaySettings(polarity="fluoro"), i0=I0)
        diagnostic = apply_display(intensity, DisplaySettings(polarity="diagnostic"), i0=I0)
        np.testing.assert_allclose(fluoro + diagnostic, 1.0, atol=1e-6)

    def test_default_is_fluoro(self, intensity):
        assert DisplaySettings().polarity == "fluoro"
        default = apply_display(intensity, i0=I0)
        explicit = apply_display(intensity, DisplaySettings(polarity="fluoro"), i0=I0)
        np.testing.assert_array_equal(default, explicit)


class TestLogScaling:
    """The default mapping: linear in the line integral, like a real detector chain."""

    @staticmethod
    def _from_line_integral(values: np.ndarray) -> np.ndarray:
        return (I0 * np.exp(-values)).astype(np.float32)

    def test_default_scaling_is_log(self):
        assert DisplaySettings().scaling == "log"

    def test_air_stays_at_the_bright_end(self):
        img = apply_display(self._from_line_integral(np.zeros((4, 4))), i0=I0)
        np.testing.assert_allclose(img, 1.0, atol=1e-6)

    def test_display_is_linear_in_path_length(self):
        settings = DisplaySettings(scaling="log", log_window=(0.0, 4.0))
        thin, thick = 1.0, 2.0
        img = apply_display(self._from_line_integral(np.array([[thin, thick]])), settings, i0=I0)
        # Fluoro polarity, so display = 1 - integral / window width.
        assert img[0, 0] == pytest.approx(1.0 - thin / 4.0, abs=1e-5)
        assert img[0, 1] == pytest.approx(1.0 - thick / 4.0, abs=1e-5)

    def test_saturates_beyond_the_window(self):
        settings = DisplaySettings(scaling="log", log_window=(0.0, 3.0))
        img = apply_display(self._from_line_integral(np.array([[3.0, 9.0]])), settings, i0=I0)
        assert img[0, 0] == pytest.approx(0.0, abs=1e-5)
        assert img[0, 1] == pytest.approx(0.0, abs=1e-5)

    def test_window_offset_places_the_bright_end(self):
        settings = DisplaySettings(scaling="log", log_window=(2.0, 4.0))
        img = apply_display(self._from_line_integral(np.array([[2.0, 3.0, 4.0]])), settings, i0=I0)
        assert img[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert img[0, 1] == pytest.approx(0.5, abs=1e-5)
        assert img[0, 2] == pytest.approx(0.0, abs=1e-5)

    def test_narrower_window_gives_more_soft_tissue_contrast(self):
        integrals = self._from_line_integral(np.array([[1.0, 1.4]]))
        wide = apply_display(integrals, DisplaySettings(log_window=(0.0, 6.0)), i0=I0)
        tight = apply_display(integrals, DisplaySettings(log_window=(1.0, 4.0)), i0=I0)
        assert abs(tight[0, 0] - tight[0, 1]) > abs(wide[0, 0] - wide[0, 1])

    def test_torso_scale_attenuation_is_visible_not_crushed(self):
        """A whole-torso line integral must land in usable mid-grey, not near black."""
        torso = self._from_line_integral(np.full((4, 4), 2.4))
        log_img = apply_display(torso, DisplaySettings.preset("fluoro"), i0=I0)
        transmission_img = apply_display(torso, DisplaySettings.preset("transmission"), i0=I0)
        assert 0.3 < float(log_img.mean()) < 0.8
        assert float(transmission_img.mean()) < 0.2


class TestScaling:
    def test_transmission_scaling_is_physical(self, intensity):
        img = apply_display(intensity, DisplaySettings(scaling="transmission"), i0=I0)
        np.testing.assert_allclose(img, intensity / I0, atol=1e-6)

    def test_window_stretches_the_requested_interval(self, intensity):
        settings = DisplaySettings(scaling="window", window=(0.0, 0.5))
        img = apply_display(intensity, settings, i0=I0)
        # Transmission 0.5 sits at the top of the window, so it saturates.
        assert img[2, 2] == pytest.approx(1.0)
        assert img[4, 4] == pytest.approx(0.1, abs=1e-6)

    def test_per_frame_always_spans_the_full_range(self, intensity):
        img = apply_display(intensity, DisplaySettings(scaling="per_frame"), i0=I0)
        assert img.min() == pytest.approx(0.0)
        assert img.max() == pytest.approx(1.0, abs=1e-6)

    def test_output_stays_in_unit_range(self, intensity):
        for name, settings in DISPLAY_PRESETS.items():
            img = apply_display(intensity, settings, i0=I0)
            assert img.min() >= 0.0 and img.max() <= 1.0, name


class TestGamma:
    def test_gamma_above_one_brightens_midtones(self, intensity):
        linear = apply_display(intensity, DisplaySettings(scaling="transmission"), i0=I0)
        brightened = apply_display(
            intensity,
            DisplaySettings(scaling="transmission", gamma=2.0),
            i0=I0,
        )
        mid = intensity < I0  # anything attenuated at all
        assert (brightened[mid] >= linear[mid]).all()
        assert brightened[2, 2] > linear[2, 2]

    def test_extremes_are_fixed_points(self, intensity):
        img = apply_display(intensity, DisplaySettings(gamma=2.5), i0=I0)
        assert img[0, 0] == pytest.approx(1.0)


class TestValidation:
    @pytest.mark.parametrize("polarity", ["inverted", "", None])
    def test_bad_polarity_rejected(self, polarity):
        with pytest.raises(ValueError, match="polarity must be"):
            DisplaySettings(polarity=polarity)

    def test_bad_scaling_rejected(self):
        with pytest.raises(ValueError, match="scaling must be"):
            DisplaySettings(scaling="minmax")

    @pytest.mark.parametrize("log_window", [(4.0, 4.0), (5.0, 2.0), (-1.0, 3.0)])
    def test_bad_log_window_rejected(self, log_window):
        with pytest.raises(ValueError, match="log_window must satisfy"):
            DisplaySettings(log_window=log_window)

    @pytest.mark.parametrize("window", [(0.5, 0.5), (0.6, 0.4), (-0.1, 0.5), (0.0, 1.5)])
    def test_bad_window_rejected(self, window):
        with pytest.raises(ValueError, match="window must satisfy"):
            DisplaySettings(scaling="window", window=window)

    def test_non_positive_gamma_rejected(self):
        with pytest.raises(ValueError, match="gamma must be positive"):
            DisplaySettings(gamma=0.0)

    def test_non_2d_image_rejected(self):
        with pytest.raises(ValueError, match="Expected a 2D image"):
            apply_display(np.zeros((4, 4, 4), dtype=np.float32))


class TestPresets:
    def test_fluoro_preset_keeps_dense_dark_and_scaling_stable(self):
        preset = DisplaySettings.preset("fluoro")
        assert (preset.polarity, preset.scaling) == ("fluoro", "log")

    def test_diagnostic_preset_inverts(self):
        assert DisplaySettings.preset("diagnostic").polarity == "diagnostic"

    def test_fluoro_contrast_preset_uses_a_narrower_window(self, intensity):
        preset = DisplaySettings.preset("fluoro_contrast")
        fluoro = DisplaySettings.preset("fluoro")
        assert preset.polarity == "fluoro"
        width = preset.log_window[1] - preset.log_window[0]
        assert width < fluoro.log_window[1] - fluoro.log_window[0]

    def test_transmission_preset_is_the_literal_ratio(self, intensity):
        img = apply_display(intensity, DisplaySettings.preset("transmission"), i0=I0)
        np.testing.assert_allclose(img, intensity / I0, atol=1e-6)

    def test_legacy_preset_reproduces_normalize_then_invert(self, intensity):
        legacy = apply_display(intensity, DisplaySettings.preset("legacy"), i0=I0)

        # What the renderer used to do internally with normalize=True, invert=True.
        expected = intensity.astype(np.float32)
        expected = (expected - expected.min()) / (expected.max() - expected.min() + 1e-8)
        expected = 1.0 - expected

        np.testing.assert_allclose(legacy, expected, atol=1e-6)

    def test_unknown_preset_lists_the_valid_names(self):
        with pytest.raises(ValueError, match="unknown display preset"):
            DisplaySettings.preset("cine")

    def test_config_helper_selects_the_preset(self):
        config = SimulatorConfig.for_appearance("diagnostic")
        assert config.display.polarity == "diagnostic"


class TestCalibration:
    """Fitting the log window once, then holding it fixed."""

    @staticmethod
    def _torso(peak: float = 3.2) -> np.ndarray:
        line_integral = np.linspace(0.4, peak, 64, dtype=np.float32).reshape(8, 8)
        return (I0 * np.exp(-line_integral)).astype(np.float32)

    def test_calibrated_window_covers_the_frame(self):
        settings = calibrate_display(self._torso(), i0=I0)
        low, high = settings.log_window
        assert 0.0 <= low < high
        assert high == pytest.approx(3.2, abs=0.2)

    def test_calibration_uses_the_full_display_range(self):
        frame = self._torso()
        settings = calibrate_display(frame, i0=I0)
        img = apply_display(frame, settings, i0=I0)
        assert img.min() == pytest.approx(0.0, abs=0.02)
        assert img.max() == pytest.approx(1.0, abs=0.02)

    def test_calibration_beats_the_uncalibrated_default_on_contrast(self):
        frame = self._torso()
        default = apply_display(frame, DisplaySettings.preset("fluoro"), i0=I0)
        calibrated = apply_display(frame, calibrate_display(frame, i0=I0), i0=I0)
        assert calibrated.std() > default.std()

    def test_polarity_and_gamma_are_preserved(self):
        source = DisplaySettings(polarity="diagnostic", gamma=1.4)
        settings = calibrate_display(self._torso(), source, i0=I0)
        assert (settings.polarity, settings.gamma) == ("diagnostic", 1.4)
        assert settings.scaling == "log"

    def test_calibrated_window_is_reused_across_frames(self):
        """Fitting once keeps unchanged anatomy at the same display value."""
        first = self._torso()
        settings = calibrate_display(first, i0=I0)
        second = first.copy()
        second[0, :] = I0 * np.exp(-8.0)  # a dense instrument enters the field

        assert apply_display(second, settings, i0=I0)[4, 4] == pytest.approx(
            apply_display(first, settings, i0=I0)[4, 4], abs=1e-6
        )

    def test_percentiles_trim_a_small_dense_object(self):
        frame = self._torso()
        frame[0, 0] = I0 * np.exp(-20.0)  # catheter tip, a single pixel
        trimmed = calibrate_display(frame, i0=I0, percentiles=(1.0, 99.0))
        untrimmed = calibrate_display(frame, i0=I0, percentiles=(0.0, 100.0))
        assert trimmed.log_window[1] < untrimmed.log_window[1]

    @pytest.mark.parametrize("percentiles", [(50.0, 50.0), (90.0, 10.0), (-1.0, 99.0), (0.0, 101.0)])
    def test_bad_percentiles_rejected(self, percentiles):
        with pytest.raises(ValueError, match="percentiles must satisfy"):
            calibrate_display(self._torso(), i0=I0, percentiles=percentiles)

    def test_flat_frame_reports_a_useful_error(self):
        flat = np.full((8, 8), I0, dtype=np.float32)
        with pytest.raises(ValueError, match="no attenuation range"):
            calibrate_display(flat, i0=I0)


class _FakeRenderer:
    """Stands in for the Slang renderer so display wiring can be tested without a GPU."""

    def __init__(self, intensity: np.ndarray):
        self._intensity = intensity
        self.calls: list[tuple] = []

    def render(self, rotation, translation):
        self.calls.append((tuple(rotation), tuple(translation)))
        return self._intensity


def _simulator_with(intensity: np.ndarray, display: DisplaySettings | None = None):
    """Build a simulator around a fake renderer, bypassing GPU initialisation."""
    from xray_simulator.simulator import xray_simulator

    sim = xray_simulator.__new__(xray_simulator)
    sim._config = SimulatorConfig(display=display or DisplaySettings())
    sim._display = sim._config.display
    sim._renderer = _FakeRenderer(intensity)
    sim._frame_warning_issued = True
    sim._frame_times = []
    return sim


def _simulator_keeping_intensity(intensity: np.ndarray):
    """As above, but configured to retain the pre-display intensity on each frame."""
    sim = _simulator_with(intensity)
    sim._config = sim._config.with_output(keep_intensity=True)
    return sim


class TestSimulatorCalibration:
    @staticmethod
    def _frame() -> np.ndarray:
        line_integral = np.linspace(0.5, 3.0, 64, dtype=np.float32).reshape(8, 8)
        return (np.exp(-line_integral)).astype(np.float32)

    def test_calibration_updates_the_settings_in_effect(self):
        from xray_simulator.simulator import xray_simulator

        sim = _simulator_with(self._frame())
        before = sim.display
        after = xray_simulator.calibrate_display(sim)

        assert after is sim.display
        assert after.log_window != before.log_window
        assert after.scaling == "log"

    def test_calibration_defaults_to_an_ap_view(self):
        from xray_simulator.simulator import Pose, xray_simulator

        sim = _simulator_with(self._frame())
        xray_simulator.calibrate_display(sim)
        assert sim._renderer.calls == [(Pose.ap().rotation, Pose.ap().translation)]

    def test_calibration_preserves_requested_polarity(self):
        from xray_simulator.simulator import xray_simulator

        sim = _simulator_with(self._frame(), DisplaySettings(polarity="diagnostic"))
        assert xray_simulator.calibrate_display(sim).polarity == "diagnostic"


class TestTwoModes:
    """Fluoroscopy and X-ray are the same render with opposite polarity."""

    @staticmethod
    def _frame() -> np.ndarray:
        line_integral = np.linspace(0.5, 3.0, 64, dtype=np.float32).reshape(8, 8)
        return np.exp(-line_integral).astype(np.float32)

    def test_xray_preset_is_available_by_that_name(self):
        assert DisplaySettings.preset("xray").polarity == "diagnostic"

    def test_xray_and_diagnostic_are_the_same_settings(self):
        assert DisplaySettings.preset("xray") == DisplaySettings.preset("diagnostic")

    def test_the_two_modes_differ_only_in_polarity(self):
        fluoro = DisplaySettings.preset("fluoro")
        xray = DisplaySettings.preset("xray")
        assert (fluoro.scaling, fluoro.log_window) == (xray.scaling, xray.log_window)
        assert fluoro.polarity != xray.polarity

    def test_set_appearance_switches_mode(self):
        from xray_simulator.simulator import xray_simulator

        sim = _simulator_with(self._frame())
        assert xray_simulator.set_appearance(sim, "xray").polarity == "diagnostic"
        assert xray_simulator.set_appearance(sim, "fluoro").polarity == "fluoro"

    def test_set_appearance_accepts_explicit_settings(self):
        from xray_simulator.simulator import xray_simulator

        sim = _simulator_with(self._frame())
        wanted = DisplaySettings(polarity="diagnostic", scaling="transmission")
        assert xray_simulator.set_appearance(sim, wanted) == wanted

    def test_set_appearance_rejects_unknown_names(self):
        from xray_simulator.simulator import xray_simulator

        sim = _simulator_with(self._frame())
        with pytest.raises(ValueError, match="unknown display preset"):
            xray_simulator.set_appearance(sim, "cinefluorography")

    def test_set_polarity_keeps_calibration(self):
        from xray_simulator.simulator import xray_simulator

        sim = _simulator_with(self._frame())
        calibrated = xray_simulator.calibrate_display(sim)
        switched = xray_simulator.set_polarity(sim, "xray")

        assert switched.polarity == "diagnostic"
        assert switched.log_window == calibrated.log_window

    def test_set_appearance_discards_calibration_as_documented(self):
        from xray_simulator.simulator import xray_simulator

        sim = _simulator_with(self._frame())
        calibrated = xray_simulator.calibrate_display(sim)
        assert xray_simulator.set_appearance(sim, "xray").log_window != calibrated.log_window


class TestFrameRemapping:
    @staticmethod
    def _frame_with_intensity():
        from xray_simulator.simulator import Frame, Pose

        line_integral = np.linspace(0.5, 3.0, 64, dtype=np.float32).reshape(8, 8)
        intensity = np.exp(-line_integral).astype(np.float32)
        settings = DisplaySettings.preset("fluoro")
        return Frame(
            image=apply_display(intensity, settings, i0=1.0),
            pose=Pose.ap(),
            intensity=intensity,
            i0=1.0,
        )

    def test_remapping_produces_the_other_mode(self):
        frame = self._frame_with_intensity()
        as_xray = frame.with_appearance("xray")
        np.testing.assert_allclose(as_xray.image, 1.0 - frame.image, atol=1e-6)

    def test_remapping_leaves_the_original_untouched(self):
        frame = self._frame_with_intensity()
        original = frame.image.copy()
        frame.with_appearance("xray")
        np.testing.assert_array_equal(frame.image, original)

    def test_remapping_preserves_pose_and_intensity(self):
        frame = self._frame_with_intensity()
        remapped = frame.with_appearance("xray")
        assert remapped.pose == frame.pose
        np.testing.assert_array_equal(remapped.intensity, frame.intensity)

    def test_remapping_without_intensity_explains_how_to_enable_it(self):
        from xray_simulator.simulator import Frame, Pose

        frame = Frame(image=np.zeros((4, 4), dtype=np.float32), pose=Pose.ap())
        with pytest.raises(ValueError, match="keep_intensity=True"):
            frame.with_appearance("xray")

    def test_output_setting_controls_retention(self):
        assert SimulatorConfig().output.keep_intensity is False
        assert SimulatorConfig().with_output(keep_intensity=True).output.keep_intensity is True

    def test_rendering_retains_intensity_when_asked(self):
        from xray_simulator.simulator import Pose, xray_simulator

        rendered = self._frame_with_intensity().intensity
        sim = _simulator_keeping_intensity(rendered)
        frame = xray_simulator.render_frame(sim, pose=Pose.ap())

        np.testing.assert_array_equal(frame.intensity, rendered)
        np.testing.assert_allclose(frame.with_appearance("xray").image, 1.0 - frame.image, atol=1e-6)

    def test_display_mapping_does_not_touch_the_retained_intensity(self):
        from xray_simulator.simulator import Pose, xray_simulator

        rendered = self._frame_with_intensity().intensity
        sim = _simulator_keeping_intensity(rendered.copy())
        frame = xray_simulator.render_frame(sim, pose=Pose.ap())

        np.testing.assert_array_equal(frame.intensity, rendered)

    def test_rendering_omits_intensity_by_default(self):
        from xray_simulator.simulator import Pose, xray_simulator

        sim = _simulator_with(self._frame_with_intensity().intensity)
        assert xray_simulator.render_frame(sim, pose=Pose.ap()).intensity is None


class TestStabilityAcrossFrames:
    @staticmethod
    def _frames() -> list[np.ndarray]:
        """Two frames of the same anatomy, the second with a dense instrument added."""
        base = np.full((8, 8), I0, dtype=np.float32)
        base[2:6, 2:6] = I0 * 0.5
        with_instrument = base.copy()
        with_instrument[0, :] = I0 * 0.01
        return [base, with_instrument]

    def test_physical_scaling_keeps_anatomy_stable(self):
        first, second = (
            apply_display(f, DisplaySettings.preset("fluoro"), i0=I0) for f in self._frames()
        )
        # The soft-tissue block is unchanged between frames, so its display value must be too.
        assert second[3, 3] == pytest.approx(first[3, 3], abs=1e-6)

    def test_per_frame_scaling_shifts_unchanged_anatomy(self):
        first, second = (
            apply_display(f, DisplaySettings.preset("legacy"), i0=I0) for f in self._frames()
        )
        assert abs(second[3, 3] - first[3, 3]) > 0.1


class TestRealismStaysInIntensityUnits:
    def test_default_does_not_rescale(self, intensity):
        out = apply_realism(intensity, RealismConfig(gain=1.0))
        np.testing.assert_allclose(out, intensity, atol=1e-6)

    def test_noise_preserves_intensity_scale(self, intensity):
        cfg = RealismConfig(poisson_photons=500.0, seed=3)
        out = apply_realism(intensity, cfg)
        assert _brightest(out) == pytest.approx(I0, rel=0.2)

    def test_noisy_frames_share_a_display_mapping(self, intensity):
        settings = DisplaySettings.preset("fluoro")
        frames = [
            apply_display(apply_realism(intensity, RealismConfig(poisson_photons=800.0, seed=s)), settings, i0=I0)
            for s in (0, 1)
        ]
        # Noise perturbs pixels, but the mapping is fixed, so the mean cannot drift far.
        assert abs(float(frames[0].mean()) - float(frames[1].mean())) < 0.02


class TestDeprecatedFlags:
    def test_unset_flags_leave_display_untouched(self):
        display = DisplaySettings.preset("fluoro")
        assert resolve_display_settings(XrayPhysics(), display) is display

    def test_invert_true_maps_to_diagnostic(self):
        with pytest.warns(DeprecationWarning, match="invert=True"):
            resolved = resolve_display_settings(XrayPhysics(invert=True), DisplaySettings())
        assert resolved.polarity == "diagnostic"

    def test_invert_false_maps_to_fluoro(self):
        with pytest.warns(DeprecationWarning):
            resolved = resolve_display_settings(XrayPhysics(invert=False), DisplaySettings(polarity="diagnostic"))
        assert resolved.polarity == "fluoro"

    def test_normalize_true_maps_to_per_frame(self):
        with pytest.warns(DeprecationWarning, match="normalize=True"):
            resolved = resolve_display_settings(XrayPhysics(normalize=True), DisplaySettings())
        assert resolved.scaling == "per_frame"

    def test_old_defaults_reproduce_the_legacy_preset(self):
        with pytest.warns(DeprecationWarning):
            resolved = resolve_display_settings(
                XrayPhysics(normalize=True, invert=True),
                DisplaySettings(),
            )
        assert resolved == DisplaySettings.preset("legacy")

    def test_warning_names_the_replacement(self):
        with pytest.warns(DeprecationWarning, match="SimulatorConfig.display"):
            resolve_display_settings(XrayPhysics(invert=True), DisplaySettings())
