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

"""Configuration dataclasses for the Xray Simulator.

This module defines all configuration objects used by the simulator API.
All configs are frozen dataclasses with sensible defaults for clinical C-arm geometry.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class CarmGeometry:
    """C-arm geometry configuration.

    Defines the physical geometry of the X-ray imaging system including
    source-detector distances and detector specifications.

    Attributes:
        source_to_detector_mm: Distance from X-ray source to detector plane (mm).
            Also called SDD (Source-to-Detector Distance). Typical range: 990–1250 mm.
        source_to_isocenter_mm: Distance from X-ray source to isocenter (mm).
            Also called SID (Source-to-Isocenter Distance). The isocenter is the
            rotation center, typically at patient table level. Typical range: 495–780 mm.
        detector_width_px: Detector width in pixels.
        detector_height_px: Detector height in pixels.
        pixel_spacing_mm: Physical size of each detector pixel (mm).

    Vendor-Specific Configuration:
        Different C-arm vendors (GE, Siemens, Philips, Ziehm, etc.) have distinct
        geometry specifications. To configure for a specific vendor system:

        1. **SDD and SID**: These define the X-ray cone geometry and magnification.
           - Mobile C-arms typically have SDD ~1000mm with SID ~500mm (2x magnification)
           - Fixed interventional systems have larger SDD ~1200-1250mm with SID ~750-780mm

        2. **Detector size**: Varies by detector type and generation.
           - Image intensifiers (legacy): 1024×1024 pixels
           - Flat-panel detectors: 1536×1536 to 2480×1920 pixels

        3. **Pixel spacing**: Determines field of view (FOV = pixels × spacing).
           - Typical range: 0.15–0.20 mm/pixel
           - FOV = detector_px × pixel_spacing_mm (e.g., 1024 × 0.194 ≈ 200mm = 20cm)

        4. **Magnification**: Calculated as SDD / SID.
           - At isocenter, objects appear magnified by this factor on the detector.
           - Typical range: 1.5x to 2.0x

        Example vendor-specific configurations::

            # GE OEC 9900 (mobile C-arm, 12" II equivalent)
            geometry = CarmGeometry(
                source_to_detector_mm=1020.0,
                source_to_isocenter_mm=510.0,
                detector_width_px=1024,
                detector_height_px=1024,
                pixel_spacing_mm=0.194,  # ~20cm FOV
            )

            # Siemens Artis zee (fixed biplane angiography)
            geometry = CarmGeometry(
                source_to_detector_mm=1250.0,
                source_to_isocenter_mm=780.0,
                detector_width_px=2480,
                detector_height_px=1920,
                pixel_spacing_mm=0.154,  # ~38×30cm FOV
            )

            # Philips Azurion 7 (premium interventional)
            geometry = CarmGeometry(
                source_to_detector_mm=1240.0,
                source_to_isocenter_mm=780.0,
                detector_width_px=2480,
                detector_height_px=1920,
                pixel_spacing_mm=0.154,
            )

        Reference specifications (approximate values, verify with actual system docs):

        | Vendor/Model              | SDD (mm) | SID (mm) | Detector   | Pixel (mm) |
        |---------------------------|----------|----------|------------|------------|
        | GE OEC 9900               | 1020     | 510      | 1024×1024  | 0.194      |
        | GE OEC Elite CFD          | 1150     | 575      | 1920×1920  | 0.154      |
        | GE Innova IGS 540         | 1200     | 750      | 2048×2048  | 0.200      |
        | Siemens Arcadis Avantic   | 1000     | 500      | 1024×1024  | 0.195      |
        | Siemens Cios Alpha        | 1100     | 550      | 1536×1536  | 0.178      |
        | Siemens Artis zee         | 1250     | 780      | 2480×1920  | 0.154      |
        | Philips BV Pulsera        | 990      | 495      | 1024×1024  | 0.200      |
        | Philips Azurion 7         | 1240     | 780      | 2480×1920  | 0.154      |
        | Ziehm Vision RFD 3D       | 1000     | 500      | 1024×1024  | 0.194      |
    """

    source_to_detector_mm: float = 1020.0
    source_to_isocenter_mm: float = 510.0
    detector_width_px: int = 512
    detector_height_px: int = 512
    pixel_spacing_mm: float = 0.5

    @property
    def detector_size_mm(self) -> tuple[float, float]:
        """Physical detector size (width, height) in mm."""
        return (
            self.detector_width_px * self.pixel_spacing_mm,
            self.detector_height_px * self.pixel_spacing_mm,
        )


@dataclass(frozen=True)
class XrayPhysics:
    """X-ray physics configuration.

    Controls the physical parameters of the X-ray simulation: ray-marching resolution and
    incident beam intensity. How the resulting intensity is turned into displayable pixels
    is a separate concern, configured by :class:`DisplaySettings`.

    Attributes:
        step_mm: Ray-marching step size (mm). Smaller = more accurate but slower.
        i0: Unattenuated X-ray intensity (incident beam intensity).
        normalize: Deprecated. Leave as None and use ``DisplaySettings.scaling``; setting
            it True selects ``"per_frame"`` scaling and False selects ``"transmission"``.
        invert: Deprecated. Leave as None and use ``DisplaySettings.polarity``; setting it
            True selects ``"diagnostic"`` polarity and False selects ``"fluoro"``.
    """

    step_mm: float = 0.5
    i0: float = 1.0
    normalize: bool | None = None
    invert: bool | None = None


@dataclass(frozen=True)
class DisplaySettings:
    """How rendered intensity becomes a displayable image.

    The renderer output is intensity ``I = i0 · exp(-∫μ ds)``, where dense anatomy carries
    less signal than air. Two independent choices turn that into pixels, and keeping them
    separate avoids the ambiguity of a single "normalize and invert" flag pair.

    Polarity picks which way round the greys go, and is the difference between the two modes
    the simulator supports:

    * ``"fluoro"`` keeps the physical ordering, so dense structures (bone, contrast,
      instruments) are **dark** on a bright background, as on a live fluoroscopy monitor.
      This is the default because the simulator targets fluoroscopy. Preset: ``"fluoro"``.
    * ``"diagnostic"`` inverts, giving the radiograph look where dense structures are
      **bright**, which is what people mean by an X-ray image. Use it when comparing against
      diagnostic X-rays or DRR literature. Presets: ``"xray"``, or ``"diagnostic"``.

    Both modes come from the same render, so the choice costs nothing and can be changed
    afterwards; see :meth:`xray_simulator.xray_simulator.set_appearance` to switch modes and
    :attr:`xray_simulator.Frame.with_appearance` to re-map a frame already rendered.

    Scaling picks how intensity maps onto ``[0, 1]``, and all the frame-independent modes
    keep brightness stable across a cine run and comparable between runs:

    * ``"log"`` maps the line integral ``∫μ ds = -ln(I / i0)`` linearly across
      ``log_window``. This is the default because a real detector chain is logarithmic:
      transmission through a torso is only a few percent, so displaying transmission
      directly leaves almost everything crushed against black. Equivalent to the optical
      density of film, and linear in path length through tissue. Since the useful range
      depends on patient size and μ scaling, calibrate the window once per volume with
      :func:`xray_simulator.display.calibrate_display` and then keep it fixed — that is what
      makes the result both physical and stable, in the same spirit as a system's automatic
      exposure setting a working range and holding it.
    * ``"transmission"`` divides by ``i0``, so 1.0 means the beam passed unattenuated. The
      most literal physical quantity, useful for dosimetry-style analysis or thin phantoms,
      but usually too dark to look at for a whole patient.
    * ``"window"`` linearly stretches the transmission interval in ``window``, the
      equivalent of a contrast control.
    * ``"per_frame"`` rescales each frame by its own min and max. Maximises contrast per
      frame, but brightness then depends on the field of view, so a moving C-arm or an
      instrument entering the frame makes the sequence flicker. Only for one-off stills.

    Attributes:
        polarity: ``"fluoro"`` (dense dark) or ``"diagnostic"`` (dense bright).
        scaling: ``"log"``, ``"transmission"``, ``"window"`` or ``"per_frame"`` (see above).
        log_window: Line-integral interval ``(low, high)`` mapped across the display range
            when scaling is ``"log"``. ``low`` becomes the bright end, ``high`` the dark end.
            A narrower window means more contrast and earlier clipping.
        window: Transmission interval ``(low, high)`` stretched when scaling is
            ``"window"``. Must satisfy ``0 <= low < high <= 1``.
        gamma: Display gamma applied last. 1.0 leaves values linear; values above 1.0
            brighten mid-greys.

    Example:
        >>> DisplaySettings.preset("fluoro").polarity
        'fluoro'
        >>> DisplaySettings.preset("xray").polarity
        'diagnostic'
        >>> DisplaySettings.preset("xray") == DisplaySettings.preset("diagnostic")
        True
        >>> DisplaySettings(scaling="window", window=(0.0, 0.6)).window
        (0.0, 0.6)
    """

    polarity: Literal["fluoro", "diagnostic"] = "fluoro"
    scaling: Literal["log", "transmission", "window", "per_frame"] = "log"
    log_window: tuple[float, float] = (0.0, 6.0)
    window: tuple[float, float] = (0.0, 1.0)
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if self.polarity not in ("fluoro", "diagnostic"):
            raise ValueError(f"polarity must be 'fluoro' or 'diagnostic', got {self.polarity!r}")
        if self.scaling not in ("log", "transmission", "window", "per_frame"):
            raise ValueError(
                f"scaling must be 'log', 'transmission', 'window' or 'per_frame', "
                f"got {self.scaling!r}"
            )

        low, high = (float(v) for v in self.window)
        if not 0.0 <= low < high <= 1.0:
            raise ValueError(f"window must satisfy 0 <= low < high <= 1, got {self.window}")
        object.__setattr__(self, "window", (low, high))

        log_low, log_high = (float(v) for v in self.log_window)
        if not 0.0 <= log_low < log_high:
            raise ValueError(f"log_window must satisfy 0 <= low < high, got {self.log_window}")
        object.__setattr__(self, "log_window", (log_low, log_high))

        if self.gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")

    @classmethod
    def preset(cls, name: str) -> "DisplaySettings":
        """Return a named appearance preset.

        Args:
            name: One of the keys of :data:`DISPLAY_PRESETS`:

                * ``"fluoro"`` — live fluoroscopy: dense dark, logarithmic scaling over a
                  wide line-integral window that suits most body regions uncalibrated.
                * ``"fluoro_contrast"`` — same polarity over a narrower window, so soft
                  tissue and contrast media separate more strongly at the cost of clipping
                  dense structures.
                * ``"xray"`` — radiograph look: dense bright, logarithmic scaling.
                  ``"diagnostic"`` is the same settings under its radiological name.
                * ``"transmission"`` — raw ``I / i0``, dense dark and unstretched, for
                  analysis rather than viewing.
                * ``"legacy"`` — reproduces the pre-preset default (per-frame rescale then
                  invert). Kept so existing outputs can be regenerated; it flickers across
                  frames, so prefer ``"fluoro"`` for cine.

        Returns:
            The preset settings.

        Raises:
            ValueError: If the name is not a known preset.
        """
        try:
            return DISPLAY_PRESETS[name]
        except KeyError:
            raise ValueError(
                f"unknown display preset {name!r}; choose from {sorted(DISPLAY_PRESETS)}"
            ) from None


_FLUORO = DisplaySettings(polarity="fluoro", scaling="log", log_window=(0.0, 6.0))
_XRAY = DisplaySettings(polarity="diagnostic", scaling="log", log_window=(0.0, 6.0))

DISPLAY_PRESETS: dict[str, DisplaySettings] = {
    # The two modes, differing only in polarity.
    "fluoro": _FLUORO,
    "xray": _XRAY,
    "diagnostic": _XRAY,  # radiological name for the same look
    # Variants.
    "fluoro_contrast": DisplaySettings(polarity="fluoro", scaling="log", log_window=(1.0, 4.0)),
    "transmission": DisplaySettings(polarity="fluoro", scaling="transmission"),
    "legacy": DisplaySettings(polarity="diagnostic", scaling="per_frame"),
}


@dataclass(frozen=True)
class HuToMuMapping:
    """Piecewise-linear Hounsfield Unit to linear attenuation coefficient mapping.

    The curve is defined by control points and is clamped outside the outermost pair,
    which is the same construction as window/level control on a radiology viewer. With
    the default two control points ``P0 = (hu_min, mu_min)`` and ``P1 = (hu_max, mu_max)``:

        μ(HU) = mu_min                                             HU ≤ hu_min
        μ(HU) = mu_min + slope × (HU − hu_min)                     hu_min < HU < hu_max
        μ(HU) = mu_max                                             HU ≥ hu_max

        slope = (mu_max − mu_min) / (hu_max − hu_min)

    The two degrees of freedom that matter when tuning image appearance are the position
    of the ramp on the HU axis (level) and its steepness (window). ``from_window_level``,
    ``with_window_level``, ``shifted`` and ``scaled`` address those directly, so an
    interactive control (e.g. horizontal drag = level, vertical drag = contrast) maps onto
    them without touching the raw endpoints.

    Passing more than two ``control_points`` gives independent slopes per HU band (air,
    soft tissue, contrast, bone). That is supported, but every extra knot makes the curve
    harder to tune by hand, so prefer starting from the two-point ramp.

    Attributes:
        hu_min: HU value where attenuation starts to rise. Default: -1000 (air).
        hu_max: HU value where attenuation saturates. Default: 3000 (dense bone).
        mu_min: μ value (mm⁻¹) applied at and below hu_min. Default: 0.0.
        mu_max: μ value (mm⁻¹) applied at and above hu_max. Default: 0.02.
        control_points: Optional ``((HU, μ), ...)`` knots with strictly increasing HU.
            When given, these define the curve and the four scalar fields above are
            overwritten with the first and last knot so that they stay consistent.

    Suggested starting points for visual tuning. These are display-style settings to be
    adjusted against reference images, not calibrations against a kVp energy spectrum:

    | Emphasis                           | window_center | window_width |
    |------------------------------------|---------------|--------------|
    | Whole HU range (default)           | 1000          | 4000         |
    | Soft tissue and contrasted vessels | 100           | 800          |
    | Bone and dense structures          | 800           | 2000         |

    Example:
        >>> mapping = HuToMuMapping.from_window_level(window_center=100.0, window_width=800.0)
        >>> (mapping.hu_min, mapping.hu_max)
        (-300.0, 500.0)
        >>> mapping.scaled(1.5).mu_max  # steeper ramp, more contrast
        0.03
        >>> mapping.shifted(200.0).window_center  # slide the ramp along the HU axis
        300.0
    """

    hu_min: float = -1000.0
    hu_max: float = 3000.0
    mu_min: float = 0.0
    mu_max: float = 0.02
    control_points: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if self.control_points is None:
            self._validate_ramp(self.hu_min, self.hu_max, self.mu_min, self.mu_max)
            return

        points = tuple((float(hu), float(mu)) for hu, mu in self.control_points)
        if len(points) < 2:
            raise ValueError(f"control_points needs at least 2 knots, got {len(points)}")
        for (hu_lo, _), (hu_hi, _) in zip(points, points[1:]):
            if hu_hi <= hu_lo:
                raise ValueError(f"control_points must have strictly increasing HU, got {points}")
        for _, mu in points:
            if mu < 0.0:
                raise ValueError(f"control_points must have non-negative μ, got {points}")

        object.__setattr__(self, "control_points", points)
        object.__setattr__(self, "hu_min", points[0][0])
        object.__setattr__(self, "mu_min", points[0][1])
        object.__setattr__(self, "hu_max", points[-1][0])
        object.__setattr__(self, "mu_max", points[-1][1])

    @staticmethod
    def _validate_ramp(hu_min: float, hu_max: float, mu_min: float, mu_max: float) -> None:
        if hu_max <= hu_min:
            raise ValueError(f"hu_max must be greater than hu_min, got hu_min={hu_min}, hu_max={hu_max}")
        if mu_min < 0.0 or mu_max < 0.0:
            raise ValueError(f"μ values must be non-negative, got mu_min={mu_min}, mu_max={mu_max}")

    @property
    def points(self) -> tuple[tuple[float, float], ...]:
        """Return the control points ``((HU, μ), ...)`` defining the curve."""
        if self.control_points is not None:
            return self.control_points
        return ((self.hu_min, self.mu_min), (self.hu_max, self.mu_max))

    @property
    def hu_knots(self) -> tuple[float, ...]:
        """Return the HU coordinates of the control points, strictly increasing."""
        return tuple(hu for hu, _ in self.points)

    @property
    def mu_knots(self) -> tuple[float, ...]:
        """Return the μ coordinates (mm⁻¹) of the control points."""
        return tuple(mu for _, mu in self.points)

    @property
    def window_width(self) -> float:
        """Return the HU span covered by the ramp (hu_max − hu_min)."""
        return self.hu_max - self.hu_min

    @property
    def window_center(self) -> float:
        """Return the HU value at the middle of the ramp."""
        return 0.5 * (self.hu_min + self.hu_max)

    @property
    def slope(self) -> float:
        """Return the end-to-end gradient (μ per HU) across the ramp."""
        return (self.mu_max - self.mu_min) / self.window_width

    @classmethod
    def from_window_level(
        cls,
        window_center: float,
        window_width: float,
        mu_max: float = 0.02,
        mu_min: float = 0.0,
    ) -> "HuToMuMapping":
        """Build a two-point ramp from window/level parameters.

        Args:
            window_center: HU value at the middle of the ramp (level).
            window_width: HU span covered by the ramp (window). Must be positive.
            mu_max: μ value (mm⁻¹) at and above the top of the ramp.
            mu_min: μ value (mm⁻¹) at and below the bottom of the ramp.

        Returns:
            Mapping whose ramp spans ``[center − width/2, center + width/2]``.

        Raises:
            ValueError: If window_width is not positive.
        """
        if window_width <= 0.0:
            raise ValueError(f"window_width must be positive, got {window_width}")
        half = 0.5 * window_width
        return cls(
            hu_min=window_center - half,
            hu_max=window_center + half,
            mu_min=mu_min,
            mu_max=mu_max,
        )

    def with_window_level(
        self,
        window_center: float | None = None,
        window_width: float | None = None,
    ) -> "HuToMuMapping":
        """Return a mapping repositioned and rescaled on the HU axis.

        The μ values and the relative spacing of any intermediate control points are
        preserved; only the HU axis is remapped onto the requested window.

        Args:
            window_center: New level. Defaults to the current window_center.
            window_width: New window. Defaults to the current window_width.

        Returns:
            New mapping with the requested window/level.

        Raises:
            ValueError: If window_width is not positive.
        """
        center = self.window_center if window_center is None else window_center
        width = self.window_width if window_width is None else window_width
        if width <= 0.0:
            raise ValueError(f"window_width must be positive, got {width}")

        scale = width / self.window_width
        new_lo = center - 0.5 * width
        return self._rebuilt((new_lo + (hu - self.hu_min) * scale, mu) for hu, mu in self.points)

    def shifted(self, delta_hu: float) -> "HuToMuMapping":
        """Return a mapping translated along the HU axis (interactive level control).

        Args:
            delta_hu: HU offset added to every control point.

        Returns:
            New mapping with the same shape at a different HU position.
        """
        return self._rebuilt((hu + delta_hu, mu) for hu, mu in self.points)

    def scaled(self, factor: float) -> "HuToMuMapping":
        """Return a mapping with all μ values scaled (interactive contrast control).

        Args:
            factor: Non-negative multiplier applied to every μ control value.

        Returns:
            New mapping with a steeper (factor > 1) or flatter (factor < 1) ramp.

        Raises:
            ValueError: If factor is negative.
        """
        if factor < 0.0:
            raise ValueError(f"factor must be non-negative, got {factor}")
        return self._rebuilt((hu, mu * factor) for hu, mu in self.points)

    def _rebuilt(self, points: Iterable[tuple[float, float]]) -> "HuToMuMapping":
        knots = tuple(points)
        if len(knots) == 2:
            (hu_lo, mu_lo), (hu_hi, mu_hi) = knots
            return HuToMuMapping(hu_min=hu_lo, hu_max=hu_hi, mu_min=mu_lo, mu_max=mu_hi)
        return HuToMuMapping(control_points=knots)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON/YAML-friendly dictionary."""
        d: dict[str, Any] = {
            "hu_min": self.hu_min,
            "hu_max": self.hu_max,
            "mu_min": self.mu_min,
            "mu_max": self.mu_max,
        }
        if self.control_points is not None:
            d["control_points"] = [list(p) for p in self.control_points]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HuToMuMapping":
        """Create a mapping from a config dictionary.

        Accepts either explicit ``control_points``, a ``window_center``/``window_width``
        pair, or the ``hu_min``/``hu_max``/``mu_min``/``mu_max`` endpoints.

        Args:
            d: Config dictionary, e.g. parsed from YAML.

        Returns:
            Mapping built from the recognised keys.
        """
        if d.get("control_points"):
            return cls(control_points=tuple((float(hu), float(mu)) for hu, mu in d["control_points"]))

        defaults = cls()
        mu_min = float(d.get("mu_min", defaults.mu_min))
        mu_max = float(d.get("mu_max", defaults.mu_max))

        if "window_center" in d and "window_width" in d:
            return cls.from_window_level(
                window_center=float(d["window_center"]),
                window_width=float(d["window_width"]),
                mu_min=mu_min,
                mu_max=mu_max,
            )

        return cls(
            hu_min=float(d.get("hu_min", defaults.hu_min)),
            hu_max=float(d.get("hu_max", defaults.hu_max)),
            mu_min=mu_min,
            mu_max=mu_max,
        )


@dataclass(frozen=True)
class RealismSettings:
    """Realism post-processing settings.

    Controls noise, blur, and other post-processing effects to make
    simulated fluoroscopy more realistic.

    These run on rendered intensity, before the display mapping, so photon statistics are
    applied where they physically belong. Scaling the result into a display range is
    :class:`DisplaySettings`' job; realism no longer rescales frames itself, which is what
    used to make noisy cine sequences flicker.

    Attributes:
        enabled: If True, apply realism effects.
        gain: Linear intensity scaling factor.
        bias: Intensity offset (added after gain).
        poisson_photons: Photon count for Poisson noise (0 = disabled).
        gaussian_sigma: Gaussian noise sigma (0 = disabled).
        blur_sigma_px: Gaussian blur sigma in pixels (0 = disabled).
        seed: Random seed for reproducibility (None = random).
    """

    enabled: bool = False
    gain: float = 1.0
    bias: float = 0.0
    poisson_photons: float = 0.0
    gaussian_sigma: float = 0.0
    blur_sigma_px: float = 0.0
    seed: int | None = 0


@dataclass(frozen=True)
class OutputSettings:
    """Output configuration for rendered frames.

    Attributes:
        save_to_disk: If True, save rendered frames to disk.
        output_dir: Directory for saved frames (used if save_to_disk=True).
        format: Output format for saved frames.
        keep_in_gpu: If True, keep frames in GPU memory (for streaming).
        keep_intensity: If True, each Frame also carries the pre-display intensity, so the
            same render can be shown in either mode via ``Frame.with_appearance``. Costs one
            extra float32 image per frame, so it is off by default for long cine runs.
    """

    save_to_disk: bool = False
    output_dir: str | Path | None = None
    format: Literal["png", "npy", "npz"] = "png"
    keep_in_gpu: bool = False
    keep_intensity: bool = False


@dataclass(frozen=True)
class MetricsSettings:
    """Performance metrics collection settings.

    Attributes:
        enabled: If True, collect performance metrics.
        track_fps: Track frames per second.
        track_gpu_usage: Track GPU memory and utilization.
        track_jitter: Track frame timing jitter.
    """

    enabled: bool = False
    track_fps: bool = True
    track_gpu_usage: bool = True
    track_jitter: bool = True


@dataclass(frozen=True)
class PreprocessingSettings:
    """CT preprocessing settings.

    Attributes:
        hu_clip_min: Minimum HU value for clipping.
        hu_clip_max: Maximum HU value for clipping.
        clip_hu: If True, clip HU values to [hu_clip_min, hu_clip_max].
        hu_to_mu: HU to μ mapping configuration.
    """

    hu_clip_min: float = -1024.0
    hu_clip_max: float = 3071.0
    clip_hu: bool = True
    hu_to_mu: HuToMuMapping = field(default_factory=HuToMuMapping)


@dataclass(frozen=True)
class SimulatorConfig:
    """Unified configuration for the Xray Simulator.

    This is the main configuration object that bundles all settings for
    the simulator. Pass this to xray_simulator to control rendering behavior.

    Attributes:
        geometry: C-arm geometry settings.
        physics: X-ray physics settings.
        display: Polarity and scaling used to turn intensity into displayable pixels.
        realism: Realism post-processing settings.
        output: Output settings for rendered frames.
        metrics: Performance metrics settings.
        backend: Rendering backend (currently only "slang" is supported).

    Example:
        >>> config = SimulatorConfig(
        ...     geometry=CarmGeometry(detector_width_px=1024, detector_height_px=1024),
        ...     realism=RealismSettings(enabled=True, gaussian_sigma=0.01),
        ...     output=OutputSettings(save_to_disk=True, output_dir="/tmp/frames"),
        ... )
        >>> SimulatorConfig.for_appearance("fluoro").display.polarity
        'fluoro'
        >>> SimulatorConfig.for_appearance("xray").display.polarity
        'diagnostic'
    """

    geometry: CarmGeometry = field(default_factory=CarmGeometry)
    physics: XrayPhysics = field(default_factory=XrayPhysics)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    realism: RealismSettings = field(default_factory=RealismSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    metrics: MetricsSettings = field(default_factory=MetricsSettings)
    backend: Literal["slang"] = "slang"

    @classmethod
    def for_appearance(cls, preset: str, **kwargs) -> "SimulatorConfig":
        """Return a config using a named display preset.

        Args:
            preset: Preset name, see :meth:`DisplaySettings.preset`.
            **kwargs: Any other SimulatorConfig field.

        Returns:
            Config whose display settings come from the preset.
        """
        return cls(display=DisplaySettings.preset(preset), **kwargs)

    def _replace(self, **kwargs) -> "SimulatorConfig":
        fields = {
            "geometry": self.geometry,
            "physics": self.physics,
            "display": self.display,
            "realism": self.realism,
            "output": self.output,
            "metrics": self.metrics,
            "backend": self.backend,
        }
        return SimulatorConfig(**{**fields, **kwargs})

    def with_geometry(self, **kwargs) -> "SimulatorConfig":
        """Return a new config with updated geometry settings."""
        return self._replace(geometry=CarmGeometry(**{**self.geometry.__dict__, **kwargs}))

    def with_display(self, **kwargs) -> "SimulatorConfig":
        """Return a new config with updated display settings."""
        return self._replace(display=DisplaySettings(**{**self.display.__dict__, **kwargs}))

    def with_realism(self, **kwargs) -> "SimulatorConfig":
        """Return a new config with updated realism settings."""
        return self._replace(realism=RealismSettings(**{**self.realism.__dict__, **kwargs}))

    def with_output(self, **kwargs) -> "SimulatorConfig":
        """Return a new config with updated output settings."""
        return self._replace(output=OutputSettings(**{**self.output.__dict__, **kwargs}))


def resolve_display_settings(
    physics: XrayPhysics,
    display: DisplaySettings,
) -> DisplaySettings:
    """Fold deprecated ``physics.normalize`` / ``physics.invert`` into display settings.

    Args:
        physics: Physics settings, possibly carrying the deprecated flags.
        display: Display settings to start from.

    Returns:
        ``display`` unchanged when neither deprecated flag is set, otherwise a copy with
        polarity and scaling derived from them.

    Warns:
        DeprecationWarning: If either deprecated flag is set.
    """
    if physics.normalize is None and physics.invert is None:
        return display

    updates: dict[str, str] = {}
    used: list[str] = []

    if physics.invert is not None:
        updates["polarity"] = "diagnostic" if physics.invert else "fluoro"
        used.append(f"invert={physics.invert}")
    if physics.normalize is not None:
        updates["scaling"] = "per_frame" if physics.normalize else "transmission"
        used.append(f"normalize={physics.normalize}")

    resolved = DisplaySettings(**{**display.__dict__, **updates})
    warnings.warn(
        f"XrayPhysics {' and '.join(used)} is deprecated; it was applied as "
        f"DisplaySettings(polarity={resolved.polarity!r}, scaling={resolved.scaling!r}). "
        f"Set SimulatorConfig.display instead, e.g. "
        f"SimulatorConfig.for_appearance('fluoro') for live fluoroscopy or "
        f"'legacy' for the previous default appearance.",
        DeprecationWarning,
        stacklevel=3,
    )
    return resolved
