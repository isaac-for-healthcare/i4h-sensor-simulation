#!/usr/bin/env python3
"""Qt-based interactive X-ray simulation viewer.

This GUI renders fluoroscopy frames from a preprocessed CT volume and exposes
controls for geometry, pose, and simple display/noise effects.

Supported load targets:
- a cached preprocessed volume directory containing ``mu_volume.npy`` and ``metadata.json``
- a ``.nii`` or ``.nii.gz`` volume
- a 3D ``.npy`` volume
- a 2D image file, which is converted into a simple pseudo-volume for demos
"""

from __future__ import annotations

import os
import math
import sys
from pathlib import Path

import numpy as np

from fluorosim import (
    CarmGeometry,
    XrayPhysics,
    FluoroSimulator,
    OutputSettings,
    PreprocessedVolume,
    RealismSettings,
    SimulatorConfig,
    VolumeMetadata,
    VolumePreprocessor,
    HuToMuMapping,
)
from fluorosim.config import PreprocessingSettings, XrayPhysics

try:
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtGui import QDoubleValidator, QImage, QIntValidator, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSlider,
        QVBoxLayout,
        QWidget,
    )
    QT_API = "PySide6"
except ImportError:
    try:
        from PyQt6.QtCore import QTimer, Qt
        from PyQt6.QtGui import QDoubleValidator, QImage, QIntValidator, QPixmap
        from PyQt6.QtWidgets import (
            QApplication,
            QCheckBox,
            QFileDialog,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QScrollArea,
            QSlider,
            QVBoxLayout,
            QWidget,
        )
        QT_API = "PyQt6"
    except ImportError:
        try:
            from PyQt5.QtCore import QTimer, Qt
            from PyQt5.QtGui import QDoubleValidator, QImage, QIntValidator, QPixmap
            from PyQt5.QtWidgets import (
                QApplication,
                QCheckBox,
                QFileDialog,
                QGridLayout,
                QGroupBox,
                QHBoxLayout,
                QLabel,
                QLineEdit,
                QMainWindow,
                QMessageBox,
                QPushButton,
                QScrollArea,
                QSlider,
                QVBoxLayout,
                QWidget,
            )
            QT_API = "PyQt5"
        except ImportError as exc:
            raise SystemExit(
                "Qt bindings not found. Install PySide6, PyQt6, or PyQt5 to run "
                "xray_interactive_view_qt.py."
            ) from exc

# Get local script directory path
_SCRIPT_DIR = Path("~/workspace/i4h-sensor-simulation/fluoro-simulator/examples").expanduser().resolve().parent.parent
# Output directory
OUTPUT_DIR = Path(os.environ.get("FLUOROSIM_OUTPUT_DIR", str(_SCRIPT_DIR / "output"))).expanduser()# Cache directory for preprocessed volume
CACHE_DIR = Path(os.environ.get("FLUOROSIM_CACHE_DIR", str(OUTPUT_DIR / "cache"))).expanduser()
# Create the cache directory if it doesn't exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_GEOMETRY = CarmGeometry(
    detector_width_px=3072,
    detector_height_px=3072,
    pixel_spacing_mm=0.15,
    source_to_detector_mm=2220.0,
    source_to_isocenter_mm=2180.0,
)

DEFAULT_PHYSICS = XrayPhysics(
        step_mm=0.5,                    # Ray-march step size (smaller = more accurate)
        i0=1.0,                         # Unattenuated intensity
        normalize=True,                 # Normalize output to [0, 1]
        invert=True,                    # Clinical convention: bone=white, air=black
)

DEFAULT_HU_TO_MU = HuToMuMapping(
    hu_min = -1000.0,                   # Defaults to -1000
    hu_max = 3000.0,                    # Defaults to 3000 (dense bone)
    mu_min = 0.0,                       # Defaults to 0.0
    mu_max = 0.02,                      # Defaults to 0.02 (approximate mu for bone at typical X-ray energies
)

DEFAULT_PREPROCESSING_SETTINGS = PreprocessingSettings(
    hu_clip_min = -1024.0,              # Defaults to -1024 (air)
    hu_clip_max = 3071.0,               # Defaults to 3071 (bone)
    clip_hu = True,
    hu_to_mu = DEFAULT_HU_TO_MU,
)

try:
    ORIENTATION_HORIZONTAL = Qt.Orientation.Horizontal
    ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
    ALIGN_RIGHT_VCENTER = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
    KEEP_ASPECT = Qt.AspectRatioMode.KeepAspectRatio
    SMOOTH_TRANSFORM = Qt.TransformationMode.SmoothTransformation
    IMAGE_FORMAT_RGB888 = QImage.Format.Format_RGB888
    IMAGE_FORMAT_GRAY8 = QImage.Format.Format_Grayscale8
except AttributeError:
    ORIENTATION_HORIZONTAL = Qt.Horizontal
    ALIGN_CENTER = Qt.AlignCenter
    ALIGN_RIGHT_VCENTER = Qt.AlignRight | Qt.AlignVCenter
    KEEP_ASPECT = Qt.KeepAspectRatio
    SMOOTH_TRANSFORM = Qt.SmoothTransformation
    IMAGE_FORMAT_RGB888 = QImage.Format_RGB888
    IMAGE_FORMAT_GRAY8 = QImage.Format_Grayscale8


def create_synthetic_volume() -> PreprocessedVolume:
    # Create synthetic volume for demo
    print("\n  Creating synthetic sphere volume for demo...")
    import numpy as np
    shape = (128, 256, 256)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center = np.array(shape) / 2
    dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)

    # Sphere with bone density in air
    hu_volume = np.where(dist < 50, 800.0, -900.0).astype(np.float32)

    preprocessor = VolumePreprocessor.from_numpy(
        hu_volume,
        spacing_zyx_mm=(1.0, 0.5, 0.5)
    )
    return preprocessor.preprocess(output_dir=CACHE_DIR)


def load_cached_volume() -> PreprocessedVolume:
    if (CACHE_DIR / "mu_volume.npy").exists():
        print(f"  Loading cached volume from: {CACHE_DIR}")
        return PreprocessedVolume.load(CACHE_DIR)
    else:
        return create_synthetic_volume()


def load_ct_volume_from_path(path: Path) -> PreprocessedVolume:
     # Check if we have a cached preprocessed volume
    if path.exists():
        print(f"  Loading NIfTI from: {path}")
        preprocessor = VolumePreprocessor.from_nifti(nifti_path=path, settings=DEFAULT_PREPROCESSING_SETTINGS)
        print(f"  Shape: {preprocessor.shape}")
        print(f"  Spacing: {preprocessor.spacing_zyx_mm} mm")
        print(f"  HU range: {preprocessor.hu_range}")
        return preprocessor.preprocess(output_dir=CACHE_DIR)
    else:
        print("\n  ERROR: CT data not found at:")
        print(f"    NIfTI: {path}")
        print("     Will use cached or synthetic volume.")
        return load_cached_volume()


def qimage_from_array(image: np.ndarray) -> QImage:
    """Convert a uint8 RGB image to QImage."""
    rgb = np.ascontiguousarray(image)
    height, width, _channels = rgb.shape
    bytes_per_line = width * 3
    qimage = QImage(rgb.data, width, height, bytes_per_line, IMAGE_FORMAT_RGB888)
    return qimage.copy()


class SliderRow(QWidget):
    """A labeled slider with value display."""

    def __init__(
        self,
        label: str,
        minimum: int,
        maximum: int,
        value: int,
        scale: float = 1.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.scale = scale

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label)
        self.label.setMinimumWidth(170)
        self.slider = QSlider(ORIENTATION_HORIZONTAL)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(80)
        self.value_label.setAlignment(ALIGN_RIGHT_VCENTER)

        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)

        self.slider.valueChanged.connect(self._sync_label)
        self._sync_label(value)

    def _sync_label(self, value: int) -> None:
        self.value_label.setText(f"{value / self.scale:.2f}")

    def value(self) -> float:
        return self.slider.value() / self.scale

    def set_value(self, value: float) -> None:
        self.slider.setValue(int(round(value * self.scale)))


class IntegerInputRow(QWidget):
    """A labeled integer text input that keeps the last committed value."""

    def __init__(
        self,
        label: str,
        value: int,
        minimum: int,
        maximum: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._value = int(value)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label)
        self.label.setMinimumWidth(170)
        self.input = QLineEdit(str(self._value))
        self.input.setValidator(QIntValidator(minimum, maximum, self))
        self.input.setAlignment(ALIGN_RIGHT_VCENTER)
        self.input.setMinimumWidth(80)
        self.input.setMaximumWidth(120)

        layout.addWidget(self.label)
        layout.addStretch(1)
        layout.addWidget(self.input)

        self.input.editingFinished.connect(self._commit_text)

    def _commit_text(self) -> None:
        text = self.input.text().strip()

        if text and self.input.hasAcceptableInput():
            self._value = int(text)
        else:
            self.input.setText(str(self._value))

    def value(self) -> int:
        return self._value

    def set_value(self, value: int) -> None:
        self._value = int(value)
        self.input.setText(str(self._value))


class FloatInputRow(QWidget):
    """A labeled floating-point text input that keeps the last committed value."""

    def __init__(
        self,
        label: str,
        value: float,
        minimum: float,
        maximum: float,
        decimals: int = 6,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._value = float(value)
        self._decimals = decimals

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label)
        self.label.setMinimumWidth(170)
        self.input = QLineEdit(self._format_value(self._value))
        self.input.setValidator(QDoubleValidator(minimum, maximum, decimals, self))
        self.input.setAlignment(ALIGN_RIGHT_VCENTER)
        self.input.setMinimumWidth(80)
        self.input.setMaximumWidth(120)

        layout.addWidget(self.label)
        layout.addStretch(1)
        layout.addWidget(self.input)

        self.input.editingFinished.connect(self._commit_text)

    def _format_value(self, value: float) -> str:
        return f"{value:.{self._decimals}f}".rstrip("0").rstrip(".")

    def _commit_text(self) -> None:
        text = self.input.text().strip()

        if text and self.input.hasAcceptableInput():
            self._value = float(text)
        self.input.setText(self._format_value(self._value))

    def value(self) -> float:
        return self._value

    def set_value(self, value: float) -> None:
        self._value = float(value)
        self.input.setText(self._format_value(self._value))


class XraySimulatorWindow(QMainWindow):
    """Main GUI window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"X-Ray Simulation Viewer ({QT_API})")
        self.resize(1500, 950)

        self.volume: PreprocessedVolume = load_cached_volume()
        self.realism_enabled = False
        self.simulator: FluoroSimulator | None = None
        self._display_image: np.ndarray | None = None
        self._needs_simulator_reset = True

        self.render_timer = QTimer(self)
        #self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self.render_current_view)

        self._build_ui()
        self._set_status("Loaded synthetic demo volume.")
        self.render_timer.start(33)  # Start rendering at 30 fps immediately after UI is shown

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        self.image_label = QLabel("Rendering...")
        self.image_label.setAlignment(ALIGN_CENTER)
        self.image_label.setMinimumSize(700, 700)
        self.image_label.setStyleSheet("background:#111; color:#ddd; border:1px solid #333;")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.image_label)
        root.addWidget(scroll, 3)

        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(10)
        root.addWidget(sidebar, 2)

        load_group = QGroupBox("Data")
        load_layout = QVBoxLayout(load_group)
        self.load_button = QPushButton("Load CT Volume")
        self.load_button.clicked.connect(self.load_ct_volume)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        load_layout.addWidget(self.load_button)
        load_layout.addWidget(self.status_label)
        sidebar_layout.addWidget(load_group)

        geometry_group = QGroupBox("Geometry")
        geometry_layout = QVBoxLayout(geometry_group)
        self.geometry_sliders: dict[str, SliderRow] = {
            "sdd": SliderRow("Source-Detector [mm]", 300, 3000, int(DEFAULT_GEOMETRY.source_to_detector_mm)),
            "sid": SliderRow("Source-Iso [mm]", 250, 2950, int(DEFAULT_GEOMETRY.source_to_isocenter_mm)),
        }
        for slider in self.geometry_sliders.values():
            geometry_layout.addWidget(slider)
        self.detector_inputs: dict[str, IntegerInputRow] = {
            "width": IntegerInputRow("Detector Width [px]", DEFAULT_GEOMETRY.detector_width_px, 1, 8192),
            "height": IntegerInputRow("Detector Height [px]", DEFAULT_GEOMETRY.detector_height_px, 1, 8192),
        }
        for detector_input in self.detector_inputs.values():
            geometry_layout.addWidget(detector_input)
        self.pixel_spacing_input = FloatInputRow(
            "Pixel Spacing [mm]",
            DEFAULT_GEOMETRY.pixel_spacing_mm,
            0.001,
            100.0,
        )
        geometry_layout.addWidget(self.pixel_spacing_input)
        sidebar_layout.addWidget(geometry_group)

        pose_group = QGroupBox("Pose")
        pose_layout = QVBoxLayout(pose_group)
        self.pose_sliders: dict[str, SliderRow] = {
           "rx": SliderRow("X Rotation [deg]", -1800, 1800, 0, scale=10.0),
            "ry": SliderRow("Y Rotation [deg]", -1800, 1800, 0, scale=10.0),
            "rz": SliderRow("Z Rotation [deg]", -1800, 1800, 0, scale=10.0),
            "tx": SliderRow("X Translation [mm]", -1500, 1500, 0),
            "ty": SliderRow("Y Translation [mm]", -1500, 1500, 0),
            "tz": SliderRow("Z Translation [mm]", -1500, 1500, 0),
        }
        for slider in self.pose_sliders.values():
            pose_layout.addWidget(slider)
        sidebar_layout.addWidget(pose_group)

        realism_group = QGroupBox("Realism")
        realism_layout = QVBoxLayout(realism_group)
        self.realism_checkbox = QCheckBox("Enable Realism")
        self.realism_checkbox.toggled.connect(self.toggle_realism)
        realism_layout.addWidget(self.realism_checkbox)

        self.realism_sliders_group = QGroupBox("Realism Sliders")
        realism_sliders_layout = QVBoxLayout(self.realism_sliders_group)
        self.realism_sliders: dict[str, SliderRow] = {
            "poisson": SliderRow("Poisson Noise [photons]", 0, 100, 0, scale=0.01),
            "gaussian": SliderRow("Gaussian Noise Sigma", 0, 100, 0, scale=400.0),
            "blur": SliderRow("Gaussian Blur [px]", 0, 100, 0, scale=10.0),
        }
        for slider in self.realism_sliders.values():
            slider.slider.valueChanged.connect(self.on_realism_slider_changed)
            realism_sliders_layout.addWidget(slider)
        self.realism_sliders_group.setEnabled(False)
        realism_layout.addWidget(self.realism_sliders_group)
        sidebar_layout.addWidget(realism_group)

        actions = QHBoxLayout()
        self.reset_button = QPushButton("Reset Controls")
        self.reset_button.clicked.connect(self.reset_controls)
        actions.addWidget(self.reset_button)
        sidebar_layout.addLayout(actions)
        sidebar_layout.addStretch(1)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def geometry_controls(self) -> CarmGeometry:
        return CarmGeometry(
            detector_width_px=self.detector_inputs["width"].value(),
            detector_height_px=self.detector_inputs["height"].value(),
            pixel_spacing_mm=self.pixel_spacing_input.value(),
            source_to_detector_mm=self.geometry_sliders["sdd"].value(),
            source_to_isocenter_mm=self.geometry_sliders["sid"].value(),
        )
    
    def realism_controls(self) -> RealismSettings:
        if not self.realism_enabled:
            return RealismSettings(
                enabled=False,
                gaussian_sigma=0.0,
                poisson_photons=0.0,
                blur_sigma_px=0.0,
                seed=0,
            )

        poisson_level = self.realism_sliders["poisson"].value()
        gaussian_level = self.realism_sliders["gaussian"].value()
        blur_level = self.realism_sliders["blur"].value()

        return RealismSettings(
            enabled=True,
            gaussian_sigma=gaussian_level,
            poisson_photons=poisson_level,
            blur_sigma_px=blur_level,
            seed=0,
        )

    def _rebuild_simulator_if_needed(self) -> bool:
        sid = self.geometry_sliders["sid"].value()
        sdd = max(self.geometry_sliders["sdd"].value(), sid + 1.0)

        if sdd != self.geometry_sliders["sdd"].value():
            self.geometry_sliders["sdd"].set_value(sdd)

        new_simulator_required = (
            self.simulator is None
            or self._needs_simulator_reset
            or self.simulator.config.geometry != self.geometry_controls()
            or self.simulator.config.realism != self.realism_controls()
        )
        if not new_simulator_required:
            return False

        config = SimulatorConfig(
            geometry=self.geometry_controls(),
            physics=DEFAULT_PHYSICS,
            realism=self.realism_controls(),
            output=OutputSettings(save_to_disk=False),
        )
        print(config.realism)
        self.simulator = FluoroSimulator(self.volume, config)
        self._needs_simulator_reset = False
        return True

    def rotation_controls(self) -> tuple[float, float, float]:
        return (
            math.radians(self.pose_sliders["rx"].value() - 90.0),
            math.radians(self.pose_sliders["ry"].value()),
            math.radians(self.pose_sliders["rz"].value()),
        )

    def translation_controls(self) -> tuple[float, float, float]:
        return (
            self.pose_sliders["tx"].value(),
            self.pose_sliders["ty"].value(),
            self.pose_sliders["tz"].value(),
        )

    def render_current_view(self) -> None:
        try:
            simulator_changed = self._rebuild_simulator_if_needed()
            assert self.simulator is not None
            do_render = ( simulator_changed
                         or self.current_rotation != self.rotation_controls()
                         or self.current_translation != self.translation_controls()
            )
            if do_render:
                self.current_rotation = self.rotation_controls()
                self.current_translation = self.translation_controls()
                frame = self.simulator.render_frame(
                    rotation=self.current_rotation,
                    translation=self.current_translation,
                )
                display = self._format_for_display(frame.image)
                self._display_image = display
                self._update_image_label(display)
                self._set_status(
                    f"Rendered {display.shape[1]}x{display.shape[0]} frame from "
                    f"{self.volume.metadata.source or 'synthetic volume'}."
                )
        except Exception as exc:
            self._set_status(f"Render failed: {exc}")
            QMessageBox.critical(self, "Render Error", str(exc))

    def _format_for_display(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32, copy=False)
        low = float(np.min(img))
        high = float(np.max(img))
        if high > low:
            img = (img - low) / (high - low)
        else:
            img = np.zeros_like(img, dtype=np.float32)

        gray = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
        return np.repeat(gray[:, :, np.newaxis], 3, axis=2)

    def _update_image_label(self, rgb: np.ndarray) -> None:
        qimage = qimage_from_array(rgb)
        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(
            self.image_label.size(),
            KEEP_ASPECT,
            SMOOTH_TRANSFORM,
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._display_image is not None:
            self._update_image_label(self._display_image)

    def reset_controls(self) -> None:
        self.geometry_sliders["sdd"].set_value(DEFAULT_GEOMETRY.source_to_detector_mm)
        self.geometry_sliders["sid"].set_value(DEFAULT_GEOMETRY.source_to_isocenter_mm)
        self.detector_inputs["width"].set_value(DEFAULT_GEOMETRY.detector_width_px)
        self.detector_inputs["height"].set_value(DEFAULT_GEOMETRY.detector_height_px)
        self.pixel_spacing_input.set_value(DEFAULT_GEOMETRY.pixel_spacing_mm)
        for key in ("rx", "ry", "rz", "tx", "ty", "tz"):
            self.pose_sliders[key].set_value(0.0)
        self.realism_checkbox.setChecked(False)
        for slider in self.realism_sliders.values():
            slider.set_value(0.0)

    def toggle_realism(self, checked: bool) -> None:
        self.realism_enabled = checked
        self.realism_sliders_group.setEnabled(checked)
        self._needs_simulator_reset = True

    def on_realism_slider_changed(self, _value: int) -> None:
        self._needs_simulator_reset = True

    def load_ct_volume(self) -> None:
        path_str, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Load Image or Volume",
            str(Path.cwd()),
            (
                "Supported Files (*.nii *.nii.gz *.npy *.png *.jpg *.jpeg *.bmp *.tif *.tiff);;"
                "All Files (*)"
            ),
        )
        if not path_str:
            return

        try:
            path = Path(path_str)
            self.volume = load_ct_volume_from_path(path)
            self._needs_simulator_reset = True
            self._set_status(f"Loaded: {path}")
        except Exception as exc:
            self._set_status(f"Load failed: {exc}")
            QMessageBox.critical(self, "Load Error", str(exc))


def main() -> int:
    app = QApplication(sys.argv)
    window = XraySimulatorWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
