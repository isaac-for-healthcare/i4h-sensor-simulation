#!/usr/bin/env python3
"""Interactive cone-beam X-ray projection with geometry sliders.

Generates a synthetic 3D attenuation phantom, projects it with a diverging
beam (Beer–Lambert law), and shows the 2D image as a numpy array. Sliders
control source–iso distance (SOD), source–detector distance (SID), and
in-plane rotation about the patient (iso-center).

Run:
    python3 xray_interactive_view.py

Requires: numpy, scipy, matplotlib
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from scipy.ndimage import map_coordinates

import os
from pathlib import Path

from fluorosim import (
    CarmGeometry,
    FluoroSimulator,
    OutputSettings,
    Pose,
    PreprocessedVolume,
    RealismSettings,
    SimulatorConfig,
    VolumePreprocessor,
)
from fluorosim.config import XrayPhysics
from fluorosim.rendering import realism

# Path to DICOM CT data (e.g., generated from NV-Generate-CTMR inference_tutorial script)
CT_BASENAME = "sample_20260403_141223_077521_image"
NIFTI_CT_FILE = NIFTI_CT_PATH = Path(CT_BASENAME + ".nii.gz")
NIFTI_CT_DIR = Path("~/workspace/NV-Generate-CTMR/output/").expanduser()
NIFTI_CT_PATH = NIFTI_CT_DIR / NIFTI_CT_FILE
print(f"Input CT file: {NIFTI_CT_PATH}")

_SCRIPT_DIR = Path("~/workspace/i4h-sensor-simulation/fluoro-simulator/examples").expanduser().resolve().parent.parent

# Output directory
OUTPUT_DIR = Path(os.environ.get("FLUOROSIM_OUTPUT_DIR", str(_SCRIPT_DIR / "output"))).expanduser()

# Cache directory for preprocessed volume
CACHE_DIR = Path(os.environ.get("FLUOROSIM_CACHE_DIR", str(OUTPUT_DIR / "cache" / CT_BASENAME))).expanduser()

# Create the cache directory if it doesn't exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Alternative: pre-generated mu_volume if it exists
CACHED_MU_VOLUME = CACHE_DIR / "mu_volume.npy"
print(f"Cache Directory: {CACHE_DIR}")

def load_ct_volume() -> np.ndarray:

    print("\n[Step 1] Loading CT Volume...")

    # Check if we have a cached preprocessed volume
    if (CACHE_DIR / "mu_volume.npy").exists():
        print(f"  Loading cached volume from: {CACHE_DIR}")
        mu = PreprocessedVolume.load(CACHE_DIR)

    elif NIFTI_CT_PATH.exists():
        print(f"  Loading NIfTI from: {NIFTI_CT_PATH}")
        preprocessor = VolumePreprocessor.from_nifti(NIFTI_CT_PATH)
        print(f"  Shape: {preprocessor.shape}")
        print(f"  Spacing: {preprocessor.spacing_zyx_mm} mm")
        print(f"  HU range: {preprocessor.hu_range}")

        mu = preprocessor.preprocess(output_dir=CACHE_DIR)
    else:
        print("\n  ERROR: CT data not found at:")
        print(f"    NIfTI: {NIFTI_CT_PATH}")
        print("  Please update the paths in this script.")
        print("\n  Alternatively, run with a synthetic volume:")
        print("    python -m ...run_imagecas --synthetic")

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
        mu = preprocessor.preprocess(output_dir=CACHE_DIR)

    return mu

def create_simulator(
    mu_volume: np.ndarray,
    geometry: CarmGeometry,
    realism: RealismSettings,
    physics: XrayPhysics) -> FluoroSimulator:
    config = SimulatorConfig(
        geometry=geometry,
        realism=realism,
        output=OutputSettings(
            save_to_disk=False,
            output_dir=str(OUTPUT_DIR),
            format="png",
        ),
    )

    return FluoroSimulator(mu_volume, config)

def render_xray_image(
    simulator: FluoroSimulator,
    rotation: (float, float, float),
    translation: (float, float, float)
) -> np.ndarray:
    frame = simulator.render_frame(rotation, translation)
    return frame.image


_geometry=CarmGeometry(
    detector_width_px=1024,
    detector_height_px=1024,
    pixel_spacing_mm=0.5,
    source_to_detector_mm=2220.0,
    source_to_isocenter_mm=2180.0
)

_realism=RealismSettings(
    enabled=False,
    gaussian_sigma=0.0,
    poisson_photons=0,
    blur_sigma_px=0.0,
    seed=0,
)

_physics = XrayPhysics(
    step_mm=0.5,                    # Ray-march step size (smaller = more accurate)
    i0=1.0,                         # Unattenuated intensity
    normalize=True,                 # Normalize output to [0, 1]
    invert=True,                    # Clinical convention: bone=white, air=black
)

_simulator: FluroSimulator

def main() -> None:
    global _geometry
    global _realism
    global _physics
    global _simulator

    mu = load_ct_volume()

    init_x_angle = 0
    init_y_angle = 0
    init_z_angle = 0

    _simulator = create_simulator(
        mu_volume=mu,
        geometry=_geometry,
        realism=_realism,
        physics=_physics
    )

    image = render_xray_image(
        _simulator,
        rotation=(math.radians(init_x_angle-90), math.radians(init_y_angle), math.radians(init_z_angle)),
        translation=(0, 0, 0),
    )

    height, width = image.shape[:2]

    # Get the default DPI from Matplotlib rcParams
    dpi = plt.rcParams['figure.dpi']

    # Calculate figsize in inches (pixels / dpi)
    figsize = width / float(dpi) / 2 + 2, height / float(dpi) / 2

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.10, bottom=0.42, right=0.98, top=0.94)
    im = ax.imshow(image, cmap="gray", interpolation="bilinear")
    ax.set_title("Synthetic X-ray")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("−log transmission (relative)")

    # Sliders: SID, SOD, rotation (degrees)
    ax_sdd = fig.add_axes([0.12, 0.36, 0.76, 0.03])
    ax_sid = fig.add_axes([0.12, 0.32, 0.76, 0.03])
    ax_xang = fig.add_axes([0.12, 0.28, 0.76, 0.03])
    ax_yang = fig.add_axes([0.12, 0.24, 0.76, 0.03])
    ax_zang = fig.add_axes([0.12, 0.20, 0.76, 0.03])
    ax_xtran = fig.add_axes([0.12, 0.16, 0.76, 0.03])
    ax_ytran = fig.add_axes([0.12, 0.12, 0.76, 0.03])
    ax_ztran = fig.add_axes([0.12, 0.08, 0.76, 0.03])

    sdd_slider = Slider(
        ax=ax_sdd,
        label="SDD (source–detector) [mm]",
        valmin=250.0,
        valmax=3000.0,
        valinit=_geometry.source_to_isocenter_mm
    )
    sid_slider = Slider(
        ax=ax_sid,
        label="SID (source–iso) [mm]",
        valmin=250.0,
        valmax=3000.0,
        valinit=_geometry.source_to_detector_mm
    )
    xang_slider = Slider(
        ax=ax_xang,
        label="X-axis Rot [deg]",
        valmin=-180.0,
        valmax=180.0,
        valinit=init_x_angle,
    )
    yang_slider = Slider(
        ax=ax_yang,
        label="Y-axis Rot [deg]",
        valmin=-180.0,
        valmax=180.0,
        valinit=init_y_angle,
    )
    zang_slider = Slider(
        ax=ax_zang,
        label="Z-axis Rot [deg]",
        valmin=-180.0,
        valmax=180.0,
        valinit=init_z_angle,
    )
    xtran_slider = Slider(
        ax=ax_xtran,
        label="X-axis Trans [mm]",
        valmin=-1024.0,
        valmax=1024.0,
        valinit=0,
    )
    ytran_slider = Slider(
        ax=ax_ytran,
        label="Y-axis Trans [mm]",
        valmin=-1024.0,
        valmax=1024.0,
        valinit=0,
    )
    ztran_slider = Slider(
        ax=ax_ztran,
        label="Z-axis Trans [mm]",
        valmin=-1024.0,
        valmax=1024.0,
        valinit=0,
    )

    def update(_val=None) -> None:
        global _geometry
        global _simulator
        global _realism
        global _physics
        sdd = float(sdd_slider.val)
        sid = float(sid_slider.val)
        xang = float(xang_slider.val)
        yang = float(yang_slider.val)
        zang = float(zang_slider.val)
        xtran = float(xtran_slider.val)
        ytran = float(ytran_slider.val)
        ztran = float(ztran_slider.val)
        if sdd <= sid + 5.0:
            sdd = sid + 5.0
            sdd_slider.eventson = False
            sdd_slider.set_val(sid)
            sdd_slider.eventson = True

        if sid != _geometry.source_to_isocenter_mm or sdd != _geometry.source_to_detector_mm:
            new_geom=CarmGeometry(
                detector_width_px=_geometry.detector_width_px,
                detector_height_px=_geometry.detector_height_px,
                pixel_spacing_mm=_geometry.pixel_spacing_mm,
                source_to_detector_mm=sdd,
                source_to_isocenter_mm=sid
            )
            _simulator = create_simulator(
                mu_volume=mu,
                geometry=new_geom,
                realism=_realism,
                physics=_physics
            )
            _geometry = new_geom

        img = render_xray_image(
            _simulator,
            rotation=(math.radians(xang-90), math.radians(yang), math.radians(zang)),
            translation=(xtran, ytran, ztran)
        )
        im.set_data(img)
        im.set_clim(float(np.percentile(img, 1)), float(np.percentile(img, 99.5)))
        fig.canvas.draw_idle()

    sdd_slider.on_changed(update)
    sid_slider.on_changed(update)
    xang_slider.on_changed(update)
    yang_slider.on_changed(update)
    zang_slider.on_changed(update)
    xtran_slider.on_changed(update)
    ytran_slider.on_changed(update)
    ztran_slider.on_changed(update)

    reset_ax = fig.add_axes([0.82, 0.02, 0.12, 0.04])
    reset_btn = Button(reset_ax, "Reset", hovercolor="0.9")

    def reset(_event) -> None:
        sdd_slider.reset()
        sid_slider.reset()
        xang_slider.reset()
        yang_slider.reset()
        zang_slider.reset()
        xtran_slider.reset()
        ytran_slider.reset()
        ztran_slider.reset()
        update()

    reset_btn.on_clicked(reset)

    update()
    print("2D image array shape:", image.shape, "dtype:", image.dtype)
    plt.show()


if __name__ == "__main__":
    main()
