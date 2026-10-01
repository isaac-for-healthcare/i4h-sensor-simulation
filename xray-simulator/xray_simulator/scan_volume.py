# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable HU volumes and reproducible DICOM conversion.

This file is deliberately identical in patient_digital_twin and xray_simulator.
Keep copies synchronized; artifacts record its hash. Array axes are index labels,
not anatomical directions. Integer indices locate voxel centres.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Conversion:
    world_frame: str = "RAS"  # RAS or LPS; both right-handed, Z superior
    world_unit: str = "m"  # m or mm
    origin: str = "dicom"  # dicom or source_center
    array_axes: str = "kji"  # Any permutation of native i,j,k
    spacing_ijk_mm: tuple | None = None  # None preserves the original voxel grid
    interpolation: str = "linear"  # linear or nearest; used only for resampling
    fill_hu: float = -1000.0


def digest(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def implementation():
    from importlib.metadata import PackageNotFoundError, version

    versions = {
        "helper_sha256": digest(__file__),
        "numpy": np.__version__,
        "PyYAML": yaml.__version__,
    }
    for name in ("SimpleITK", "nibabel"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def affine(image):
    a = np.eye(4)
    a[:3, :3] = np.asarray(image.GetDirection()).reshape(3, 3) @ np.diag(
        image.GetSpacing()
    )
    a[:3, 3] = image.GetOrigin()
    return a


def read_ct(directory, series_uid=None):
    """Reject ambiguous selection and geometry that cannot use one regular affine."""
    import SimpleITK as sitk

    directory = Path(directory).resolve()
    ids = list(sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory)) or [])
    if series_uid is None:
        if len(ids) != 1:
            raise ValueError(
                f"Select one series_uid explicitly; found {len(ids)} series"
            )
        series_uid = ids[0]
    if series_uid not in ids:
        raise ValueError("Selected series_uid was not found")
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), series_uid)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    reader.SetOutputPixelType(sitk.sitkFloat32)
    reader.MetaDataDictionaryArrayUpdateOn()
    image = reader.Execute()  # GDCM applies modality rescale; do not apply it again.
    if image.GetDimension() != 3 or len(files) != image.GetSize()[2] or len(files) < 2:
        raise ValueError(
            "Supported DICOM input is a regular, multi-slice, single-frame CT series"
        )
    a = affine(image)
    if not np.isfinite(a).all() or min(image.GetSpacing()) <= 0:
        raise ValueError("Invalid image geometry")
    d = np.asarray(image.GetDirection()).reshape(3, 3)
    if not np.allclose(d.T @ d, np.eye(3), atol=1e-6):
        raise ValueError(
            "Non-orthogonal DICOM slice geometry requires explicit reconstruction"
        )
    for k in range(len(files)):

        def tag(key):
            return reader.GetMetaData(k, key).strip()

        if tag("0008|0060") != "CT":
            raise ValueError("Expected CT modality")
        position = np.array([float(v) for v in tag("0020|0032").split("\\")])
        orientation = np.array([float(v) for v in tag("0020|0037").split("\\")])
        spacing = np.array([float(v) for v in tag("0028|0030").split("\\")])
        expected_position = (a @ [0, 0, k, 1])[:3]
        if (
            not np.allclose(position, expected_position, atol=1e-3, rtol=0)
            or not np.allclose(orientation, np.r_[d[:, 0], d[:, 1]], atol=1e-6, rtol=0)
            or not np.allclose(spacing[::-1], image.GetSpacing()[:2], atol=1e-6, rtol=0)
        ):
            raise ValueError(
                "Irregular slice geometry: an explicit reconstruction is required"
            )
    source = {
        "kind": "dicom",
        "series_uid": series_uid,
        "frame": "DICOM_LPS",
        "unit": "mm",
        "size_ijk": list(image.GetSize()),
        "index_ijk_to_lps_mm": a.tolist(),
        "files": [
            {"path": str(Path(p).resolve().relative_to(directory)), "sha256": digest(p)}
            for p in files
        ],
    }
    return image, source


def convert(image, options=Conversion()):
    """Convert grid/coordinates independently. Integer indices denote voxel centres."""
    import SimpleITK as sitk

    c = options
    if (
        c.world_frame not in ("RAS", "LPS")
        or c.world_unit not in ("m", "mm")
        or c.origin not in ("dicom", "source_center")
        or sorted(c.array_axes) != list("ijk")
        or c.interpolation not in ("linear", "nearest")
        or not np.isfinite(c.fill_hu)
    ):
        raise ValueError("Invalid conversion settings")
    source_a = affine(image)
    source_center = (source_a @ np.r_[(np.array(image.GetSize()) - 1) / 2, 1])[:3]
    if c.spacing_ijk_mm is not None:
        spacing = np.asarray(c.spacing_ijk_mm, dtype=float)
        if (
            spacing.shape != (3,)
            or not np.isfinite(spacing).all()
            or np.any(spacing <= 0)
        ):
            raise ValueError("spacing_ijk_mm must contain three positive finite values")
        extent = (np.array(image.GetSize()) - 1) * np.array(image.GetSpacing())
        size = np.ceil(extent / spacing).astype(int) + 1
        method = (
            sitk.sitkLinear if c.interpolation == "linear" else sitk.sitkNearestNeighbor
        )
        image = sitk.Resample(
            image,
            size.tolist(),
            sitk.Transform(3, sitk.sitkIdentity),
            method,
            image.GetOrigin(),
            spacing.tolist(),
            image.GetDirection(),
            c.fill_hu,
            sitk.sitkFloat32,
        )
    grid_a = affine(image)
    world_from_lps = np.eye(4)
    signs = [-1, -1, 1] if c.world_frame == "RAS" else [1, 1, 1]
    world_from_lps[:3, :3] = np.diag(signs) * (0.001 if c.world_unit == "m" else 1)
    if c.origin == "source_center":
        world_from_lps[:3, 3] = -world_from_lps[:3, :3] @ source_center
    array_to_ijk = np.eye(4)
    array_to_ijk[:3, :3] = np.eye(3)[:, ["ijk".index(v) for v in c.array_axes]]
    values = sitk.GetArrayFromImage(image).transpose(
        ["kji".index(v) for v in c.array_axes]
    )
    values = np.ascontiguousarray(values, dtype=np.float32)
    metadata = {
        "array_axes": c.array_axes,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "intensity_unit": "HU",
        "world_frame": c.world_frame,
        "world_unit": c.world_unit,
        "index_location": "voxel_center",
        "resampled": c.spacing_ijk_mm is not None,
        "world_from_lps_mm": world_from_lps.tolist(),
        "array_index_to_world": (world_from_lps @ grid_a @ array_to_ijk).tolist(),
        "array_index_to_source_ijk": (
            np.linalg.inv(source_a) @ grid_a @ array_to_ijk
        ).tolist(),
    }
    return values, metadata


class ScanVolume:
    """HU array plus its complete, serializable scan/world geometry."""

    def __init__(self, values, metadata):
        import copy

        self.metadata = copy.deepcopy(metadata)
        out = self.metadata["output"]
        self.values = np.ascontiguousarray(values, dtype=np.float32).copy()
        a = np.asarray(out["array_index_to_world"], dtype=float)
        if (
            self.values.ndim != 3
            or not self.values.size
            or not np.isfinite(self.values).all()
            or a.shape != (4, 4)
            or not np.isfinite(a).all()
            or not np.allclose(a[3], [0, 0, 0, 1])
            or np.linalg.matrix_rank(a[:3, :3]) != 3
            or sorted(out["array_axes"]) != list("ijk")
            or out["world_frame"] not in ("RAS", "LPS")
            or out["world_unit"] not in ("m", "mm", "micron")
            or out.get("intensity_unit") != "HU"
            or list(self.values.shape) != out["shape"]
        ):
            raise ValueError("Invalid scan volume geometry, shape, or HU values")
        self.values.setflags(write=False)

    @property
    def array_axes(self):
        return self.metadata["output"]["array_axes"]

    @property
    def meters_per_unit(self):
        return {"m": 1.0, "mm": 0.001, "micron": 1e-6}[
            self.metadata["output"]["world_unit"]
        ]

    @property
    def frame(self):
        return self.metadata["output"]["world_frame"]

    @property
    def ijk_to_world(self):
        # Columns select native i,j,k from the recorded array-axis affine.
        a = np.asarray(self.metadata["output"]["array_index_to_world"], dtype=float)
        p = np.eye(4)
        p[:3, :3] = np.eye(3)[:, [self.array_axes.index(c) for c in "ijk"]]
        return a @ p

    @property
    def ijk_to_ras_m(self):
        t = np.diag([self.meters_per_unit] * 3 + [1.0])
        if self.frame == "LPS":
            t = np.diag([-1.0, -1.0, 1.0, 1.0]) @ t
        return t @ self.ijk_to_world

    @property
    def values_kji(self):
        return self.values.transpose([self.array_axes.index(c) for c in "kji"])

    def save(self, output, *, filename="volume.npy"):
        """Write one NumPy file and one YAML sidecar into a new directory."""
        import copy

        output = Path(output)
        if Path(filename).name != filename or not filename.endswith(".npy"):
            raise ValueError("filename must be a simple .npy basename")
        output.mkdir(parents=True, exist_ok=False)
        path = output / filename
        np.save(path, self.values, allow_pickle=False)
        recipe = copy.deepcopy(self.metadata)
        recipe["output"].update(file=filename, sha256=digest(path))
        sidecar = output / "volume.yaml"
        sidecar.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
        return sidecar


def from_array(
    values,
    array_index_to_world,
    *,
    array_axes="ijk",
    world_frame="RAS",
    world_unit="mm",
    source=None,
):
    """Describe an existing grid without changing its values, axes, or spacing."""
    values = np.asarray(values)
    return ScanVolume(
        values,
        {
            "schema_version": 1,
            "implementation": implementation(),
            "source": source or {"kind": "numpy"},
            "conversion": None,
            "output": {
                "array_axes": array_axes,
                "shape": list(values.shape),
                "dtype": "float32",
                "intensity_unit": "HU",
                "world_frame": world_frame,
                "world_unit": world_unit,
                "index_location": "voxel_center",
                "resampled": False,
                "array_index_to_world": np.asarray(
                    array_index_to_world, dtype=float
                ).tolist(),
            },
        },
    )


def from_nifti(path):
    """Read the native NIfTI array and affine; no canonicalization or resampling."""
    import nibabel as nib

    path = Path(path)
    image = nib.load(str(path))
    units = image.header.get_xyzt_units()[0]
    unit = {"unknown": "mm", "mm": "mm", "meter": "m", "micron": "micron"}.get(units)
    if unit is None:
        raise ValueError(f"Unsupported spatial units: {units}")
    return from_array(
        image.get_fdata(dtype=np.float32),
        image.affine,
        world_unit=unit,
        source={
            "kind": "nifti",
            "file": path.name,
            "sha256": digest(path),
            "header_spatial_unit": units,
        },
    )


def from_dicom(directory, *, series_uid=None, conversion=Conversion()):
    image, source = read_ct(directory, series_uid)
    values, output = convert(image, conversion)
    settings = asdict(conversion)
    if settings["spacing_ijk_mm"] is not None:
        settings["spacing_ijk_mm"] = list(settings["spacing_ijk_mm"])
    return ScanVolume(
        values,
        {
            "schema_version": 1,
            "implementation": implementation(),
            "source": source,
            "conversion": settings,
            "output": output,
        },
    )


def load_artifact(path, *, verify_hash=True):
    path = Path(path)
    recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
    if recipe.get("schema_version") != 1:
        raise ValueError("Unsupported scan artifact schema")
    name = recipe["output"]["file"]
    if Path(name).name != name:
        raise ValueError("Volume must be beside the YAML artifact")
    volume = path.parent / name
    if verify_hash and digest(volume) != recipe["output"]["sha256"]:
        raise ValueError("Volume hash does not match YAML metadata")
    return ScanVolume(np.load(volume, allow_pickle=False), recipe)


def export_ct(directory, output, *, series_uid=None, options=Conversion()):
    return from_dicom(directory, series_uid=series_uid, conversion=options).save(output)


def replay(recipe_path, source_path, output):
    """Re-run a DICOM/NIfTI recipe and verify source, geometry, and output hash."""
    recipe = yaml.safe_load(Path(recipe_path).read_text(encoding="utf-8"))
    if recipe["schema_version"] != 1 or recipe["implementation"] != implementation():
        raise ValueError("Replay requires the recorded helper and dependency versions")
    source = recipe["source"]
    if source["kind"] == "dicom":
        for item in source["files"]:
            if digest(Path(source_path) / item["path"]) != item["sha256"]:
                raise ValueError(f"Input changed: {item['path']}")
        scan = from_dicom(
            source_path,
            series_uid=source["series_uid"],
            conversion=Conversion(**recipe["conversion"]),
        )
    elif source["kind"] == "nifti":
        if digest(source_path) != source["sha256"]:
            raise ValueError("Input NIfTI changed")
        scan = from_nifti(source_path)
    else:
        raise ValueError("Replay requires original DICOM or NIfTI input")
    saved = scan.save(output, filename=recipe["output"]["file"])
    if yaml.safe_load(saved.read_text()) != recipe:
        raise ValueError(
            "Replay differs: inspect geometry, source list, and volume hash"
        )
    return saved
