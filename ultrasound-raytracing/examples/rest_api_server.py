# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
REST API Ultrasound Simulation Server

A stateless REST API that renders GPU-accelerated B-mode ultrasound images
on demand. Each request specifies probe type, pose, and simulation parameters;
the server returns the rendered image as a PNG.

Usage:
    python rest_api_server.py [--host 0.0.0.0] [--port 8000] [--mesh-dir mesh]

Endpoints:
    GET  /api/v1/health         — Health check
    GET  /api/v1/probe-types    — List available probe types with defaults
    GET  /api/v1/sim-params     — Default simulation parameters
    POST /api/v1/simulate       — Run simulation, returns PNG image

Requires:
    pip install fastapi uvicorn pillow
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import threading
import time
from enum import Enum
from typing import Any, Optional

import numpy as np

# Add root to path so raysim is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import raysim.cuda as rs
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger("rest_api_us")

# ---------------------------------------------------------------------------
# Pydantic models — request / response schemas
# ---------------------------------------------------------------------------


class ProbeType(str, Enum):
    curvilinear = "curvilinear"
    linear = "linear"
    phased = "phased"


class PoseParams(BaseModel):
    """6-DOF probe pose: position (mm) + Euler rotation (radians)."""

    position: list[float] = Field(
        default=[-14.0, -122.0, 72.0],
        min_length=3,
        max_length=3,
        description="Probe position [x, y, z] in mm.",
    )
    rotation: list[float] = Field(
        default=[float(np.deg2rad(-90)), float(np.deg2rad(180)), 0.0],
        min_length=3,
        max_length=3,
        description="Probe rotation [rx, ry, rz] in radians.",
    )


class SimulationParams(BaseModel):
    """Tunable simulation parameters (all optional — defaults used if omitted)."""

    conv_psf: bool = Field(default=True, description="Enable PSF convolution.")
    buffer_size: int = Field(default=4096, ge=64, description="Simulation buffer size.")
    t_far: float = Field(default=180.0, gt=0, description="Maximum ray travel distance (mm).")
    median_clip_filter: bool = Field(default=False, description="Enable median-clip filter.")
    b_mode_size: list[int] = Field(
        default=[512, 512],
        min_length=2,
        max_length=2,
        description="Output B-mode image size [height, width] in pixels.",
    )
    contact_epsilon: Optional[float] = Field(
        default=None, description="Contact detection epsilon. None keeps default."
    )
    dynamic_range_min: float = Field(
        default=-60.0, description="B-mode dynamic range lower bound (dB)."
    )
    dynamic_range_max: float = Field(
        default=0.0, description="B-mode dynamic range upper bound (dB)."
    )


class CurvilinearProbeParams(BaseModel):
    """Configuration specific to curvilinear probes."""

    num_elements_x: int = Field(default=256, ge=1, description="Number of elements.")
    sector_angle: float = Field(default=73.0, gt=0, description="Field of view (degrees).")
    radius: float = Field(default=45.0, gt=0, description="Probe radius (mm).")
    frequency: float = Field(default=5.0, gt=0, description="Frequency (MHz).")
    elevational_height: float = Field(default=7.0, gt=0, description="Elevational aperture height (mm).")
    num_el_samples: int = Field(default=10, ge=1, description="Elevational samples.")


class LinearProbeParams(BaseModel):
    """Configuration specific to linear-array probes."""

    num_elements_x: int = Field(default=256, ge=1, description="Number of elements.")
    width: float = Field(default=50.0, gt=0, description="Array width (mm).")
    frequency: float = Field(default=7.5, gt=0, description="Frequency (MHz).")
    elevational_height: float = Field(default=5.0, gt=0, description="Elevational aperture height (mm).")
    num_el_samples: int = Field(default=10, ge=1, description="Elevational samples.")


class PhasedProbeParams(BaseModel):
    """Configuration specific to phased-array probes."""

    num_elements_x: int = Field(default=128, ge=1, description="Number of elements.")
    width: float = Field(default=20.0, gt=0, description="Array width (mm).")
    sector_angle: float = Field(default=90.0, gt=0, description="Sector angle (degrees).")
    frequency: float = Field(default=3.5, gt=0, description="Frequency (MHz).")
    elevational_height: float = Field(default=5.0, gt=0, description="Elevational aperture height (mm).")
    num_el_samples: int = Field(default=10, ge=1, description="Elevational samples.")


class SimulateRequest(BaseModel):
    """Full specification for a single simulation frame."""

    probe_type: ProbeType = Field(default=ProbeType.curvilinear, description="Type of ultrasound probe.")
    pose: PoseParams = Field(default_factory=PoseParams, description="Probe 6-DOF pose.")
    sim_params: SimulationParams = Field(
        default_factory=SimulationParams, description="Simulation parameters."
    )
    probe_config: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Override default probe geometry. Keys must match the probe type's parameters "
            "(e.g. num_elements_x, sector_angle, radius, frequency, …). "
            "Omitted keys keep their defaults."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "probe_type": "curvilinear",
                    "pose": {
                        "position": [-14.0, -122.0, 72.0],
                        "rotation": [-1.5708, 3.1416, 0.0],
                    },
                    "sim_params": {
                        "conv_psf": True,
                        "buffer_size": 4096,
                        "t_far": 180.0,
                        "b_mode_size": [512, 512],
                    },
                }
            ]
        }
    }


class SimulateResponse(BaseModel):
    """Metadata returned alongside the image (used for JSON responses)."""

    probe_type: str
    pose: dict
    image_size: list[int]
    dynamic_range: list[float]
    sim_time_ms: float


class ProbeTypeInfo(BaseModel):
    name: str
    defaults: dict


class HealthResponse(BaseModel):
    status: str
    mesh_loaded: bool
    gpu_available: bool


# ---------------------------------------------------------------------------
# Probe defaults mapping
# ---------------------------------------------------------------------------

_PROBE_DEFAULTS: dict[str, dict] = {
    "curvilinear": {
        "cls": rs.CurvilinearProbe,
        "params_model": CurvilinearProbeParams,
        "kwargs": CurvilinearProbeParams().model_dump(),
    },
    "linear": {
        "cls": rs.LinearArrayProbe,
        "params_model": LinearProbeParams,
        "kwargs": LinearProbeParams().model_dump(),
    },
    "phased": {
        "cls": rs.PhasedArrayProbe,
        "params_model": PhasedProbeParams,
        "kwargs": PhasedProbeParams().model_dump(),
    },
}


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------


def _build_scene(mesh_dir: str) -> tuple[rs.World, rs.Materials]:
    """Build the simulation world and materials from mesh files."""
    materials = rs.Materials()
    world = rs.World("water")

    mesh_specs: list[tuple[str, str]] = [
        ("Tumor1.obj", "fat"),
        ("Tumor2.obj", "water"),
        ("Liver.obj", "liver"),
        ("Skin.obj", "fat"),
        ("Bone.obj", "bone"),
        ("Vessels.obj", "water"),
        ("Gallbladder.obj", "water"),
        ("Spleen.obj", "liver"),
        ("Heart.obj", "liver"),
        ("Stomach.obj", "water"),
        ("Pancreas.obj", "liver"),
        ("Small_bowel.obj", "water"),
        ("Colon.obj", "water"),
    ]

    loaded = 0
    for obj_file, mat_name in mesh_specs:
        path = os.path.join(mesh_dir, obj_file)
        if os.path.exists(path):
            idx = materials.get_index(mat_name)
            world.add(rs.Mesh(path, idx))
            logger.info("Loaded mesh %s (material=%s)", obj_file, mat_name)
            loaded += 1
        else:
            logger.warning("Mesh file not found, skipping: %s", path)

    if loaded == 0:
        logger.error("No meshes loaded from %s — simulation will produce empty images", mesh_dir)

    return world, materials


# ---------------------------------------------------------------------------
# Simulator singleton (thread-safe)
# ---------------------------------------------------------------------------


class SimulatorService:
    """
    Thread-safe wrapper around the GPU raytracing simulator.

    Initializes the CUDA/OptiX resources once at startup and serializes
    simulation calls through a lock (the GPU pipeline is not re-entrant).
    """

    def __init__(self, mesh_dir: str):
        self._lock = threading.Lock()
        logger.info("Building scene from %s …", mesh_dir)
        self._world, self._materials = _build_scene(mesh_dir)
        self._simulator = rs.RaytracingUltrasoundSimulator(self._world, self._materials)
        self._ready = True
        logger.info("Simulator ready")

    @property
    def ready(self) -> bool:
        return self._ready

    def simulate(
        self,
        probe_type: str,
        position: list[float],
        rotation: list[float],
        sim_params: SimulationParams,
        probe_config: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Run a single simulation frame.

        Returns the normalized uint8 B-mode image and metadata dict.
        """
        # Build pose
        pose = rs.Pose(
            np.array(position, dtype=np.float32),
            np.array(rotation, dtype=np.float32),
        )

        # Build probe with (optionally overridden) geometry
        spec = _PROBE_DEFAULTS[probe_type]
        kwargs = dict(spec["kwargs"])  # copy defaults
        if probe_config:
            # Validate overrides against the probe's Pydantic model
            model_cls = spec["params_model"]
            merged = model_cls(**{**kwargs, **probe_config})
            kwargs = merged.model_dump()
        probe = spec["cls"](pose, **kwargs)

        # Build SimParams
        sp = rs.SimParams()
        sp.conv_psf = sim_params.conv_psf
        sp.buffer_size = sim_params.buffer_size
        sp.t_far = sim_params.t_far
        sp.median_clip_filter = sim_params.median_clip_filter
        sp.b_mode_size = tuple(sim_params.b_mode_size)
        sp.enable_cuda_timing = True
        if sim_params.contact_epsilon is not None:
            sp.contact_epsilon = sim_params.contact_epsilon

        # Run on GPU (serialized)
        t0 = time.perf_counter()
        with self._lock:
            b_mode = self._simulator.simulate(probe, sp)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Normalize
        min_val = sim_params.dynamic_range_min
        max_val = sim_params.dynamic_range_max
        normalized = np.clip((b_mode - min_val) / (max_val - min_val), 0, 1)
        img_uint8 = (normalized * 255).astype(np.uint8)

        meta = {
            "probe_type": probe_type,
            "image_size": [int(img_uint8.shape[0]), int(img_uint8.shape[1])],
            "dynamic_range": [min_val, max_val],
            "sim_time_ms": round(elapsed_ms, 2),
        }

        return img_uint8, meta


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

_simulator_service: SimulatorService | None = None


def _get_sim() -> SimulatorService:
    if _simulator_service is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    return _simulator_service


def create_app(mesh_dir: str = "mesh") -> FastAPI:
    """Build and return the FastAPI application."""

    app = FastAPI(
        title="Ultrasound Simulation REST API",
        description=(
            "GPU-accelerated ray-tracing ultrasound simulation service. "
            "Submit probe type, 6-DOF pose, and simulation parameters to receive "
            "a rendered B-mode image."
        ),
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Startup / shutdown
    # ------------------------------------------------------------------

    @app.on_event("startup")
    def startup():
        global _simulator_service
        _simulator_service = SimulatorService(mesh_dir)

    # ------------------------------------------------------------------
    # GET / — serve the web frontend
    # ------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index():
        """Serve the single-page frontend."""
        client_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "templates",
            "rest_client.html",
        )
        return FileResponse(client_path, media_type="text/html")

    # ------------------------------------------------------------------
    # GET /api/v1/health
    # ------------------------------------------------------------------

    @app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
    def health():
        """Service health check."""
        sim = _simulator_service
        return HealthResponse(
            status="ok" if sim and sim.ready else "degraded",
            mesh_loaded=sim is not None and sim.ready,
            gpu_available=sim is not None,
        )

    # ------------------------------------------------------------------
    # GET /api/v1/probe-types
    # ------------------------------------------------------------------

    @app.get("/api/v1/probe-types", response_model=list[ProbeTypeInfo], tags=["Configuration"])
    def list_probe_types():
        """List available probe types with their default geometry parameters."""
        result = []
        for name, spec in _PROBE_DEFAULTS.items():
            model = spec["params_model"]
            result.append(ProbeTypeInfo(name=name, defaults=model().model_dump()))
        return result

    # ------------------------------------------------------------------
    # GET /api/v1/sim-params
    # ------------------------------------------------------------------

    @app.get("/api/v1/sim-params", response_model=SimulationParams, tags=["Configuration"])
    def default_sim_params():
        """Return default simulation parameters."""
        return SimulationParams()

    # ------------------------------------------------------------------
    # POST /api/v1/simulate  — returns PNG image
    # ------------------------------------------------------------------

    @app.post(
        "/api/v1/simulate",
        responses={
            200: {
                "content": {"image/png": {}},
                "description": "Rendered B-mode ultrasound image.",
            }
        },
        tags=["Simulation"],
    )
    def simulate(req: SimulateRequest):
        """
        Run an ultrasound simulation and return the B-mode image as PNG.

        The response includes the following custom headers:
        - `X-Probe-Type` — probe type used
        - `X-Sim-Time-Ms` — GPU simulation time in milliseconds
        - `X-Image-Width` / `X-Image-Height` — output image dimensions
        """
        sim = _get_sim()

        try:
            img_uint8, meta = sim.simulate(
                probe_type=req.probe_type.value,
                position=req.pose.position,
                rotation=req.pose.rotation,
                sim_params=req.sim_params,
                probe_config=req.probe_config,
            )
        except Exception as exc:
            logger.exception("Simulation failed")
            raise HTTPException(status_code=500, detail=f"Simulation error: {exc}") from exc

        # Encode as PNG
        pil_img = Image.fromarray(img_uint8, mode="L")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)

        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={
                "X-Probe-Type": meta["probe_type"],
                "X-Sim-Time-Ms": str(meta["sim_time_ms"]),
                "X-Image-Width": str(meta["image_size"][1]),
                "X-Image-Height": str(meta["image_size"][0]),
            },
        )

    # ------------------------------------------------------------------
    # POST /api/v1/simulate/json  — returns image + metadata as JSON
    # ------------------------------------------------------------------

    @app.post("/api/v1/simulate/json", tags=["Simulation"])
    def simulate_json(req: SimulateRequest):
        """
        Run an ultrasound simulation and return the B-mode image as a
        base64-encoded PNG together with simulation metadata.
        """
        import base64

        sim = _get_sim()

        try:
            img_uint8, meta = sim.simulate(
                probe_type=req.probe_type.value,
                position=req.pose.position,
                rotation=req.pose.rotation,
                sim_params=req.sim_params,
                probe_config=req.probe_config,
            )
        except Exception as exc:
            logger.exception("Simulation failed")
            raise HTTPException(status_code=500, detail=f"Simulation error: {exc}") from exc

        # Encode image to base64 PNG
        pil_img = Image.fromarray(img_uint8, mode="L")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            **meta,
            "pose": {
                "position": req.pose.position,
                "rotation": req.pose.rotation,
            },
            "image_base64": img_b64,
            "image_format": "png",
        }

    return app


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="REST API Ultrasound Simulation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--mesh-dir", default="mesh", help="Path to mesh directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    import uvicorn

    app = create_app(mesh_dir=args.mesh_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
