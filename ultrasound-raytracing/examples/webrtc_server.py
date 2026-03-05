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
Real-time WebRTC Ultrasound Simulation Server

Streams GPU-accelerated ultrasound simulation frames to remote clients via WebRTC.
Clients send probe pose and parameter updates over a WebRTC DataChannel;
the server renders B-mode images on the GPU and streams them as a video track.

Usage:
    python webrtc_server.py [--host 0.0.0.0] [--port 80] [--mesh-dir mesh]

Architecture:
    - aiohttp serves the signaling endpoint (POST /offer) and static client page (GET /)
    - aiortc handles WebRTC peer connections with a video track + data channel
    - A dedicated simulation thread runs the GPU raytracing loop and pushes frames
      into an asyncio queue consumed by the video track
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, Optional

import numpy as np

# Add root to path so raysim is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import raysim.cuda as rs
from aiohttp import web
from aiortc import (
    MediaStreamTrack,
    RTCPeerConnection,
    RTCSessionDescription,
)
from av import VideoFrame

logger = logging.getLogger("webrtc_us")

# ---------------------------------------------------------------------------
# Scene setup (shared across all sessions — world/materials are read-only)
# ---------------------------------------------------------------------------

_MESH_DIR: str = "mesh"


def _build_scene(mesh_dir: str) -> tuple[rs.World, rs.Materials]:
    """Build the simulation world and materials. Called once at startup."""
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

    for obj_file, mat_name in mesh_specs:
        path = os.path.join(mesh_dir, obj_file)
        if os.path.exists(path):
            idx = materials.get_index(mat_name)
            world.add(rs.Mesh(path, idx))
            logger.info("Loaded mesh %s (material=%s)", obj_file, mat_name)
        else:
            logger.warning("Mesh file not found, skipping: %s", path)

    return world, materials


# ---------------------------------------------------------------------------
# Per-session simulation state
# ---------------------------------------------------------------------------

# Default probe configurations (matching existing server.py)
_DEFAULT_POSITION = np.array([-14, -122, 72], dtype=np.float32)
_DEFAULT_ROTATION = np.array([np.deg2rad(-90), np.deg2rad(180), np.deg2rad(0)], dtype=np.float32)

_PROBE_DEFAULTS: Dict[str, dict] = {
    "curvilinear": dict(
        cls=rs.CurvilinearProbe,
        kwargs=dict(
            num_elements_x=256,
            sector_angle=73.0,
            radius=45.0,
            frequency=5.0,
            elevational_height=7.0,
            num_el_samples=10,
        ),
    ),
    "linear": dict(
        cls=rs.LinearArrayProbe,
        kwargs=dict(
            num_elements_x=256,
            width=50.0,
            frequency=7.5,
            elevational_height=5.0,
            num_el_samples=10,
        ),
    ),
    "phased": dict(
        cls=rs.PhasedArrayProbe,
        kwargs=dict(
            num_elements_x=128,
            width=20.0,
            sector_angle=90.0,
            frequency=3.5,
            elevational_height=5.0,
            num_el_samples=10,
        ),
    ),
}


@dataclass
class SessionState:
    """Mutable state for a single WebRTC session."""

    active_probe_type: str = "curvilinear"
    probes: Dict[str, rs.BaseProbe] = field(default_factory=dict)
    sim_params: rs.SimParams = field(default_factory=rs.SimParams)
    target_fps: int = 30
    running: bool = True
    # Latest pose request (set by data channel, consumed by sim thread)
    _pending_pose: Optional[np.ndarray] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    # Event signalled when state changes (pose, probe, params) so the sim
    # thread only renders when there is something new.
    _dirty: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self):
        # Build probes
        for name, spec in _PROBE_DEFAULTS.items():
            pose = rs.Pose(_DEFAULT_POSITION.copy(), _DEFAULT_ROTATION.copy())
            self.probes[name] = spec["cls"](pose, **spec["kwargs"])

        # Configure sim defaults
        self.sim_params.conv_psf = True
        self.sim_params.buffer_size = 4096
        self.sim_params.t_far = 180.0
        self.sim_params.enable_cuda_timing = False
        self.sim_params.median_clip_filter = False
        self.sim_params.b_mode_size = (512, 512)

    @property
    def active_probe(self) -> rs.BaseProbe:
        return self.probes[self.active_probe_type]

    def set_pending_pose(self, pose_array: np.ndarray):
        with self._lock:
            self._pending_pose = pose_array
        self.mark_dirty()

    def consume_pending_pose(self) -> Optional[np.ndarray]:
        with self._lock:
            p = self._pending_pose
            self._pending_pose = None
            return p

    def mark_dirty(self):
        """Signal the sim thread that something changed and a new frame is needed."""
        self._dirty.set()

    def wait_for_change(self, timeout: float) -> bool:
        """Block until state changes or timeout expires. Returns True if signalled."""
        triggered = self._dirty.wait(timeout=timeout)
        self._dirty.clear()
        return triggered


# ---------------------------------------------------------------------------
# Video track that streams simulation frames
# ---------------------------------------------------------------------------


class UltrasoundVideoTrack(MediaStreamTrack):
    """
    A MediaStreamTrack that streams the latest simulation frame.

    The simulation thread sets `latest_frame` whenever a new render is ready.
    `recv()` always returns immediately with the most recent frame (or a
    black placeholder), so aiortc's sender never stalls.
    """

    kind = "video"
    TARGET_FPS = 30

    def __init__(self):
        super().__init__()
        self._frame_count = 0
        self._start: Optional[float] = None
        # Start with a small black placeholder; replaced on first sim result
        black = np.zeros((64, 64, 3), dtype=np.uint8)
        self._latest_frame: VideoFrame = VideoFrame.from_ndarray(black, format="rgb24")
        self._lock = threading.Lock()

    def set_latest_frame(self, frame: VideoFrame):
        """Called from the simulation thread with a new rendered frame."""
        with self._lock:
            self._latest_frame = frame

    async def recv(self) -> VideoFrame:
        if self._start is None:
            self._start = time.time()

        # Pace to TARGET_FPS — sleep until the next frame is due
        self._frame_count += 1
        target_time = self._start + self._frame_count / self.TARGET_FPS
        wait = target_time - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        with self._lock:
            frame = self._latest_frame

        # Stamp for aiortc
        frame.pts = self._frame_count
        frame.time_base = Fraction(1, self.TARGET_FPS)
        return frame


# ---------------------------------------------------------------------------
# Simulation loop (runs in a dedicated thread)
# ---------------------------------------------------------------------------


def _simulation_loop(
    session: SessionState,
    mesh_dir: str,
    video_track: UltrasoundVideoTrack,
):
    """
    Blocking simulation loop intended to run in a background thread.

    Builds a fresh World + Simulator per session so that OptiX acceleration
    structures are not shared/consumed across reconnects.
    """
    min_val, max_val = -60.0, 0.0

    # Build scene AND simulator on THIS thread (CUDA context + OptiX ownership)
    world, materials = _build_scene(mesh_dir)
    simulator = rs.RaytracingUltrasoundSimulator(world, materials)
    logger.info("Simulation thread started (scene + simulator created on sim thread)")

    # Render one initial frame so the client sees something immediately
    need_render = True

    while session.running:
        if not need_render:
            # Block until a pose/param change arrives (true idle — no GPU work)
            signalled = session.wait_for_change(timeout=2.0)
            if not signalled:
                # Pure timeout, nothing changed — loop back and wait again
                continue
            # Something changed — check what
            pending = session.consume_pending_pose()
            if pending is not None:
                pos = np.array(pending[:3], dtype=np.float32)
                rot = np.array(pending[3:6], dtype=np.float32)
                session.active_probe.set_pose(rs.Pose(pos, rot))
            # Either a pose changed or probe/params changed — render either way
            need_render = True

        # Run GPU simulation
        try:
            b_mode = simulator.simulate(session.active_probe, session.sim_params)
        except Exception:
            logger.exception("Simulation error")
            time.sleep(0.1)
            need_render = False
            continue

        # Normalize to uint8 grayscale
        normalized = np.clip((b_mode - min_val) / (max_val - min_val), 0, 1)
        gray = (normalized * 255).astype(np.uint8)

        # Convert grayscale to RGB (WebRTC/VP8 expects color frames)
        h, w = gray.shape
        rgb = np.stack([gray, gray, gray], axis=-1)

        # Build an av.VideoFrame and hand it to the track
        frame = VideoFrame.from_ndarray(rgb, format="rgb24")
        video_track.set_latest_frame(frame)

        need_render = False

    logger.info("Simulation thread stopped")


# ---------------------------------------------------------------------------
# Data-channel message handler
# ---------------------------------------------------------------------------


def _handle_datachannel_message(session: SessionState, raw: str, channel):
    """Process a JSON message received on the WebRTC data channel."""
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON on data channel: %s", raw[:200])
        return

    action = msg.get("action")

    if action == "set_pose":
        # Expects {"action": "set_pose", "pose": [x,y,z,rx,ry,rz]}
        pose = msg.get("pose")
        if pose and len(pose) == 6:
            session.set_pending_pose(np.array(pose, dtype=np.float32))

    elif action == "set_pose_delta":
        # Expects {"action": "set_pose_delta", "delta": [dx,dy,dz,drx,dry,drz]}
        delta = msg.get("delta")
        if delta and len(delta) == 6:
            current = session.active_probe.get_pose()
            new_pos = current.position + np.array(delta[:3], dtype=np.float32)
            new_rot = current.rotation + np.array(delta[3:6], dtype=np.float32)
            new_pose = np.concatenate([new_pos, new_rot])
            session.set_pending_pose(new_pose)

    elif action == "set_probe_type":
        ptype = msg.get("probe_type", "curvilinear")
        if ptype in session.probes:
            session.active_probe_type = ptype
            session.mark_dirty()
            channel.send(json.dumps({"event": "probe_changed", "probe_type": ptype}))

    elif action == "set_sim_params":
        params = msg.get("params", {})
        if "median_clip_filter" in params:
            session.sim_params.median_clip_filter = bool(params["median_clip_filter"])
        if "b_mode_size" in params:
            sz = params["b_mode_size"]
            if isinstance(sz, list) and len(sz) == 2:
                session.sim_params.b_mode_size = (int(sz[0]), int(sz[1]))
        if "t_far" in params:
            session.sim_params.t_far = float(params["t_far"])
        if "target_fps" in params:
            session.target_fps = max(1, min(60, int(params["target_fps"])))
        session.mark_dirty()
        channel.send(json.dumps({"event": "params_updated"}))

    elif action == "get_pose":
        current = session.active_probe.get_pose()
        pose_list = current.position.tolist() + current.rotation.tolist()
        channel.send(json.dumps({
            "event": "pose",
            "pose": pose_list,
            "probe_type": session.active_probe_type,
        }))

    elif action == "get_probe_types":
        channel.send(json.dumps({
            "event": "probe_types",
            "types": list(session.probes.keys()),
        }))

    else:
        logger.warning("Unknown data channel action: %s", action)


# ---------------------------------------------------------------------------
# aiohttp signaling endpoints
# ---------------------------------------------------------------------------

_peer_connections: set[RTCPeerConnection] = set()


async def offer_handler(request: web.Request) -> web.Response:
    """
    POST /offer — WebRTC signaling endpoint.

    Expects JSON: {"sdp": "...", "type": "offer"}
    Returns JSON: {"sdp": "...", "type": "answer"}
    """
    body = await request.json()
    offer = RTCSessionDescription(sdp=body["sdp"], type=body["type"])

    pc = RTCPeerConnection()
    _peer_connections.add(pc)

    loop = asyncio.get_event_loop()
    mesh_dir: str = request.app["mesh_dir"]

    # Per-session state
    session = SessionState()

    # Create the video track
    video_track = UltrasoundVideoTrack()
    pc.addTrack(video_track)

    # Handle data channel
    @pc.on("datachannel")
    def on_datachannel(channel):
        logger.info("Data channel opened: %s", channel.label)

        @channel.on("message")
        def on_message(message):
            _handle_datachannel_message(session, message, channel)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state: %s", pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            session.running = False
            _peer_connections.discard(pc)
            await pc.close()

    # Start simulation thread
    sim_thread = threading.Thread(
        target=_simulation_loop,
        args=(session, mesh_dir, video_track),
        daemon=True,
        name="sim-thread",
    )
    sim_thread.start()

    # Complete SDP negotiation
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    })


async def index_handler(request: web.Request) -> web.Response:
    """GET / — serve the demo client page."""
    client_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "templates",
        "webrtc_client.html",
    )
    return web.FileResponse(client_path)


async def on_shutdown(app: web.Application):
    """Clean up all peer connections on server shutdown."""
    coros = [pc.close() for pc in _peer_connections]
    await asyncio.gather(*coros)
    _peer_connections.clear()


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(mesh_dir: str = "mesh") -> web.Application:
    """Build the aiohttp application."""
    app = web.Application()
    app.on_shutdown.append(on_shutdown)

    # Store mesh directory — each session builds its own world/simulator
    app["mesh_dir"] = mesh_dir

    # Routes
    app.router.add_get("/", index_handler)
    app.router.add_post("/offer", offer_handler)

    # Serve static files from templates/
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    if os.path.isdir(templates_dir):
        app.router.add_static("/static/", templates_dir, name="static")

    return app


def main():
    parser = argparse.ArgumentParser(description="WebRTC Ultrasound Simulation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=80, help="Bind port")
    parser.add_argument("--mesh-dir", default="mesh", help="Path to mesh directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Suppress noisy mDNS resolution warnings from aioice
    logging.getLogger("aioice").setLevel(logging.WARNING)

    global _MESH_DIR
    _MESH_DIR = args.mesh_dir

    app = create_app(mesh_dir=args.mesh_dir)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
