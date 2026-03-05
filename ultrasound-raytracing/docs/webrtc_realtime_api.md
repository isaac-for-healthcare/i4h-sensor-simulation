# WebRTC Real-time Ultrasound Simulation

Stream GPU-accelerated ultrasound simulation to a remote browser in real-time over WebRTC.

## Architecture

```
┌────────────────────────────── GPU Server ──────────────────────────────┐
│                                                                        │
│  aiohttp (/offer)          Sim Thread (GPU)        aiortc (VP8/RTP)   │
│  ┌──────────────┐    ┌───────────────────────┐    ┌────────────────┐  │
│  │  Signaling   │    │  raysim.cuda           │    │  WebRTC Video  │──┼──► UDP
│  │  SDP offer/  │    │  simulate() on GPU     │───▶│  Track         │  │
│  │  answer      │    │  → numpy → VideoFrame  │    │  VP8 encode    │  │
│  └──────────────┘    └───────────▲────────────┘    └────────────────┘  │
│                                  │                                     │
│  DataChannel (JSON)              │                                     │
│  ◄── pose, probe type, params ───┘                                     │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────── Browser Client ─────────────────────────────┐
│  <video> element ◄── WebRTC stream                                    │
│  DataChannel ──► pose/param JSON messages                             │
│  Keyboard (WASD/Arrows) + Sliders for probe control                   │
└────────────────────────────────────────────────────────────────────────┘
```

### Why WebRTC over HTTP?

| Metric               | HTTP + PNG (existing) | WebRTC (this)      |
|-----------------------|-----------------------|--------------------|
| Round-trip latency    | 80–150 ms             | 20–40 ms           |
| Protocol overhead     | TCP + HTTP headers    | UDP + RTP          |
| Frame encoding        | PNG (lossless, slow)  | VP8 (hw-friendly)  |
| Browser decode        | JS Image() decode     | Native HW decode   |
| Control channel       | Separate HTTP calls   | Same connection DC |
| Adaptive quality      | None                  | Built-in BWE       |

## Requirements

```bash
pip install aiortc aiohttp av numpy
```

`raysim` must already be built and importable (see main project README).

## Quick Start

```bash
cd ultrasound-raytracing

# Start the server (default port 8080)
python examples/webrtc_server.py --mesh-dir mesh

# Open in browser
# http://<server-ip>:8080
```

### Server Options

```
--host        Bind address (default: 0.0.0.0)
--port        Bind port (default: 8080)
--mesh-dir    Path to mesh OBJ files (default: mesh)
-v/--verbose  Enable debug logging
```

## Client Controls

### Sliders (sidebar)
- **X/Y/Z position** — translate the probe
- **Roll/Pitch/Yaw** — rotate the probe
- **Max depth** — `t_far` simulation parameter
- **Target FPS** — server-side frame rate cap

### Keyboard
| Key         | Action           |
|-------------|------------------|
| `W` / `S`   | Move Y axis      |
| `A` / `D`   | Move X axis      |
| `Q` / `E`   | Move Z axis      |
| `↑` / `↓`   | Pitch            |
| `←` / `→`   | Yaw              |
| `[` / `]`   | Roll             |

### Probe types
Click **Curvilinear**, **Linear**, or **Phased** in the sidebar to switch.

## DataChannel Protocol

All messages are JSON sent over the WebRTC DataChannel labeled `"control"`.

### Client → Server

```jsonc
// Absolute pose
{ "action": "set_pose", "pose": [x, y, z, rx, ry, rz] }

// Relative pose delta
{ "action": "set_pose_delta", "delta": [dx, dy, dz, drx, dry, drz] }

// Switch probe
{ "action": "set_probe_type", "probe_type": "linear" }

// Update sim parameters
{ "action": "set_sim_params", "params": { "t_far": 200, "target_fps": 25 } }

// Request current pose
{ "action": "get_pose" }

// List probe types
{ "action": "get_probe_types" }
```

### Server → Client

```jsonc
// Pose response
{ "event": "pose", "pose": [x, y, z, rx, ry, rz], "probe_type": "curvilinear" }

// Probe changed confirmation
{ "event": "probe_changed", "probe_type": "linear" }

// Params updated confirmation
{ "event": "params_updated" }

// Available probe types
{ "event": "probe_types", "types": ["curvilinear", "linear", "phased"] }
```

## Deployment Notes

### Running behind a reverse proxy / NAT

For LAN deployments, the default STUN-less configuration works fine. For WAN:

1. Configure TURN/STUN servers in `RTCPeerConnection` configuration
2. Or use a VPN / direct IP route to the GPU server

### Multi-client

Each WebRTC connection spawns its own simulation thread with independent state.
The GPU simulator instances share the read-only world/materials. On a single GPU,
3–5 concurrent sessions at 30 FPS is realistic depending on mesh complexity.

### Docker

Add to the existing Dockerfile:

```dockerfile
RUN pip install aiortc aiohttp av
EXPOSE 8080
CMD ["python", "examples/webrtc_server.py", "--mesh-dir", "mesh"]
```
