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

"""Launch plumbing shared by the Warp chord integrators."""

from __future__ import annotations

from typing import Any

import warp as wp

from .rays import DetectorRays


def warp_rays(rays: DetectorRays) -> tuple[wp.vec3, wp.vec3, wp.vec3, wp.vec3]:
    """Source, first pixel and pixel steps as kernel arguments."""
    return tuple(
        wp.vec3(*(float(value) for value in vector))
        for vector in (rays.source_mm, rays.first_pixel_mm, rays.column_step_mm, rays.row_step_mm)
    )


def torch_stream(device: Any, as_torch: bool) -> Any:
    """Torch's current stream on ``device``, so a copy handed to torch cannot overtake the kernel."""
    if not (as_torch and device.is_cuda):
        return None
    return wp.stream_from_torch(wp.device_to_torch(device))


def read_out(buffer: wp.array, *, as_torch: bool) -> Any:
    """Snapshot a reused output buffer.

    Copied because the buffer is overwritten next frame, and on a CPU device ``numpy()`` is a view
    onto it rather than a snapshot.
    """
    if as_torch:
        return wp.to_torch(buffer).clone()
    return buffer.numpy().copy()
