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

"""One frame's vessel geometry and contrast. numpy only, so it imports without ``warp-lang``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class VesselState:
    """One frame's vessel geometry and contrast, per environment.

    Args:
        points_mm: Node positions, ``(envs, nodes, 3)``, in the frame the rays are expressed in.
            These come from the deformation solver, so they move; everything else is indexed to match.
        edges: Node index pairs, ``(segments, 2)``, shared across environments because the
            topology is the anatomy's and does not deform.
        radii_mm: Lumen radius per node, ``(envs, nodes)`` or ``(nodes,)`` when undeformed.
        mu_per_mm: Attenuation the contrast adds over the carve's fill, per node, ``(envs,
            nodes)`` or ``(nodes,)``. Supplied by the transport model.
    """

    points_mm: np.ndarray
    edges: np.ndarray
    radii_mm: np.ndarray
    mu_per_mm: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points_mm, dtype=np.float32)
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"points_mm must be (envs, nodes, 3), got {points.shape}")
        envs, nodes = points.shape[0], points.shape[1]
        edges = np.asarray(self.edges, dtype=np.int32).reshape(-1, 2)
        if edges.size and (edges.min() < 0 or edges.max() >= nodes):
            raise ValueError(f"edges index outside the {nodes} supplied nodes")
        object.__setattr__(self, "points_mm", points)
        object.__setattr__(self, "edges", edges)
        for name in ("radii_mm", "mu_per_mm"):
            values = np.asarray(getattr(self, name), dtype=np.float32)
            # Length-checked before broadcasting, so a wrong-sized array reports which field is
            # wrong rather than letting numpy's remapped-shape message surface instead.
            if values.ndim == 1 and len(values) == nodes:
                values = np.broadcast_to(values, (envs, nodes))
            if values.shape != (envs, nodes):
                raise ValueError(f"{name} must be ({envs}, {nodes}) or ({nodes},), got {values.shape}")
            object.__setattr__(self, name, np.ascontiguousarray(values))


__all__ = ["VesselState"]
