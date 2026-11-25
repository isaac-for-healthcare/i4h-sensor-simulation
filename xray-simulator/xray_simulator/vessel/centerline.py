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

"""A vessel tree as a centerline graph with a lumen radius at every node."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

#: Laplacian passes applied to a centerline before anything geometric is built on it.
#: Centerlines extracted from a segmentation sit at voxel centres, so the path is a staircase
#: that turns a mean 32 degrees per node on a 1.5 mm aorta -- discretisation, not anatomy, and it
#: wrecks any chain of convex primitives laid along it because the joint planes then lean further
#: than a segment is long. Two passes bring the mean turn to 7 degrees while moving nodes 0.37 mm
#: on average and 1.12 mm at worst, under one voxel. More passes keep straightening but start
#: cutting the corners off genuine bends.
SMOOTHING_PASSES = 2
_SMOOTHING_RATE = 0.5


def smooth_centerline(points_mm: np.ndarray, edges: np.ndarray, *, passes: int = SMOOTHING_PASSES) -> np.ndarray:
    """Relax a voxel-stepped centerline towards the smooth curve it samples.

    Each pass moves every node a fraction of the way to the mean of its graph neighbours. Degree
    is taken from the edge list, so bifurcations and endpoints need no special case: an endpoint
    has one neighbour and simply drifts towards it.
    """
    points = np.asarray(points_mm, dtype=np.float64).copy()
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if not passes or not edges.size:
        return points
    degree = np.bincount(edges.reshape(-1), minlength=len(points)).astype(np.float64)
    degree[degree == 0.0] = 1.0
    for _ in range(int(passes)):
        neighbour_sum = np.zeros_like(points)
        np.add.at(neighbour_sum, edges[:, 0], points[edges[:, 1]])
        np.add.at(neighbour_sum, edges[:, 1], points[edges[:, 0]])
        points += _SMOOTHING_RATE * (neighbour_sum / degree[:, None] - points)
    return points


@dataclass(frozen=True, slots=True)
class VesselTree:
    """Centerline nodes, their connectivity and the lumen radius at each, in millimetres.

    Positions are in whatever frame the volume's ``ijk_to_world_mm`` maps into, so the carve and
    the compositing agree on where the vessel is without a second transform.

    Attributes:
        points_mm: Node positions, ``(nodes, 3)``.
        edges: Node index pairs, ``(segments, 2)``. Orientation does not matter.
        radii_mm: Lumen radius per node, ``(nodes,)``, strictly positive.
    """

    points_mm: np.ndarray
    edges: np.ndarray
    radii_mm: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points_mm, dtype=np.float64)
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError(f"points_mm must be (nodes, 3), got {points.shape}")
        radii = np.asarray(self.radii_mm, dtype=np.float64).reshape(-1)
        if len(radii) != len(points):
            raise ValueError(
                f"radii_mm has {len(radii)} entries but points_mm has {len(points)}; one radius per node is required"
            )
        if not np.all(radii > 0.0):
            raise ValueError("radii_mm must be strictly positive")
        edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        if edges.size and (edges.min() < 0 or edges.max() >= len(points)):
            raise ValueError(f"edges index outside the {len(points)} supplied nodes")
        object.__setattr__(self, "points_mm", points)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "radii_mm", radii)

    def smoothed(self, passes: int = SMOOTHING_PASSES) -> VesselTree:
        """The same tree with its voxel staircase relaxed; see :func:`smooth_centerline`."""
        return replace(self, points_mm=smooth_centerline(self.points_mm, self.edges, passes=passes))

    def with_radius_scale(self, scale: float) -> VesselTree:
        """The same tree with every radius scaled, widening or narrowing the lumen."""
        return replace(self, radii_mm=self.radii_mm * float(scale))


__all__ = ["SMOOTHING_PASSES", "VesselTree", "smooth_centerline"]
