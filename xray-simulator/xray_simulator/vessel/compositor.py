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

"""Carve, rest state and subtraction for a vessel rendered rather than baked into the scan.

Turning this on changes what the volume renderer is handed: :attr:`VesselContrast.mask_volume`
is the scan with its lumen removed, so the volume render *is* the contrast-free mask frame. The
vessel is then composited back per frame with :meth:`VesselContrast.absorber`, and a DSA frame is
the live transmission divided by the mask's (:func:`subtract`). Without it nothing here runs and
the scan renders exactly as before.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..absorbers.rays import DetectorRays
from .attenuation import PosedVessel, VesselAttenuation, VesselState
from .carve import CarveResult, carve_lumen, node_contrast_mu_per_mm
from .centerline import VesselTree
from .contrast import ContrastAgent, HuToMu

#: Iodine concentration a selective injection reaches in the lumen, mg I/mL. Used as the default
#: saturation point of the subtracted channel.
DSA_REFERENCE_MG_PER_ML = 60.0


class VesselContrast:
    """A contrast lumen carved out of a volume and composited back as moving geometry.

    Args:
        mu_kji: Attenuation volume, not modified.
        ijk_to_world_mm: Voxel-centre index to world millimetres. Every position here -- the
            tree, displacements, the rays -- is in that world frame.
        tree: The vessel's centerline, already smoothed if it came off a voxel grid.
        curve: The HU-to-mu curve ``mu_kji`` was built with, so injected contrast lands on the
            same calibration as the anatomy; :meth:`HuToMu.from_mapping` adapts a preset.
        vessel_mask_kji: Optional segmentation, for the coverage report or a union carve.
        supersample: Sub-voxel samples per axis for the carve boundary.
        include_segmentation: Carve the tube's union with ``vessel_mask_kji``.
        agent: Contrast medium; defaults to neat iodinated contrast.
        device: Warp device for the chord integrator.

    Raises:
        ValueError: If the carve is empty, or any input is inconsistent with the volume.
    """

    def __init__(
        self,
        mu_kji: np.ndarray,
        ijk_to_world_mm: np.ndarray,
        tree: VesselTree,
        curve: HuToMu,
        *,
        vessel_mask_kji: np.ndarray | None = None,
        supersample: int = 1,
        include_segmentation: bool = False,
        agent: ContrastAgent | None = None,
        device: str | None = None,
    ) -> None:
        self._carve = carve_lumen(
            mu_kji,
            ijk_to_world_mm,
            tree,
            vessel_mask_kji=vessel_mask_kji,
            supersample=int(supersample),
            include_segmentation=include_segmentation,
        )
        # Per node rather than the tree's mean, because a real lumen's attenuation varies by a third
        # across the anatomy and a single value is what a rest-state residual would be made of.
        self._rest = VesselState(
            points_mm=tree.points_mm[None].astype(np.float32),
            edges=tree.edges,
            radii_mm=tree.radii_mm.astype(np.float32),
            mu_per_mm=node_contrast_mu_per_mm(mu_kji, ijk_to_world_mm, tree, self._carve).astype(np.float32),
        )
        self._curve = curve
        self._agent = agent if agent is not None else ContrastAgent()
        self._attenuation = VesselAttenuation(device=device)

    @property
    def carve(self) -> CarveResult:
        """What was taken out of the volume, and the measurements describing it."""
        return self._carve

    @property
    def mask_volume(self) -> np.ndarray:
        """The volume to render: the scan with its lumen removed, i.e. the contrast-free mask."""
        return self._carve.mu_volume

    @property
    def rest_state(self) -> VesselState:
        """The vessel where the scan found it, at the contrast the scan was taken with.

        Compositing this onto :attr:`mask_volume` reproduces the original image, because the same
        geometry is added back that was taken out.
        """
        return self._rest

    @property
    def curve(self) -> HuToMu:
        return self._curve

    @property
    def agent(self) -> ContrastAgent:
        return self._agent

    def mu_excess(self, concentration_mg_per_ml: np.ndarray) -> np.ndarray:
        """Attenuation iodine at these concentrations adds over the carve's fill material."""
        return self._agent.mu_excess(
            np.asarray(concentration_mg_per_ml, dtype=np.float64),
            baseline_mu=self._carve.fill_mu_per_mm,
            curve=self._curve,
        )

    def state(
        self,
        displacement_mm: np.ndarray | None = None,
        concentration_mg_per_ml: np.ndarray | None = None,
    ) -> VesselState:
        """The rest vessel moved and refilled for one frame.

        Displacement is added to the rest state rather than replacing it, so the composited
        centerline keeps the smoothing it was built with.

        Args:
            displacement_mm: Movement from rest, ``(envs, nodes, 3)``. ``None`` leaves it at rest.
            concentration_mg_per_ml: Iodine per node, ``(envs, nodes)`` or ``(nodes,)``. ``None``
                keeps the scan's own contrast.
        """
        points_mm = self._rest.points_mm
        if displacement_mm is not None:
            shift = np.asarray(displacement_mm, dtype=np.float32)
            points_mm = points_mm + shift.reshape(-1, *points_mm.shape[1:])
        mu_per_mm = self._rest.mu_per_mm
        if concentration_mg_per_ml is not None:
            mu_per_mm = self.mu_excess(concentration_mg_per_ml).astype(np.float32)
        envs = points_mm.shape[0]
        if mu_per_mm.ndim == 2 and mu_per_mm.shape[0] != envs:
            mu_per_mm = np.broadcast_to(mu_per_mm, (envs, mu_per_mm.shape[1]))
        radii_mm = np.broadcast_to(self._rest.radii_mm[:1], (envs, self._rest.radii_mm.shape[1]))
        return replace(self._rest, points_mm=points_mm, radii_mm=radii_mm, mu_per_mm=mu_per_mm)

    def absorber(self, state: VesselState | None = None) -> PosedVessel:
        """A vessel state as an :class:`~xray_simulator.absorbers.Absorber`, at rest by default."""
        return self._attenuation.posed(self._rest if state is None else state)

    def line_integral(self, state: VesselState, rays: DetectorRays, *, as_torch: bool = False) -> Any:
        """The vessel's :math:`\\int \\mu \\, ds` along ``rays``, ``(envs, height, width)``."""
        return self._attenuation.line_integral(state, rays, as_torch=as_torch)

    def subtracted_window(
        self, rays: DetectorRays, reference_mg_per_ml: float = DSA_REFERENCE_MG_PER_ML
    ) -> tuple[float, float]:
        """Log-domain display window for the subtracted channel, fixed up front.

        A subtracted frame is almost all unattenuated, so a window fitted to the anatomy renders
        it blank, and fitting it adaptively is worse: the first frame is acquired before the
        injection arrives, so the window lands on an empty image and sticks there. Deciding it
        once also keeps the mapping identical across runs, which matters when a policy learns
        from these pixels.

        The reference is the agent, not the scan: many studies are acquired with little or no
        contrast, and scaling to what the study contains would put an angiographic injection far
        off the top of the range. What saturates is the longest lumen chord in view, filled at
        ``reference_mg_per_ml``.

        Returns:
            ``(0, line_integral)`` mapping to unattenuated and to a saturated pixel.
        """
        opacified = replace(self._rest, mu_per_mm=np.ones_like(self._rest.mu_per_mm))
        path_mm = float(np.max(self._attenuation.line_integral(opacified, rays)))
        mu_per_mm = float(self.mu_excess(np.asarray([reference_mg_per_ml]))[0])
        return (0.0, max(mu_per_mm * path_mm, 1.0e-6))


def subtract(live_transmission: Any, mask_transmission: Any) -> Any:
    """A DSA frame: the live transmission divided by the contrast-free mask's.

    Dividing transmissions is subtracting line integrals, which is what a DSA system does in the
    log domain. Everything static cancels identically -- including the carve's own residual, which
    can be several times the wall-motion signal and would otherwise dominate it.
    """
    if isinstance(mask_transmission, np.ndarray):
        return live_transmission / np.maximum(mask_transmission, 1.0e-8)
    return live_transmission / mask_transmission.clamp(min=1.0e-8)


__all__ = ["DSA_REFERENCE_MG_PER_ML", "VesselContrast", "subtract"]
