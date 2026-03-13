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

"""Differentiable DRR Renderer using Slang Automatic Differentiation.

This module implements a truly differentiable Digitally Reconstructed Radiograph
(DRR) renderer using Slang's compiler-level automatic differentiation. Unlike
finite-difference approaches, this provides:

1. **Exact Gradients**: Slang's autodiff computes analytical derivatives
2. **GPU Acceleration**: All computation runs on CUDA
3. **PyTorch Integration**: Seamless `torch.autograd.Function` wrapper
4. **Memory Efficient**: No need to store intermediate buffers for finite differences

Architecture:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DiffDRR with Slang Autodiff                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     ┌───────────────────────────────────────────────┐    │
│  │ PyTorch      │     │               Slang Shader                    │    │
│  │ Autograd     │◄────┤  [Differentiable] computePixelIntensity()     │    │
│  │              │     │                                               │    │
│  │ rotation ────┼────►│  ┌─────────────┐   ┌──────────────────────┐  │    │
│  │ translation ─┼────►│  │ Euler→R     │──►│ Ray Generation       │  │    │
│  │              │     │  └─────────────┘   └──────────────────────┘  │    │
│  │              │     │                            │                  │    │
│  │              │     │                            ▼                  │    │
│  │              │     │  ┌──────────────────────────────────────────┐│    │
│  │              │     │  │ Fixed-Step Ray-March + Trilinear Sample ││    │
│  │              │     │  │    ∫ μ(s) ds = Σ μᵢ · Δs                 ││    │
│  │              │     │  └──────────────────────────────────────────┘│    │
│  │              │     │                            │                  │    │
│  │              │     │                            ▼                  │    │
│  │   ∂L/∂θ  ◄───┼─────┤  ┌──────────────────────────────────────────┐│    │
│  │   ∂L/∂t  ◄───┼─────┤  │ Beer-Lambert: I = I₀·exp(-∫μds)         ││    │
│  │              │     │  │                                          ││    │
│  │   Image  ◄───┼─────┤  │ bwd_diff() → ∂I/∂θ, ∂I/∂t               ││    │
│  └──────────────┘     │  └──────────────────────────────────────────┘│    │
│                       └───────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

Usage:
    >>> from diffdrr_slang_renderer import SlangDiffDRRRenderer, SlangDiffDRRConfig
    >>>
    >>> # Initialize renderer
    >>> renderer = SlangDiffDRRRenderer(mu_volume, spacing_zyx_mm)
    >>>
    >>> # Forward pass only
    >>> image = renderer.render(rotation=[0, 0, 0], translation=[0, 0, 0])
    >>>
    >>> # With gradients (PyTorch integration)
    >>> torch_renderer = TorchSlangDiffDRR(mu_volume, spacing_zyx_mm)
    >>> rot = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    >>> trans = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    >>> image = torch_renderer(rot, trans)
    >>> loss = (image - target).pow(2).mean()
    >>> loss.backward()  # Gradients computed via Slang autodiff!
    >>> print(rot.grad, trans.grad)

Requirements:
    - slangpy >= 0.40
    - CUDA-capable GPU
    - PyTorch (optional, for autograd integration)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

# Optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    import slangpy
    SLANG_AVAILABLE = True
except ImportError:
    slangpy = None  # type: ignore
    SLANG_AVAILABLE = False


# Path to the Slang shader file
_SLANG_SHADER_PATH = Path(__file__).parent / "diffdrr_slang.slang"


@dataclass(frozen=True)
class SlangDiffDRRConfig:
    """Configuration for Slang-based differentiable DRR renderer.

    Attributes:
        det_height_px: Detector height in pixels.
        det_width_px: Detector width in pixels.
        pixel_spacing_mm: Pixel pitch on detector (mm).
        source_to_detector_mm: Source-to-detector distance (mm).
        source_to_isocenter_mm: Source-to-isocenter distance (mm).
        step_mm: Ray-marching step size (mm). Smaller = more accurate but slower.
        i0: Unattenuated X-ray intensity.
        normalize: If True, normalize output to [0, 1].
        invert: If True, invert so bone=white, air=black (clinical X-ray convention).
        eps: Numerical stability constant.
    """

    det_height_px: int = 512
    det_width_px: int = 512
    pixel_spacing_mm: float = 0.5
    source_to_detector_mm: float = 1020.0
    source_to_isocenter_mm: float = 510.0
    step_mm: float = 0.5
    i0: float = 1.0
    normalize: bool = True
    invert: bool = True  # Clinical convention: bone=white, air=black
    eps: float = 1e-8


class SlangDiffDRRRenderer:
    """Differentiable DRR Renderer using Slang's Automatic Differentiation.

    This renderer uses Slang's compiler-level autodiff to compute exact gradients
    of the rendered image with respect to pose parameters (rotation and translation).

    Key Features:
    - True autodiff (not finite differences)
    - GPU-accelerated via CUDA
    - Fixed-step ray marching with trilinear interpolation
    - Compatible with PyTorch's autograd system

    Example:
        >>> renderer = SlangDiffDRRRenderer(mu_volume, spacing_zyx_mm)
        >>>
        >>> # Forward only
        >>> image = renderer.render(rotation=[0, 0, 0], translation=[0, 0, 0])
        >>>
        >>> # Forward + backward
        >>> image, grads = renderer.render_with_gradients(
        ...     rotation=[5, 0, 0],
        ...     translation=[0, 0, 0],
        ...     grad_output=upstream_gradient,
        ... )
        >>> print(f"∂L/∂rotation: {grads['rotation']}")
        >>> print(f"∂L/∂translation: {grads['translation']}")
    """

    def __init__(
        self,
        mu_volume: np.ndarray,
        spacing_zyx_mm: tuple[float, float, float],
        cfg: SlangDiffDRRConfig = SlangDiffDRRConfig(),
    ):
        """Initialize the Slang differentiable DRR renderer.

        Args:
            mu_volume: 3D numpy array (Z, Y, X) of linear attenuation coefficients (mm^-1).
            spacing_zyx_mm: Voxel spacing in (Z, Y, X) order, in mm.
            cfg: Renderer configuration.

        Raises:
            RuntimeError: If Slang/slangpy is not available.
            ValueError: If mu_volume is not 3D.
        """
        if not SLANG_AVAILABLE:
            raise RuntimeError(
                "Slang is not available. Install with: pip install slangpy\n"
                "Requires slangpy >= 0.40"
            )

        if mu_volume.ndim != 3:
            raise ValueError(f"Expected mu_volume to be 3D; got shape={mu_volume.shape}")

        self._cfg = cfg
        self._spacing_zyx = spacing_zyx_mm
        self._vol_shape_zyx = mu_volume.shape

        # Convert spacing from ZYX to XYZ
        sz, sy, sx = spacing_zyx_mm
        self._spacing_xyz = (sx, sy, sz)

        # Store volume as contiguous float32
        self._mu_volume = np.ascontiguousarray(mu_volume.astype(np.float32))

        # Initialize Slang
        self._init_slang()

    def _init_slang(self):
        """Initialize Slang device and load the differentiable shader."""
        print("[SlangDiffDRR] Initializing...")

        # Create CUDA device
        self._device = slangpy.create_device(slangpy.DeviceType.cuda)
        # API changed: adapter_info.name -> info.adapter_name in slangpy 0.40+
        device_name = getattr(self._device.info, 'adapter_name', 'Unknown GPU')
        print(f"  Device: {device_name}")

        # Check shader exists
        if not _SLANG_SHADER_PATH.exists():
            raise FileNotFoundError(f"Slang shader not found: {_SLANG_SHADER_PATH}")

        # Load the Slang module with autodiff enabled
        self._module = slangpy.Module.load_from_file(
            self._device,
            str(_SLANG_SHADER_PATH),
            options={"defines": {"ENABLE_AUTODIFF": "1"}},
        )
        print(f"  Module: {_SLANG_SHADER_PATH.name}")

        # Get entry points for differentiable rendering
        self._forward_fn = self._module.find_function("renderDRR_forward")
        self._backward_fn = self._module.find_function("renderDRR_backward")
        self._normalize_fn = self._module.find_function("normalizeImage")
        print("  Functions loaded: forward, backward, normalize")

        # Create GPU resources
        self._create_resources()
        print("[SlangDiffDRR] Ready (Differentiable mode enabled)")

    def _create_resources(self):
        """Create GPU textures and buffers."""
        z, y, x = self._vol_shape_zyx
        cfg = self._cfg
        self._num_voxels = x * y * z

        # Volume texture (3D)
        self._mu_texture = self._device.create_texture(
            type=slangpy.TextureType.texture_3d,
            format=slangpy.Format.r32_float,
            width=x,
            height=y,
            depth=z,
            usage=slangpy.TextureUsage.shader_resource,
            data=self._mu_volume,
        )
        print(f"  Volume texture: {x}x{y}x{z}")

        # Sampler with trilinear interpolation
        sampler_desc = slangpy.SamplerDesc({
            "min_filter": slangpy.TextureFilteringMode.linear,
            "mag_filter": slangpy.TextureFilteringMode.linear,
            "mip_filter": slangpy.TextureFilteringMode.linear,
            "address_u": slangpy.TextureAddressingMode.clamp_to_edge,
            "address_v": slangpy.TextureAddressingMode.clamp_to_edge,
            "address_w": slangpy.TextureAddressingMode.clamp_to_edge,
        })
        self._sampler = self._device.create_sampler(sampler_desc)

        # Output image texture (2D)
        self._output_texture = self._device.create_texture(
            type=slangpy.TextureType.texture_2d,
            format=slangpy.Format.r32_float,
            width=cfg.det_width_px,
            height=cfg.det_height_px,
            usage=slangpy.TextureUsage.shader_resource | slangpy.TextureUsage.unordered_access,
        )

        # Gradient output texture (for upstream gradient ∂L/∂I)
        self._grad_output_texture = self._device.create_texture(
            type=slangpy.TextureType.texture_2d,
            format=slangpy.Format.r32_float,
            width=cfg.det_width_px,
            height=cfg.det_height_px,
            usage=slangpy.TextureUsage.shader_resource | slangpy.TextureUsage.unordered_access,
        )

        print(f"  Output: {cfg.det_width_px}x{cfg.det_height_px}")

        # Per-pixel gradient textures (kept for compatibility)
        self._grad_rotation_texture = self._device.create_texture(
            type=slangpy.TextureType.texture_2d,
            format=slangpy.Format.rgba32_float,
            width=cfg.det_width_px,
            height=cfg.det_height_px,
            usage=slangpy.TextureUsage.shader_resource | slangpy.TextureUsage.unordered_access,
        )

        self._grad_translation_texture = self._device.create_texture(
            type=slangpy.TextureType.texture_2d,
            format=slangpy.Format.rgba32_float,
            width=cfg.det_width_px,
            height=cfg.det_height_px,
            usage=slangpy.TextureUsage.shader_resource | slangpy.TextureUsage.unordered_access,
        )
        print(f"  Gradient textures: {cfg.det_width_px}x{cfg.det_height_px}")

    def _build_params(
        self,
        rotation: tuple[float, float, float],
        translation: tuple[float, float, float],
    ) -> tuple[dict, dict, dict]:
        """Build parameter dictionaries for shader dispatch."""
        sz, sy, sx = self._spacing_zyx
        z, y, x = self._vol_shape_zyx
        cfg = self._cfg

        vol_info = {
            "spacing": slangpy.float3(sx, sy, sz),
            "dimensions": slangpy.int3(x, y, z),
            "origin": slangpy.float3(0.0, 0.0, 0.0),
        }

        carm = {
            "sdd": float(cfg.source_to_detector_mm),
            "sid": float(cfg.source_to_isocenter_mm),
            "detectorSize": slangpy.float2(
                cfg.det_width_px * cfg.pixel_spacing_mm,
                cfg.det_height_px * cfg.pixel_spacing_mm,
            ),
            "detectorPixels": slangpy.int2(cfg.det_width_px, cfg.det_height_px),
            "pixelSpacing": float(cfg.pixel_spacing_mm),
        }

        pose = {
            "rotation": slangpy.float3(*rotation),
            "translation": slangpy.float3(*translation),
        }

        return vol_info, carm, pose

    def render(
        self,
        rotation: Union[tuple[float, float, float], np.ndarray] = (0.0, 0.0, 0.0),
        translation: Union[tuple[float, float, float], np.ndarray] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """Render a DRR image at the specified pose (forward pass only).

        Args:
            rotation: Euler angles (rx, ry, rz) in radians.
            translation: Translation (tx, ty, tz) in mm.

        Returns:
            2D float32 numpy array of shape (H, W).
        """
        # Convert to tuples
        if isinstance(rotation, np.ndarray):
            rotation = tuple(rotation.flatten()[:3].tolist())
        if isinstance(translation, np.ndarray):
            translation = tuple(translation.flatten()[:3].tolist())

        vol_info, carm, pose = self._build_params(rotation, translation)
        cfg = self._cfg

        # Dispatch forward kernel
        thread_count = slangpy.uint3(cfg.det_width_px, cfg.det_height_px, 1)

        self._forward_fn.dispatch(
            thread_count=thread_count,
            muVolume=self._mu_texture,
            volumeSampler=self._sampler,
            outputImage=self._output_texture,
            volInfo=vol_info,
            carm=carm,
            pose=pose,
            stepMM=float(cfg.step_mm),
            i0=float(cfg.i0),
        )

        # Read back result
        image = self._output_texture.to_numpy()
        image = image.reshape(cfg.det_height_px, cfg.det_width_px)

        # Normalize if requested
        if cfg.normalize:
            vmin = float(np.min(image))
            vmax = float(np.max(image))
            image = (image - vmin) / (vmax - vmin + cfg.eps)

        # Invert for clinical X-ray convention (bone=white, air=black)
        if cfg.invert:
            image = 1.0 - image

        return image.astype(np.float32)

    def render_with_gradients(
        self,
        rotation: Union[tuple[float, float, float], np.ndarray],
        translation: Union[tuple[float, float, float], np.ndarray],
        grad_output: Optional[np.ndarray] = None,
        max_steps: int = 2048,
    ) -> tuple[np.ndarray, dict]:
        """Render DRR and compute gradients via Slang autodiff.

        This uses Slang's automatic differentiation with custom backward derivatives
        for texture sampling to compute exact gradients of the rendered image
        with respect to pose parameters and optionally the volume.

        The implementation follows Slang's autodiff-texture example pattern:
        - Hardware texture sampling for fast forward pass
        - Software trilinear interpolation for backward pass (gradient flow)
        - Atomic fixed-point accumulation for thread-safe gradient updates

        Args:
            rotation: Euler angles (rx, ry, rz) in radians.
            translation: Translation (tx, ty, tz) in mm.
            grad_output: Upstream gradient ∂L/∂I. If None, uses ones (gradient of sum).
            max_steps: Maximum ray-march steps (for differentiable path).

        Returns:
            Tuple of (image, gradients_dict) where:
            - image: Rendered DRR as numpy array (H, W)
            - gradients_dict: {
                'rotation': np.ndarray of shape (3,),
                'translation': np.ndarray of shape (3,),
                'volume': np.ndarray of shape (Z, Y, X) - gradients w.r.t. mu volume
              }

        Example:
            >>> # Compute gradients for 2D/3D registration
            >>> img, grads = renderer.render_with_gradients(
            ...     rotation=[5, 0, 0],
            ...     translation=[10, 0, 0],
            ...     grad_output=2 * (synthetic - target),  # MSE gradient
            ... )
            >>> # Update pose
            >>> rotation -= lr * grads['rotation']
            >>> translation -= lr * grads['translation']
        """
        # Convert to tuples
        if isinstance(rotation, np.ndarray):
            rotation = tuple(rotation.flatten()[:3].tolist())
        if isinstance(translation, np.ndarray):
            translation = tuple(translation.flatten()[:3].tolist())

        cfg = self._cfg
        vol_info, carm, pose = self._build_params(rotation, translation)

        # Prepare upstream gradient
        if grad_output is None:
            grad_output = np.ones((cfg.det_height_px, cfg.det_width_px), dtype=np.float32)
        else:
            grad_output = np.ascontiguousarray(grad_output.astype(np.float32))

        # Upload gradient to texture
        grad_output_texture = self._device.create_texture(
            type=slangpy.TextureType.texture_2d,
            format=slangpy.Format.r32_float,
            width=cfg.det_width_px,
            height=cfg.det_height_px,
            usage=slangpy.TextureUsage.shader_resource,
            data=grad_output,
        )

        thread_count = slangpy.uint3(cfg.det_width_px, cfg.det_height_px, 1)

        # Step 1: Forward pass
        self._forward_fn.dispatch(
            thread_count=thread_count,
            muVolume=self._mu_texture,
            volumeSampler=self._sampler,
            outputImage=self._output_texture,
            volInfo=vol_info,
            carm=carm,
            pose=pose,
            stepMM=float(cfg.step_mm),
            i0=float(cfg.i0),
        )

        # Step 2: Backward pass - compute per-pixel gradients
        self._backward_fn.dispatch(
            thread_count=thread_count,
            muVolume=self._mu_texture,
            volumeSampler=self._sampler,
            gradOutput=grad_output_texture,
            gradRotation=self._grad_rotation_texture,
            gradTranslation=self._grad_translation_texture,
            volInfo=vol_info,
            carm=carm,
            pose=pose,
            stepMM=float(cfg.step_mm),
            i0=float(cfg.i0),
        )

        # Read back results
        image = self._output_texture.to_numpy().reshape(cfg.det_height_px, cfg.det_width_px)

        # Read per-pixel gradients and reduce to total gradients
        grad_rot_pixels = self._grad_rotation_texture.to_numpy()
        grad_rot_pixels = grad_rot_pixels.reshape(cfg.det_height_px, cfg.det_width_px, 4)
        grad_rotation = grad_rot_pixels[:, :, :3].sum(axis=(0, 1))  # Sum over all pixels

        grad_trans_pixels = self._grad_translation_texture.to_numpy()
        grad_trans_pixels = grad_trans_pixels.reshape(cfg.det_height_px, cfg.det_width_px, 4)
        grad_translation = grad_trans_pixels[:, :, :3].sum(axis=(0, 1))  # Sum over all pixels

        # Normalize if requested
        if cfg.normalize:
            vmin = float(np.min(image))
            vmax = float(np.max(image))
            image = (image - vmin) / (vmax - vmin + cfg.eps)

        # Invert for clinical X-ray convention (bone=white, air=black)
        if cfg.invert:
            image = 1.0 - image

        return image.astype(np.float32), {
            'rotation': grad_rotation.astype(np.float32),
            'translation': grad_translation.astype(np.float32),
        }

    @property
    def config(self) -> SlangDiffDRRConfig:
        """Return renderer configuration."""
        return self._cfg

    @property
    def volume_shape_zyx(self) -> tuple[int, int, int]:
        """Return volume shape in ZYX order."""
        return self._vol_shape_zyx

    @property
    def spacing_xyz(self) -> tuple[float, float, float]:
        """Return voxel spacing in XYZ order."""
        return self._spacing_xyz

    def __repr__(self) -> str:
        return (
            f"SlangDiffDRRRenderer(\n"
            f"  volume_shape={self._vol_shape_zyx},\n"
            f"  detector=({self._cfg.det_height_px}×{self._cfg.det_width_px}),\n"
            f"  step_mm={self._cfg.step_mm},\n"
            f"  sdd={self._cfg.source_to_detector_mm}mm\n"
            f")"
        )


# =============================================================================
# PyTorch Integration
# =============================================================================

if TORCH_AVAILABLE:

    class SlangDiffDRRFunction(torch.autograd.Function):
        """PyTorch autograd Function for Slang DiffDRR.

        This integrates Slang's automatic differentiation with PyTorch's autograd
        system, enabling end-to-end gradient flow through the DRR renderer.

        Usage:
            >>> renderer = SlangDiffDRRRenderer(volume, spacing)
            >>> rot = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> trans = torch.tensor([0., 0., 0.], requires_grad=True)
            >>>
            >>> image = SlangDiffDRRFunction.apply(renderer, rot, trans)
            >>> loss = (image - target).pow(2).mean()
            >>> loss.backward()  # Gradients computed via Slang autodiff!
            >>>
            >>> print(rot.grad)    # ∂L/∂rotation
            >>> print(trans.grad)  # ∂L/∂translation
        """

        @staticmethod
        def forward(
            ctx,
            renderer: SlangDiffDRRRenderer,
            rotation: torch.Tensor,
            translation: torch.Tensor,
        ) -> torch.Tensor:
            """Forward pass: render DRR image.

            Args:
                ctx: Autograd context.
                renderer: SlangDiffDRRRenderer instance.
                rotation: Tensor of shape (3,) with Euler angles in radians.
                translation: Tensor of shape (3,) with translation in mm.

            Returns:
                Rendered image as tensor of shape (H, W).
            """
            ctx.renderer = renderer
            ctx.save_for_backward(rotation, translation)

            # Convert to numpy and render
            rot_np = rotation.detach().cpu().numpy()
            trans_np = translation.detach().cpu().numpy()

            image_np = renderer.render(rot_np, trans_np)

            device = rotation.device
            return torch.from_numpy(image_np).to(device)

        @staticmethod
        def backward(
            ctx,
            grad_output: torch.Tensor,
        ) -> tuple[None, torch.Tensor, torch.Tensor]:
            """Backward pass: compute gradients via Slang autodiff.

            Args:
                ctx: Autograd context with saved tensors.
                grad_output: Gradient of loss w.r.t. output image, shape (H, W).

            Returns:
                Tuple of (None, grad_rotation, grad_translation).
                First None is for the renderer argument.
            """
            rotation, translation = ctx.saved_tensors
            renderer = ctx.renderer

            rot_np = rotation.detach().cpu().numpy()
            trans_np = translation.detach().cpu().numpy()
            grad_out_np = grad_output.detach().cpu().numpy()

            # Use Slang's autodiff for gradient computation
            _, grads = renderer.render_with_gradients(
                rot_np, trans_np, grad_output=grad_out_np
            )

            grad_rotation = torch.from_numpy(grads['rotation']).to(rotation.device)
            grad_translation = torch.from_numpy(grads['translation']).to(translation.device)

            return None, grad_rotation, grad_translation


    class TorchSlangDiffDRR(torch.nn.Module):
        """PyTorch Module for Slang-based differentiable DRR.

        This wraps SlangDiffDRRRenderer as a torch.nn.Module, enabling:
        - Integration with PyTorch neural network pipelines
        - Compatibility with torch.compile()
        - Easy parameter management

        Example:
            >>> # As part of a registration network
            >>> class RegistrationNet(torch.nn.Module):
            ...     def __init__(self, volume, spacing):
            ...         super().__init__()
            ...         self.drr = TorchSlangDiffDRR(volume, spacing)
            ...         self.pose_net = PoseEstimator()
            ...
            ...     def forward(self, x):
            ...         pose = self.pose_net(x)
            ...         rotation, translation = pose[:, :3], pose[:, 3:]
            ...         return self.drr(rotation[0], translation[0])

            >>> # For direct pose optimization
            >>> drr = TorchSlangDiffDRR(volume, spacing)
            >>> rot = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> trans = torch.tensor([0., 850., 0.], requires_grad=True)
            >>> optimizer = torch.optim.Adam([rot, trans], lr=0.01)
            >>>
            >>> for step in range(100):
            ...     optimizer.zero_grad()
            ...     synthetic = drr(rot, trans)
            ...     loss = -ncc(synthetic, target)  # Negative NCC
            ...     loss.backward()
            ...     optimizer.step()
        """

        def __init__(
            self,
            mu_volume: np.ndarray,
            spacing_zyx_mm: tuple[float, float, float],
            cfg: SlangDiffDRRConfig = SlangDiffDRRConfig(),
        ):
            """Initialize the PyTorch Slang DRR module.

            Args:
                mu_volume: 3D attenuation volume (Z, Y, X).
                spacing_zyx_mm: Voxel spacing in mm.
                cfg: Renderer configuration.
            """
            super().__init__()
            self._renderer = SlangDiffDRRRenderer(mu_volume, spacing_zyx_mm, cfg)
            self._cfg = cfg

        def forward(
            self,
            rotation: torch.Tensor,
            translation: torch.Tensor,
        ) -> torch.Tensor:
            """Forward pass through the differentiable renderer.

            Args:
                rotation: Tensor of shape (3,) or (N, 3) with Euler angles in radians.
                translation: Tensor of shape (3,) or (N, 3) with translation in mm.

            Returns:
                Rendered image(s) as tensor of shape (H, W) or (N, H, W).
            """
            if rotation.dim() == 1:
                return SlangDiffDRRFunction.apply(self._renderer, rotation, translation)
            else:
                # Batch rendering
                batch_size = rotation.shape[0]
                results = []
                for i in range(batch_size):
                    img = SlangDiffDRRFunction.apply(
                        self._renderer,
                        rotation[i],
                        translation[i],
                    )
                    results.append(img)
                return torch.stack(results)

        def render_numpy(
            self,
            rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
            translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        ) -> np.ndarray:
            """Render directly to numpy (no gradients).

            Args:
                rotation: Euler angles in radians.
                translation: Translation in mm.

            Returns:
                Rendered image as numpy array.
            """
            return self._renderer.render(rotation, translation)

        @property
        def config(self) -> SlangDiffDRRConfig:
            """Return renderer configuration."""
            return self._cfg

        @property
        def renderer(self) -> SlangDiffDRRRenderer:
            """Access the underlying SlangDiffDRRRenderer."""
            return self._renderer

        def __repr__(self) -> str:
            return f"TorchSlangDiffDRR({self._renderer})"


# =============================================================================
# Convenience Functions
# =============================================================================

def render_diffdrr_slang(
    mu_volume: np.ndarray,
    spacing_zyx_mm: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cfg: SlangDiffDRRConfig = SlangDiffDRRConfig(),
) -> np.ndarray:
    """One-shot differentiable DRR rendering using Slang.

    This is a convenience function for quick rendering without explicitly
    creating a renderer instance.

    Args:
        mu_volume: 3D attenuation volume (Z, Y, X) in mm^-1.
        spacing_zyx_mm: Voxel spacing in mm.
        rotation: Euler angles in radians.
        translation: Translation in mm.
        cfg: Renderer configuration.

    Returns:
        Rendered DRR image as numpy array of shape (H, W).
    """
    renderer = SlangDiffDRRRenderer(mu_volume, spacing_zyx_mm, cfg)
    return renderer.render(rotation, translation)


def create_slang_diffdrr_optimizer(
    mu_volume: np.ndarray,
    spacing_zyx_mm: tuple[float, float, float],
    initial_rotation: np.ndarray,
    initial_translation: np.ndarray,
    cfg: SlangDiffDRRConfig = SlangDiffDRRConfig(),
    lr: float = 1e-2,
) -> tuple["TorchSlangDiffDRR", torch.Tensor, torch.Tensor, torch.optim.Optimizer]:
    """Create a Slang DiffDRR setup for gradient-based pose optimization.

    This helper function sets up everything needed for 2D/3D registration:
    - Slang DiffDRR module
    - Learnable rotation and translation parameters
    - Adam optimizer

    Args:
        mu_volume: 3D attenuation volume (Z, Y, X).
        spacing_zyx_mm: Voxel spacing in mm.
        initial_rotation: Initial rotation in radians, shape (3,).
        initial_translation: Initial translation in mm, shape (3,).
        cfg: Renderer configuration.
        lr: Learning rate for Adam optimizer.

    Returns:
        Tuple of (drr_module, rotation, translation, optimizer)

    Example:
        >>> drr, rot, trans, opt = create_slang_diffdrr_optimizer(
        ...     volume, spacing,
        ...     initial_rotation=[0, 0, 0],
        ...     initial_translation=[0, 850, 0],
        ...     lr=0.01,
        ... )
        >>>
        >>> for step in range(100):
        ...     opt.zero_grad()
        ...     synthetic = drr(rot, trans)
        ...     loss = mse_loss(synthetic, target)
        ...     loss.backward()
        ...     opt.step()
        ...     print(f"Step {step}: loss={loss.item():.4f}")
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for optimizer creation")

    drr = TorchSlangDiffDRR(mu_volume, spacing_zyx_mm, cfg)

    rotation = torch.tensor(
        np.asarray(initial_rotation, dtype=np.float32),
        requires_grad=True,
    )
    translation = torch.tensor(
        np.asarray(initial_translation, dtype=np.float32),
        requires_grad=True,
    )

    optimizer = torch.optim.Adam([rotation, translation], lr=lr)

    return drr, rotation, translation, optimizer
