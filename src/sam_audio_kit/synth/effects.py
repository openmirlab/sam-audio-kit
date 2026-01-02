"""
Latent space effects for creative audio manipulation.

Provides tools for manipulating audio in the latent space:
- Interpolation and morphing
- Texture transfer
- Enhancement and transformation
- Blending and mixing
"""

from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .codec import LatentRepresentation


@dataclass
class EffectResult:
    """Result of applying an effect."""

    latent: LatentRepresentation
    """Output latent representation."""

    effect_name: str
    """Name of the effect applied."""

    params: dict
    """Parameters used for the effect."""


class LatentEffects:
    """
    Collection of latent space effects.

    Example:
        >>> effects = LatentEffects()
        >>>
        >>> # Interpolate between two sounds
        >>> morphed = effects.interpolate(vocals_latent, drums_latent, alpha=0.3)
        >>>
        >>> # Apply texture from one source to another
        >>> textured = effects.texture_transfer(piano_latent, guitar_latent, strength=0.5)
        >>>
        >>> # Time-varying morph
        >>> evolved = effects.crossfade(start_latent, end_latent, curve="exponential")
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize effects processor.

        Args:
            device: Device for computations
        """
        self.device = device

    def _ensure_latent(
        self,
        x: Union[LatentRepresentation, torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[LatentRepresentation]]:
        """Extract tensor and original representation."""
        if isinstance(x, LatentRepresentation):
            return x.latent, x
        return x, None

    def _wrap_result(
        self,
        tensor: torch.Tensor,
        reference: Optional[LatentRepresentation],
        source: str = "effect_output",
    ) -> LatentRepresentation:
        """Wrap tensor in LatentRepresentation."""
        if reference is not None:
            return LatentRepresentation(
                latent=tensor,
                sample_rate=reference.sample_rate,
                hop_length=reference.hop_length,
                source=source,
            )
        else:
            # Default values if no reference
            return LatentRepresentation(
                latent=tensor,
                sample_rate=48000,
                hop_length=512,
                source=source,
            )

    # =========================================================================
    # Interpolation and Morphing
    # =========================================================================

    def interpolate(
        self,
        latent_a: Union[LatentRepresentation, torch.Tensor],
        latent_b: Union[LatentRepresentation, torch.Tensor],
        alpha: float = 0.5,
        mode: Literal["linear", "spherical"] = "linear",
    ) -> LatentRepresentation:
        """
        Interpolate between two latent representations.

        Args:
            latent_a: First latent (alpha=0)
            latent_b: Second latent (alpha=1)
            alpha: Interpolation factor (0=a, 1=b)
            mode: Interpolation mode
                - "linear": Simple weighted average
                - "spherical": SLERP for smoother transitions

        Returns:
            Interpolated latent

        Example:
            >>> # 30% drums, 70% vocals
            >>> hybrid = effects.interpolate(vocals, drums, alpha=0.3)
        """
        a, ref_a = self._ensure_latent(latent_a)
        b, ref_b = self._ensure_latent(latent_b)

        # Ensure same length (truncate to shorter)
        min_len = min(a.shape[-1], b.shape[-1])
        a = a[:, :, :min_len]
        b = b[:, :, :min_len]

        if mode == "linear":
            result = (1 - alpha) * a + alpha * b
        elif mode == "spherical":
            # Spherical linear interpolation (SLERP)
            a_norm = F.normalize(a, dim=1)
            b_norm = F.normalize(b, dim=1)

            # Compute angle
            dot = (a_norm * b_norm).sum(dim=1, keepdim=True).clamp(-1, 1)
            omega = torch.acos(dot)

            # SLERP formula
            sin_omega = torch.sin(omega)
            # Avoid division by zero
            sin_omega = torch.where(
                sin_omega.abs() < 1e-6,
                torch.ones_like(sin_omega),
                sin_omega,
            )

            s_a = torch.sin((1 - alpha) * omega) / sin_omega
            s_b = torch.sin(alpha * omega) / sin_omega

            result = s_a * a + s_b * b
        else:
            raise ValueError(f"Unknown interpolation mode: {mode}")

        return self._wrap_result(result, ref_a or ref_b, f"interpolate_{alpha:.2f}")

    def crossfade(
        self,
        latent_start: Union[LatentRepresentation, torch.Tensor],
        latent_end: Union[LatentRepresentation, torch.Tensor],
        curve: Literal["linear", "exponential", "cosine", "sigmoid"] = "linear",
        overlap_frames: Optional[int] = None,
    ) -> LatentRepresentation:
        """
        Time-varying crossfade between two latents.

        Args:
            latent_start: Starting latent
            latent_end: Ending latent
            curve: Crossfade curve shape
            overlap_frames: Number of frames to crossfade (default: full length)

        Returns:
            Crossfaded latent (morphs from start to end over time)

        Example:
            >>> # Smooth transition from vocals to synth over entire duration
            >>> evolved = effects.crossfade(vocals, synth, curve="cosine")
        """
        start, ref_start = self._ensure_latent(latent_start)
        end, ref_end = self._ensure_latent(latent_end)

        # Match lengths
        min_len = min(start.shape[-1], end.shape[-1])
        start = start[:, :, :min_len]
        end = end[:, :, :min_len]

        n_frames = min_len
        if overlap_frames is not None:
            n_frames = min(overlap_frames, min_len)

        # Create crossfade curve
        t = torch.linspace(0, 1, n_frames, device=start.device)

        if curve == "linear":
            alpha = t
        elif curve == "exponential":
            alpha = t ** 2
        elif curve == "cosine":
            alpha = 0.5 * (1 - torch.cos(t * np.pi))
        elif curve == "sigmoid":
            # Sigmoid centered at 0.5
            alpha = torch.sigmoid(10 * (t - 0.5))
        else:
            raise ValueError(f"Unknown curve: {curve}")

        # Reshape for broadcasting: (1, 1, T)
        alpha = alpha.view(1, 1, -1)

        # Apply crossfade
        if overlap_frames is not None and overlap_frames < min_len:
            # Partial crossfade at the end
            result = start.clone()
            result[:, :, -n_frames:] = (
                (1 - alpha) * start[:, :, -n_frames:] + alpha * end[:, :, -n_frames:]
            )
        else:
            # Full crossfade
            result = (1 - alpha) * start + alpha * end

        return self._wrap_result(result, ref_start or ref_end, f"crossfade_{curve}")

    def morph_sequence(
        self,
        latents: List[Union[LatentRepresentation, torch.Tensor]],
        steps_between: int = 10,
        mode: Literal["linear", "spherical"] = "linear",
    ) -> LatentRepresentation:
        """
        Create a morphing sequence through multiple latents.

        Args:
            latents: List of latent representations to morph through
            steps_between: Number of interpolation steps between each pair
            mode: Interpolation mode

        Returns:
            Concatenated morphing sequence

        Example:
            >>> # Morph through: vocals → drums → bass → vocals
            >>> sequence = effects.morph_sequence([vocals, drums, bass, vocals])
        """
        if len(latents) < 2:
            raise ValueError("Need at least 2 latents for morphing sequence")

        segments = []
        ref = None

        for i in range(len(latents) - 1):
            a, ref_a = self._ensure_latent(latents[i])
            b, _ = self._ensure_latent(latents[i + 1])
            ref = ref or ref_a

            for step in range(steps_between):
                alpha = step / steps_between
                interp = self.interpolate(a, b, alpha, mode)
                segments.append(interp.latent)

        # Add final latent
        final, _ = self._ensure_latent(latents[-1])
        segments.append(final)

        result = torch.cat(segments, dim=-1)
        return self._wrap_result(result, ref, "morph_sequence")

    # =========================================================================
    # Texture and Style Transfer
    # =========================================================================

    def extract_texture(
        self,
        latent: Union[LatentRepresentation, torch.Tensor],
        window_size: int = 16,
    ) -> torch.Tensor:
        """
        Extract texture statistics from latent.

        Computes mean and std across time windows for texture representation.

        Args:
            latent: Source latent
            window_size: Window size in frames

        Returns:
            Texture descriptor tensor
        """
        tensor, _ = self._ensure_latent(latent)

        # Compute statistics per window
        B, C, T = tensor.shape
        n_windows = T // window_size

        if n_windows == 0:
            # If latent is shorter than window, use global stats
            mean = tensor.mean(dim=-1, keepdim=True)
            std = tensor.std(dim=-1, keepdim=True)
            return torch.cat([mean, std], dim=1)

        # Reshape into windows
        trimmed = tensor[:, :, :n_windows * window_size]
        windows = trimmed.view(B, C, n_windows, window_size)

        # Stats per window
        means = windows.mean(dim=-1)  # (B, C, n_windows)
        stds = windows.std(dim=-1)

        # Concatenate as texture descriptor
        return torch.cat([means, stds], dim=1)  # (B, 2*C, n_windows)

    def apply_texture(
        self,
        content_latent: Union[LatentRepresentation, torch.Tensor],
        style_latent: Union[LatentRepresentation, torch.Tensor],
        strength: float = 0.5,
        preserve_dynamics: bool = True,
    ) -> LatentRepresentation:
        """
        Apply texture/style from one latent to another.

        Args:
            content_latent: Content source (structure to preserve)
            style_latent: Style source (texture to transfer)
            strength: How much style to apply (0=none, 1=full)
            preserve_dynamics: Keep content's temporal dynamics

        Returns:
            Stylized latent

        Example:
            >>> # Make piano sound "gritty" like distorted guitar
            >>> gritty_piano = effects.apply_texture(piano, distorted_guitar, strength=0.6)
        """
        content, ref = self._ensure_latent(content_latent)
        style, _ = self._ensure_latent(style_latent)

        # Compute style statistics
        style_mean = style.mean(dim=-1, keepdim=True)
        style_std = style.std(dim=-1, keepdim=True) + 1e-6

        # Compute content statistics
        content_mean = content.mean(dim=-1, keepdim=True)
        content_std = content.std(dim=-1, keepdim=True) + 1e-6

        if preserve_dynamics:
            # Normalize content, apply style stats
            normalized = (content - content_mean) / content_std
            stylized = normalized * style_std + style_mean
        else:
            # Direct style transfer
            stylized = style_mean + (content - content_mean) * (style_std / content_std)

        # Blend with original
        result = (1 - strength) * content + strength * stylized

        return self._wrap_result(result, ref, f"texture_transfer_{strength:.2f}")

    def match_energy(
        self,
        source: Union[LatentRepresentation, torch.Tensor],
        target: Union[LatentRepresentation, torch.Tensor],
    ) -> LatentRepresentation:
        """
        Match the energy envelope of source to target.

        Args:
            source: Source latent (content to keep)
            target: Target latent (energy envelope to match)

        Returns:
            Source with matched energy
        """
        src, ref = self._ensure_latent(source)
        tgt, _ = self._ensure_latent(target)

        # Compute energy (RMS per frame)
        src_energy = (src ** 2).mean(dim=1, keepdim=True).sqrt() + 1e-6
        tgt_energy = (tgt ** 2).mean(dim=1, keepdim=True).sqrt() + 1e-6

        # Match lengths
        min_len = min(src.shape[-1], tgt.shape[-1])
        src = src[:, :, :min_len]
        src_energy = src_energy[:, :, :min_len]
        tgt_energy = tgt_energy[:, :, :min_len]

        # Scale source to match target energy
        result = src * (tgt_energy / src_energy)

        return self._wrap_result(result, ref, "energy_matched")

    # =========================================================================
    # Enhancement and Transformation
    # =========================================================================

    def enhance(
        self,
        latent: Union[LatentRepresentation, torch.Tensor],
        brightness: float = 0.0,
        warmth: float = 0.0,
        presence: float = 0.0,
    ) -> LatentRepresentation:
        """
        Enhance latent with tonal adjustments.

        These are heuristic adjustments in latent space - results may vary.

        Args:
            latent: Input latent
            brightness: Increase high-frequency content (-1 to 1)
            warmth: Increase low-frequency content (-1 to 1)
            presence: Increase mid-frequency content (-1 to 1)

        Returns:
            Enhanced latent

        Example:
            >>> # Make vocals brighter and more present
            >>> enhanced = effects.enhance(vocals, brightness=0.3, presence=0.2)
        """
        tensor, ref = self._ensure_latent(latent)
        result = tensor.clone()

        C = tensor.shape[1]

        if brightness != 0:
            # Boost upper channels (heuristic: upper channels ≈ higher frequencies)
            upper_mask = torch.linspace(0, 1, C, device=tensor.device).view(1, C, 1)
            result = result + brightness * 0.3 * (result * upper_mask)

        if warmth != 0:
            # Boost lower channels
            lower_mask = torch.linspace(1, 0, C, device=tensor.device).view(1, C, 1)
            result = result + warmth * 0.3 * (result * lower_mask)

        if presence != 0:
            # Boost middle channels
            mid_mask = 1 - 2 * torch.abs(
                torch.linspace(-0.5, 0.5, C, device=tensor.device)
            ).view(1, C, 1)
            result = result + presence * 0.3 * (result * mid_mask)

        return self._wrap_result(result, ref, "enhanced")

    def normalize(
        self,
        latent: Union[LatentRepresentation, torch.Tensor],
        mode: Literal["peak", "rms", "channel"] = "peak",
    ) -> LatentRepresentation:
        """
        Normalize latent representation.

        Args:
            latent: Input latent
            mode: Normalization mode
                - "peak": Scale so max absolute value is 1
                - "rms": Scale to unit RMS
                - "channel": L2 normalize each channel

        Returns:
            Normalized latent
        """
        tensor, ref = self._ensure_latent(latent)

        if mode == "peak":
            max_val = tensor.abs().max()
            if max_val > 0:
                result = tensor / max_val
            else:
                result = tensor
        elif mode == "rms":
            rms = (tensor ** 2).mean().sqrt()
            if rms > 0:
                result = tensor / rms
            else:
                result = tensor
        elif mode == "channel":
            result = F.normalize(tensor, dim=1)
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")

        return self._wrap_result(result, ref, f"normalized_{mode}")

    def time_stretch(
        self,
        latent: Union[LatentRepresentation, torch.Tensor],
        factor: float = 1.0,
    ) -> LatentRepresentation:
        """
        Time-stretch latent representation.

        Args:
            latent: Input latent
            factor: Stretch factor (>1 = longer, <1 = shorter)

        Returns:
            Time-stretched latent

        Example:
            >>> # Make 2x slower
            >>> stretched = effects.time_stretch(latent, factor=2.0)
        """
        tensor, ref = self._ensure_latent(latent)

        if factor == 1.0:
            return self._wrap_result(tensor, ref, "time_stretch_1.0")

        B, C, T = tensor.shape
        new_T = int(T * factor)

        # Use interpolation for stretching
        result = F.interpolate(
            tensor,
            size=new_T,
            mode="linear",
            align_corners=True,
        )

        return self._wrap_result(result, ref, f"time_stretch_{factor:.2f}")

    def reverse(
        self,
        latent: Union[LatentRepresentation, torch.Tensor],
    ) -> LatentRepresentation:
        """
        Reverse latent in time.

        Args:
            latent: Input latent

        Returns:
            Time-reversed latent
        """
        tensor, ref = self._ensure_latent(latent)
        result = torch.flip(tensor, dims=[-1])
        return self._wrap_result(result, ref, "reversed")

    # =========================================================================
    # Blending and Mixing
    # =========================================================================

    def blend(
        self,
        latents: List[Union[LatentRepresentation, torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> LatentRepresentation:
        """
        Blend multiple latents with weights.

        Args:
            latents: List of latent representations
            weights: Weights for each latent (normalized automatically)

        Returns:
            Blended latent

        Example:
            >>> # Mix: 50% vocals, 30% drums, 20% bass
            >>> mix = effects.blend([vocals, drums, bass], weights=[0.5, 0.3, 0.2])
        """
        if len(latents) == 0:
            raise ValueError("Need at least one latent")

        if weights is None:
            weights = [1.0] * len(latents)

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        # Extract tensors
        tensors = []
        ref = None
        for lat in latents:
            t, r = self._ensure_latent(lat)
            tensors.append(t)
            ref = ref or r

        # Match lengths
        min_len = min(t.shape[-1] for t in tensors)
        tensors = [t[:, :, :min_len] for t in tensors]

        # Weighted sum
        result = sum(w * t for w, t in zip(weights, tensors))

        return self._wrap_result(result, ref, "blended")

    def layer(
        self,
        base: Union[LatentRepresentation, torch.Tensor],
        overlay: Union[LatentRepresentation, torch.Tensor],
        mode: Literal["add", "multiply", "screen", "overlay"] = "add",
        amount: float = 1.0,
    ) -> LatentRepresentation:
        """
        Layer two latents using blend modes (inspired by image editing).

        Args:
            base: Base latent
            overlay: Overlay latent
            mode: Blend mode
                - "add": Simple addition
                - "multiply": Multiplicative blend
                - "screen": Inverse multiply (brightens)
                - "overlay": Combines multiply and screen
            amount: How much overlay to apply

        Returns:
            Layered result
        """
        base_t, ref = self._ensure_latent(base)
        overlay_t, _ = self._ensure_latent(overlay)

        # Match lengths
        min_len = min(base_t.shape[-1], overlay_t.shape[-1])
        base_t = base_t[:, :, :min_len]
        overlay_t = overlay_t[:, :, :min_len]

        # Normalize to 0-1 range for blend modes
        base_norm = (base_t - base_t.min()) / (base_t.max() - base_t.min() + 1e-6)
        overlay_norm = (overlay_t - overlay_t.min()) / (overlay_t.max() - overlay_t.min() + 1e-6)

        if mode == "add":
            blended = base_t + amount * overlay_t
        elif mode == "multiply":
            blended_norm = base_norm * overlay_norm
            blended = base_t * (1 - amount) + amount * blended_norm * (base_t.max() - base_t.min())
        elif mode == "screen":
            blended_norm = 1 - (1 - base_norm) * (1 - overlay_norm)
            blended = base_t * (1 - amount) + amount * blended_norm * (base_t.max() - base_t.min())
        elif mode == "overlay":
            # Overlay = multiply dark, screen light
            mask = (base_norm < 0.5).float()
            multiply = 2 * base_norm * overlay_norm
            screen = 1 - 2 * (1 - base_norm) * (1 - overlay_norm)
            blended_norm = mask * multiply + (1 - mask) * screen
            blended = base_t * (1 - amount) + amount * blended_norm * (base_t.max() - base_t.min())
        else:
            raise ValueError(f"Unknown blend mode: {mode}")

        return self._wrap_result(blended, ref, f"layer_{mode}")


__all__ = [
    "LatentEffects",
    "EffectResult",
]
