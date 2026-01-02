"""
Synthesis module for SAM-Audio latent space manipulation.

This module provides tools for creative audio manipulation in the latent space,
enabling workflows like:

- Separation → Synthesis: Separate stems, manipulate, re-synthesize
- Granular Synthesis: Build grain databases, remix with similarity matching
- Latent Effects: Interpolation, morphing, texture transfer
- Sound Design: Create new sounds from separated elements

Acknowledgments:
    The neural/latent sampling approach is inspired by the work of Naotokui:
    - https://huggingface.co/spaces/naotokui/latentgranular
    - https://arxiv.org/abs/2507.19202

Example:
    >>> from sam_audio_kit import SamAudioInfer
    >>> from sam_audio_kit.synth import LatentSynthesizer
    >>>
    >>> # Load model
    >>> model = SamAudioInfer.from_pretrained("base")
    >>>
    >>> # Create synthesizer
    >>> synth = LatentSynthesizer(model)
    >>>
    >>> # Separate and manipulate
    >>> vocals = model.separate("song.wav", "vocals")
    >>> drums = model.separate("song.wav", "drums")
    >>>
    >>> # Morph between them
    >>> hybrid = synth.interpolate(vocals.target, drums.target, alpha=0.3)
    >>> synth.save("hybrid.wav", hybrid)
    >>>
    >>> # Granular remix
    >>> synth.build_database([drums.target], names=["drums"])
    >>> drum_vocals = synth.granular_synthesize(vocals.target, temperature=0.3)
    >>> synth.save("drum_vocals.wav", drum_vocals)
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import torch

from .codec import DACVAECodec, LatentRepresentation
from .effects import LatentEffects
from .granular import GrainDatabase, LatentGranular


class LatentSynthesizer:
    """
    Unified interface for latent space synthesis and manipulation.

    Combines encoding/decoding, effects, and granular synthesis into
    a single easy-to-use class.

    Example:
        >>> synth = LatentSynthesizer(model)
        >>>
        >>> # Encode audio to latent
        >>> latent = synth.encode("audio.wav")
        >>>
        >>> # Apply effects
        >>> brightened = synth.enhance(latent, brightness=0.3)
        >>> morphed = synth.interpolate(latent_a, latent_b, alpha=0.5)
        >>>
        >>> # Decode back to audio
        >>> audio = synth.decode(morphed)
        >>> synth.save("output.wav", audio)
        >>>
        >>> # Granular synthesis
        >>> synth.build_database([drums, bass], names=["drums", "bass"])
        >>> remixed = synth.granular_synthesize(vocals, temperature=0.3)
    """

    def __init__(
        self,
        model: Any,
        grain_size: int = 4,
        grain_stride: int = 2,
        device: Optional[str] = None,
    ):
        """
        Initialize synthesizer.

        Args:
            model: SamAudioInfer instance
            grain_size: Grain size for granular synthesis (frames)
            grain_stride: Stride between grains
            device: Device for computations
        """
        self.codec = DACVAECodec(model, device=device)
        self.effects = LatentEffects(device=device or "cuda")
        self.granular = LatentGranular(
            self.codec,
            grain_size=grain_size,
            stride=grain_stride,
            device=device or "cuda",
        )

        self._device = device or "cuda"

    # =========================================================================
    # Encoding / Decoding
    # =========================================================================

    def encode(
        self,
        audio: Any,
        normalize: bool = False,
    ) -> LatentRepresentation:
        """
        Encode audio to latent representation.

        Args:
            audio: Audio file path, tensor, or numpy array

        Returns:
            LatentRepresentation
        """
        return self.codec.encode(audio, normalize=normalize)

    def decode(
        self,
        latent: Union[LatentRepresentation, torch.Tensor],
    ) -> torch.Tensor:
        """
        Decode latent to audio.

        Args:
            latent: Latent representation

        Returns:
            Audio tensor
        """
        return self.codec.decode(latent)

    def save(
        self,
        path: Union[str, Path],
        audio_or_latent: Union[torch.Tensor, LatentRepresentation],
        sample_rate: Optional[int] = None,
    ) -> None:
        """
        Save audio to file. Decodes latent if needed.

        Args:
            path: Output file path
            audio_or_latent: Audio tensor or latent to decode
            sample_rate: Optional sample rate override
        """
        if isinstance(audio_or_latent, LatentRepresentation):
            audio = self.decode(audio_or_latent)
            sample_rate = sample_rate or audio_or_latent.sample_rate
        else:
            audio = audio_or_latent
            sample_rate = sample_rate or self.codec.sample_rate

        self.codec.save(path, audio, sample_rate)

    # =========================================================================
    # Effects
    # =========================================================================

    def interpolate(
        self,
        a: Any,
        b: Any,
        alpha: float = 0.5,
        mode: Literal["linear", "spherical"] = "linear",
    ) -> LatentRepresentation:
        """
        Interpolate between two audio sources.

        Args:
            a: First audio (alpha=0)
            b: Second audio (alpha=1)
            alpha: Interpolation factor
            mode: Interpolation mode

        Returns:
            Interpolated latent

        Example:
            >>> # 30% drums, 70% vocals
            >>> hybrid = synth.interpolate(vocals, drums, alpha=0.3)
        """
        latent_a = self._ensure_encoded(a)
        latent_b = self._ensure_encoded(b)
        return self.effects.interpolate(latent_a, latent_b, alpha, mode)

    def crossfade(
        self,
        start: Any,
        end: Any,
        curve: Literal["linear", "exponential", "cosine", "sigmoid"] = "cosine",
    ) -> LatentRepresentation:
        """
        Time-varying crossfade from start to end.

        Args:
            start: Starting audio
            end: Ending audio
            curve: Crossfade curve shape

        Returns:
            Crossfaded latent (morphs over time)
        """
        latent_start = self._ensure_encoded(start)
        latent_end = self._ensure_encoded(end)
        return self.effects.crossfade(latent_start, latent_end, curve)

    def morph_sequence(
        self,
        sources: List[Any],
        steps_between: int = 10,
    ) -> LatentRepresentation:
        """
        Create morphing sequence through multiple sources.

        Args:
            sources: List of audio sources to morph through
            steps_between: Interpolation steps between each pair

        Returns:
            Morphing sequence latent
        """
        latents = [self._ensure_encoded(s) for s in sources]
        return self.effects.morph_sequence(latents, steps_between)

    def apply_texture(
        self,
        content: Any,
        style: Any,
        strength: float = 0.5,
    ) -> LatentRepresentation:
        """
        Apply texture from style source to content.

        Args:
            content: Content audio (structure to keep)
            style: Style audio (texture to transfer)
            strength: How much style to apply

        Returns:
            Stylized latent

        Example:
            >>> # Make piano "gritty" like guitar
            >>> gritty_piano = synth.apply_texture(piano, guitar, strength=0.6)
        """
        content_latent = self._ensure_encoded(content)
        style_latent = self._ensure_encoded(style)
        return self.effects.apply_texture(content_latent, style_latent, strength)

    def enhance(
        self,
        audio: Any,
        brightness: float = 0.0,
        warmth: float = 0.0,
        presence: float = 0.0,
    ) -> LatentRepresentation:
        """
        Apply tonal enhancements.

        Args:
            audio: Input audio
            brightness: High frequency boost (-1 to 1)
            warmth: Low frequency boost (-1 to 1)
            presence: Mid frequency boost (-1 to 1)

        Returns:
            Enhanced latent
        """
        latent = self._ensure_encoded(audio)
        return self.effects.enhance(latent, brightness, warmth, presence)

    def time_stretch(
        self,
        audio: Any,
        factor: float,
    ) -> LatentRepresentation:
        """
        Time-stretch audio in latent space.

        Args:
            audio: Input audio
            factor: Stretch factor (>1 = slower, <1 = faster)

        Returns:
            Time-stretched latent
        """
        latent = self._ensure_encoded(audio)
        return self.effects.time_stretch(latent, factor)

    def reverse(
        self,
        audio: Any,
    ) -> LatentRepresentation:
        """
        Reverse audio in latent space.

        Args:
            audio: Input audio

        Returns:
            Reversed latent
        """
        latent = self._ensure_encoded(audio)
        return self.effects.reverse(latent)

    def blend(
        self,
        sources: List[Any],
        weights: Optional[List[float]] = None,
    ) -> LatentRepresentation:
        """
        Blend multiple audio sources.

        Args:
            sources: List of audio sources
            weights: Blend weights (auto-normalized)

        Returns:
            Blended latent

        Example:
            >>> mix = synth.blend([vocals, drums, bass], weights=[0.5, 0.3, 0.2])
        """
        latents = [self._ensure_encoded(s) for s in sources]
        return self.effects.blend(latents, weights)

    # =========================================================================
    # Granular Synthesis
    # =========================================================================

    def build_database(
        self,
        sources: List[Any],
        names: Optional[List[str]] = None,
        augment: bool = False,
    ) -> Dict[str, Any]:
        """
        Build grain database from audio sources.

        Args:
            sources: List of audio sources (files, tensors, SeparationResults)
            names: Optional names for each source
            augment: Apply augmentation

        Returns:
            Database info dict

        Example:
            >>> synth.build_database(
            ...     [drums.target, bass.target],
            ...     names=["drums", "bass"]
            ... )
        """
        # Extract audio from SeparationResults if needed
        processed_sources = []
        for s in sources:
            if hasattr(s, "target"):  # SeparationResult
                processed_sources.append(s.target)
            else:
                processed_sources.append(s)

        self.granular.build_database(processed_sources, names, augment)
        return self.granular.database_info()

    def add_to_database(
        self,
        source: Any,
        name: str = "added",
    ) -> int:
        """
        Add more grains to existing database.

        Args:
            source: Audio source to add
            name: Name for this source

        Returns:
            Number of grains added
        """
        if hasattr(source, "target"):
            source = source.target
        return self.granular.add_to_database(source, name)

    def granular_synthesize(
        self,
        guide: Any,
        temperature: float = 0.0,
        blend_original: float = 0.0,
    ) -> LatentRepresentation:
        """
        Synthesize using granular replacement.

        Args:
            guide: Guide audio (provides rhythm/structure)
            temperature: Randomness (0=deterministic, 1=random)
            blend_original: How much original to keep

        Returns:
            Synthesized latent

        Example:
            >>> # Replace vocal rhythm with drum sounds
            >>> synth.build_database([drums.target], names=["drums"])
            >>> drum_vocals = synth.granular_synthesize(vocals.target, temperature=0.3)
        """
        if hasattr(guide, "target"):
            guide = guide.target
        return self.granular.synthesize(guide, temperature, blend_original=blend_original)

    def granular_remix(
        self,
        guide: Any,
        source_weights: Dict[str, float],
        temperature: float = 0.3,
    ) -> LatentRepresentation:
        """
        Remix guide with weighted sources.

        Args:
            guide: Guide audio
            source_weights: Dict of source names to weights
            temperature: Sampling temperature

        Returns:
            Remixed latent

        Example:
            >>> synth.build_database([drums, bass, keys], names=["drums", "bass", "keys"])
            >>> remix = synth.granular_remix(vocals, {"drums": 0.7, "bass": 0.3})
        """
        if hasattr(guide, "target"):
            guide = guide.target
        return self.granular.remix(guide, source_weights, temperature)

    def random_collage(
        self,
        duration_seconds: float,
        source_weights: Optional[Dict[str, float]] = None,
    ) -> LatentRepresentation:
        """
        Create random collage from database.

        Args:
            duration_seconds: Target duration
            source_weights: Optional source weighting

        Returns:
            Random collage latent
        """
        return self.granular.random_collage(duration_seconds, source_weights)

    def database_info(self) -> Dict[str, Any]:
        """Get information about grain database."""
        return self.granular.database_info()

    def clear_database(self) -> None:
        """Clear grain database."""
        self.granular.clear_database()

    # =========================================================================
    # Utilities
    # =========================================================================

    def _ensure_encoded(
        self,
        audio: Any,
    ) -> LatentRepresentation:
        """Ensure input is encoded to latent."""
        if isinstance(audio, LatentRepresentation):
            return audio
        elif isinstance(audio, torch.Tensor):
            # Check if already latent-like (3D with small first dim)
            if audio.dim() == 3 or (audio.dim() == 2 and audio.shape[0] > 2):
                return LatentRepresentation(
                    latent=audio if audio.dim() == 3 else audio.unsqueeze(0),
                    sample_rate=self.codec.sample_rate,
                    hop_length=self.codec.hop_length,
                    source="tensor",
                )
            else:
                return self.codec.encode(audio)
        elif hasattr(audio, "target"):
            # SeparationResult
            return self.codec.encode(audio.target)
        else:
            return self.codec.encode(audio)

    @property
    def sample_rate(self) -> int:
        """Codec sample rate."""
        return self.codec.sample_rate

    @property
    def has_database(self) -> bool:
        """Check if grain database exists."""
        return self.granular.has_database


# Public API
__all__ = [
    # Main class
    "LatentSynthesizer",
    # Components
    "DACVAECodec",
    "LatentRepresentation",
    "LatentEffects",
    "LatentGranular",
    "GrainDatabase",
]
