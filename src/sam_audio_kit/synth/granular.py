"""
Latent granular synthesis for creative audio manipulation.

Implements granular synthesis in the latent space, enabling:
- Building grain databases from separated stems
- Similarity-based grain replacement
- Texture remixing and sound design
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .codec import DACVAECodec, LatentRepresentation


@dataclass
class GrainDatabase:
    """Database of latent grains for granular synthesis."""

    grains: torch.Tensor
    """Grain tensor of shape (N, C, grain_size) - N grains."""

    grain_size: int
    """Size of each grain in frames."""

    sources: List[str]
    """Source identifiers for each grain."""

    sample_rate: int
    """Audio sample rate."""

    hop_length: int
    """Hop length for frame conversion."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    @property
    def n_grains(self) -> int:
        """Number of grains in database."""
        return self.grains.shape[0]

    @property
    def n_channels(self) -> int:
        """Number of latent channels."""
        return self.grains.shape[1]

    @property
    def grain_duration_seconds(self) -> float:
        """Duration of each grain in seconds."""
        return (self.grain_size * self.hop_length) / self.sample_rate

    def to(self, device: str) -> "GrainDatabase":
        """Move database to device."""
        return GrainDatabase(
            grains=self.grains.to(device),
            grain_size=self.grain_size,
            sources=self.sources,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            metadata=self.metadata,
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save database to file."""
        torch.save({
            "grains": self.grains.cpu(),
            "grain_size": self.grain_size,
            "sources": self.sources,
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "metadata": self.metadata,
        }, str(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GrainDatabase":
        """Load database from file."""
        data = torch.load(str(path), weights_only=False)
        return cls(
            grains=data["grains"],
            grain_size=data["grain_size"],
            sources=data["sources"],
            sample_rate=data["sample_rate"],
            hop_length=data["hop_length"],
            metadata=data.get("metadata", {}),
        )


class LatentGranular:
    """
    Latent space granular synthesis.

    Builds a database of latent "grains" from audio sources, then uses
    similarity matching to replace or remix audio in creative ways.

    Example:
        >>> from sam_audio_kit import SamAudio
        >>> from sam_audio_kit.synth import LatentGranular, DACVAECodec
        >>>
        >>> model = SamAudio.from_pretrained("base")
        >>> codec = DACVAECodec(model)
        >>> granular = LatentGranular(codec)
        >>>
        >>> # Build database from separated stems
        >>> drums = model.separate("song.wav", "drums")
        >>> granular.add_to_database(drums.target, source="drums")
        >>>
        >>> # Use vocals as guide, replace with similar drum grains
        >>> vocals = model.separate("song.wav", "vocals")
        >>> drum_vocals = granular.synthesize(vocals.target, temperature=0.5)
    """

    def __init__(
        self,
        codec: DACVAECodec,
        grain_size: int = 4,
        stride: int = 2,
        device: str = "cuda",
    ):
        """
        Initialize granular synthesizer.

        Args:
            codec: DACVAE codec for encoding/decoding
            grain_size: Size of grains in latent frames (default: 4 frames ≈ 42ms at 48kHz)
            stride: Stride for grain extraction (default: 2 for 50% overlap)
            device: Device for computations
        """
        self.codec = codec
        self.grain_size = grain_size
        self.stride = stride
        self.device = device

        self._database: Optional[GrainDatabase] = None
        self._grain_embeddings: Optional[torch.Tensor] = None

    @property
    def has_database(self) -> bool:
        """Check if database is built."""
        return self._database is not None and self._database.n_grains > 0

    @property
    def database(self) -> Optional[GrainDatabase]:
        """Get current database."""
        return self._database

    def _extract_grains(
        self,
        latent: Union[LatentRepresentation, torch.Tensor],
    ) -> Tuple[torch.Tensor, int, int]:
        """Extract grains from latent representation."""
        if isinstance(latent, LatentRepresentation):
            tensor = latent.latent
            sample_rate = latent.sample_rate
            hop_length = latent.hop_length
        else:
            tensor = latent
            sample_rate = self.codec.sample_rate
            hop_length = self.codec.hop_length

        # Handle batch dimension
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)

        B, C, T = tensor.shape
        grains = []

        for i in range(0, T - self.grain_size + 1, self.stride):
            grain = tensor[:, :, i:i + self.grain_size]
            grains.append(grain)

        if grains:
            # Stack: (n_grains, B, C, grain_size)
            grains_tensor = torch.stack(grains, dim=0)
            # Reshape to (n_grains * B, C, grain_size)
            grains_tensor = grains_tensor.view(-1, C, self.grain_size)
        else:
            grains_tensor = torch.empty(0, C, self.grain_size, device=tensor.device)

        return grains_tensor, sample_rate, hop_length

    def _compute_grain_embedding(self, grain: torch.Tensor) -> torch.Tensor:
        """Compute embedding for a grain (flattened and normalized)."""
        # Flatten: (N, C, grain_size) -> (N, C * grain_size)
        flat = grain.view(grain.shape[0], -1)
        # L2 normalize
        return F.normalize(flat, dim=-1)

    def build_database(
        self,
        sources: List[Any],
        source_names: Optional[List[str]] = None,
        augment: bool = False,
    ) -> GrainDatabase:
        """
        Build grain database from multiple sources.

        Args:
            sources: List of audio inputs (paths, tensors, LatentRepresentations)
            source_names: Optional names for each source
            augment: Apply augmentation (pitch/volume variations)

        Returns:
            Built GrainDatabase

        Example:
            >>> # Build from separated stems
            >>> drums = model.separate("song.wav", "drums")
            >>> bass = model.separate("song.wav", "bass")
            >>> db = granular.build_database([drums.target, bass.target])
        """
        all_grains = []
        all_sources = []
        sample_rate = self.codec.sample_rate
        hop_length = self.codec.hop_length

        for i, source in enumerate(sources):
            name = source_names[i] if source_names else f"source_{i}"

            # Encode if needed
            if isinstance(source, LatentRepresentation):
                latent = source
            elif isinstance(source, torch.Tensor):
                # Check if it's already latent (has right shape) or audio
                if source.dim() == 3 or (source.dim() == 2 and source.shape[0] > 2):
                    # Likely latent
                    latent = LatentRepresentation(
                        latent=source if source.dim() == 3 else source.unsqueeze(0),
                        sample_rate=sample_rate,
                        hop_length=hop_length,
                        source=name,
                    )
                else:
                    # Audio tensor
                    latent = self.codec.encode(source)
            else:
                # File path or numpy array
                latent = self.codec.encode(source)

            # Extract grains
            grains, sr, hl = self._extract_grains(latent)
            sample_rate = sr
            hop_length = hl

            all_grains.append(grains)
            all_sources.extend([name] * grains.shape[0])

            if augment:
                # Simple augmentation: scale grains
                for scale in [0.8, 1.2]:
                    scaled = grains * scale
                    all_grains.append(scaled)
                    all_sources.extend([f"{name}_aug_{scale}"] * grains.shape[0])

        if not all_grains:
            raise ValueError("No grains extracted from sources")

        grains_tensor = torch.cat(all_grains, dim=0).to(self.device)

        self._database = GrainDatabase(
            grains=grains_tensor,
            grain_size=self.grain_size,
            sources=all_sources,
            sample_rate=sample_rate,
            hop_length=hop_length,
            metadata={"n_sources": len(sources), "augmented": augment},
        )

        # Pre-compute embeddings for fast similarity search
        self._grain_embeddings = self._compute_grain_embedding(grains_tensor)

        return self._database

    def add_to_database(
        self,
        source: Any,
        source_name: str = "added",
    ) -> int:
        """
        Add more grains to existing database.

        Args:
            source: Audio input (path, tensor, LatentRepresentation)
            source_name: Name for this source

        Returns:
            Number of grains added
        """
        # Encode if needed
        if isinstance(source, LatentRepresentation):
            latent = source
        elif isinstance(source, torch.Tensor):
            if source.dim() <= 2 and source.shape[0] <= 2:
                latent = self.codec.encode(source)
            else:
                latent = LatentRepresentation(
                    latent=source if source.dim() == 3 else source.unsqueeze(0),
                    sample_rate=self.codec.sample_rate,
                    hop_length=self.codec.hop_length,
                    source=source_name,
                )
        else:
            latent = self.codec.encode(source)

        grains, sr, hl = self._extract_grains(latent)
        grains = grains.to(self.device)
        n_new = grains.shape[0]

        if self._database is None:
            # Create new database
            self._database = GrainDatabase(
                grains=grains,
                grain_size=self.grain_size,
                sources=[source_name] * n_new,
                sample_rate=sr,
                hop_length=hl,
            )
            self._grain_embeddings = self._compute_grain_embedding(grains)
        else:
            # Append to existing
            self._database = GrainDatabase(
                grains=torch.cat([self._database.grains, grains], dim=0),
                grain_size=self.grain_size,
                sources=self._database.sources + [source_name] * n_new,
                sample_rate=self._database.sample_rate,
                hop_length=self._database.hop_length,
                metadata=self._database.metadata,
            )
            new_embeddings = self._compute_grain_embedding(grains)
            self._grain_embeddings = torch.cat(
                [self._grain_embeddings, new_embeddings], dim=0
            )

        return n_new

    def find_similar_grains(
        self,
        query_grain: torch.Tensor,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find most similar grains in database.

        Args:
            query_grain: Query grain tensor (C, grain_size) or (1, C, grain_size)
            top_k: Number of similar grains to return
            threshold: Optional similarity threshold

        Returns:
            Tuple of (indices, similarities)
        """
        if not self.has_database:
            raise RuntimeError("No database built. Call build_database() first.")

        if query_grain.dim() == 2:
            query_grain = query_grain.unsqueeze(0)

        query_emb = self._compute_grain_embedding(query_grain)

        # Cosine similarity
        similarities = query_emb @ self._grain_embeddings.T  # (1, N)
        similarities = similarities.squeeze(0)

        if threshold is not None:
            mask = similarities >= threshold
            valid_indices = torch.where(mask)[0]
            if len(valid_indices) == 0:
                # Fall back to top-k if no grains pass threshold
                values, indices = similarities.topk(top_k)
                return indices, values
            similarities = similarities[mask]
            top_k = min(top_k, len(valid_indices))
            values, rel_indices = similarities.topk(top_k)
            indices = valid_indices[rel_indices]
            return indices, values
        else:
            values, indices = similarities.topk(top_k)
            return indices, values

    def synthesize(
        self,
        guide: Any,
        temperature: float = 0.0,
        top_k: int = 5,
        threshold: Optional[float] = None,
        blend_original: float = 0.0,
    ) -> LatentRepresentation:
        """
        Synthesize audio by replacing guide grains with similar database grains.

        Args:
            guide: Guide audio (provides structure/rhythm)
            temperature: Randomness in grain selection
                - 0.0 = always pick most similar
                - 1.0 = random weighted by similarity
            top_k: Number of candidates to consider
            threshold: Minimum similarity threshold
            blend_original: How much of original to keep (0=none, 1=all)

        Returns:
            Synthesized latent representation

        Example:
            >>> # Replace vocal rhythm with drum sounds
            >>> drums_from_vocals = granular.synthesize(
            ...     vocals,
            ...     temperature=0.3,  # Some variety
            ...     blend_original=0.1  # Slight original bleed
            ... )
        """
        if not self.has_database:
            raise RuntimeError("No database built. Call build_database() first.")

        # Encode guide
        if isinstance(guide, LatentRepresentation):
            guide_latent = guide
        elif isinstance(guide, torch.Tensor):
            # Check if it's already latent (3D with shape B, C, T where C is ~128)
            # vs audio (1D or 2D with shape (samples,) or (channels, samples))
            if guide.dim() == 3 and guide.shape[1] > 2:
                # Already latent: (B, C, T)
                guide_latent = LatentRepresentation(
                    latent=guide,
                    sample_rate=self._database.sample_rate,
                    hop_length=self._database.hop_length,
                    source="guide",
                )
            elif guide.dim() == 2 and guide.shape[0] > 2:
                # Already latent: (C, T) - add batch dim
                guide_latent = LatentRepresentation(
                    latent=guide.unsqueeze(0),
                    sample_rate=self._database.sample_rate,
                    hop_length=self._database.hop_length,
                    source="guide",
                )
            else:
                # Audio tensor - encode it
                guide_latent = self.codec.encode(guide)
        else:
            guide_latent = self.codec.encode(guide)

        guide_tensor = guide_latent.latent.to(self.device)
        if guide_tensor.dim() == 2:
            guide_tensor = guide_tensor.unsqueeze(0)

        B, C, T = guide_tensor.shape

        # Extract guide grains
        guide_grains, _, _ = self._extract_grains(guide_latent)
        n_grains = guide_grains.shape[0]

        # Replace each grain with similar from database
        output_grains = []

        for i in range(n_grains):
            query = guide_grains[i:i+1]  # (1, C, grain_size)
            indices, similarities = self.find_similar_grains(
                query, top_k=top_k, threshold=threshold
            )

            if temperature == 0.0:
                # Pick best match
                selected_idx = indices[0]
            else:
                # Temperature-scaled sampling
                probs = F.softmax(similarities / temperature, dim=0)
                selected_idx = indices[torch.multinomial(probs, 1).item()]

            selected_grain = self._database.grains[selected_idx]
            output_grains.append(selected_grain)

        # Reconstruct latent with overlap-add
        output_tensor = self._overlap_add(output_grains, T)

        # Blend with original if requested
        if blend_original > 0:
            output_tensor = (1 - blend_original) * output_tensor + blend_original * guide_tensor

        return LatentRepresentation(
            latent=output_tensor,
            sample_rate=self._database.sample_rate,
            hop_length=self._database.hop_length,
            source="granular_synthesis",
        )

    def _overlap_add(
        self,
        grains: List[torch.Tensor],
        target_length: int,
    ) -> torch.Tensor:
        """Reconstruct latent from grains using overlap-add."""
        if not grains:
            raise ValueError("No grains to reconstruct")

        C = grains[0].shape[0]
        device = grains[0].device

        output = torch.zeros(1, C, target_length, device=device)
        weights = torch.zeros(1, 1, target_length, device=device)

        # Hann window for smooth overlap
        window = torch.hann_window(self.grain_size, device=device)
        window = window.view(1, 1, -1)

        for i, grain in enumerate(grains):
            start = i * self.stride
            end = start + self.grain_size

            if end > target_length:
                # Handle edge case
                valid_len = target_length - start
                if valid_len > 0:
                    output[:, :, start:target_length] += grain[:, :valid_len] * window[:, :, :valid_len]
                    weights[:, :, start:target_length] += window[:, :, :valid_len]
            else:
                grain = grain.unsqueeze(0) if grain.dim() == 2 else grain
                output[:, :, start:end] += grain * window
                weights[:, :, start:end] += window

        # Normalize by weights
        weights = weights.clamp(min=1e-6)
        output = output / weights

        return output

    def remix(
        self,
        guide: Any,
        sources_weights: Optional[Dict[str, float]] = None,
        temperature: float = 0.3,
    ) -> LatentRepresentation:
        """
        Remix guide using grains from specific sources with weights.

        Args:
            guide: Guide audio
            sources_weights: Dict mapping source names to weights
                e.g., {"drums": 0.7, "bass": 0.3}
            temperature: Sampling temperature

        Returns:
            Remixed latent

        Example:
            >>> # Remix vocals with 70% drums, 30% bass
            >>> remix = granular.remix(
            ...     vocals,
            ...     sources_weights={"drums": 0.7, "bass": 0.3}
            ... )
        """
        if not self.has_database:
            raise RuntimeError("No database built.")

        if sources_weights is None:
            return self.synthesize(guide, temperature=temperature)

        # Create mask for valid sources
        source_mask = torch.zeros(self._database.n_grains, device=self.device)
        source_weights_tensor = torch.zeros(self._database.n_grains, device=self.device)

        for source_name, weight in sources_weights.items():
            for i, s in enumerate(self._database.sources):
                if source_name in s:
                    source_mask[i] = 1.0
                    source_weights_tensor[i] = weight

        # Normalize weights
        if source_mask.sum() > 0:
            source_weights_tensor = source_weights_tensor / source_weights_tensor.sum()

        # Encode guide
        if isinstance(guide, LatentRepresentation):
            guide_latent = guide
        else:
            guide_latent = self.codec.encode(guide)

        guide_tensor = guide_latent.latent.to(self.device)
        if guide_tensor.dim() == 2:
            guide_tensor = guide_tensor.unsqueeze(0)

        B, C, T = guide_tensor.shape
        guide_grains, _, _ = self._extract_grains(guide_latent)
        n_grains = guide_grains.shape[0]

        output_grains = []

        for i in range(n_grains):
            query = guide_grains[i:i+1]
            query_emb = self._compute_grain_embedding(query)

            # Compute similarities
            similarities = (query_emb @ self._grain_embeddings.T).squeeze(0)

            # Apply source mask and weights
            weighted_similarities = similarities * source_mask * source_weights_tensor

            # Temperature sampling
            if temperature > 0:
                probs = F.softmax(weighted_similarities / temperature, dim=0)
                selected_idx = torch.multinomial(probs, 1).item()
            else:
                selected_idx = weighted_similarities.argmax().item()

            output_grains.append(self._database.grains[selected_idx])

        output_tensor = self._overlap_add(output_grains, T)

        return LatentRepresentation(
            latent=output_tensor,
            sample_rate=self._database.sample_rate,
            hop_length=self._database.hop_length,
            source="granular_remix",
        )

    def random_collage(
        self,
        duration_seconds: float,
        sources_weights: Optional[Dict[str, float]] = None,
    ) -> LatentRepresentation:
        """
        Create random collage from database grains.

        Args:
            duration_seconds: Target duration in seconds
            sources_weights: Optional source weighting

        Returns:
            Random collage latent
        """
        if not self.has_database:
            raise RuntimeError("No database built.")

        n_frames = int(duration_seconds * self._database.sample_rate / self._database.hop_length)
        n_output_grains = (n_frames - self.grain_size) // self.stride + 1

        if sources_weights:
            # Weighted random selection
            weights = torch.zeros(self._database.n_grains, device=self.device)
            for source_name, weight in sources_weights.items():
                for i, s in enumerate(self._database.sources):
                    if source_name in s:
                        weights[i] = weight
            probs = weights / weights.sum()
        else:
            probs = torch.ones(self._database.n_grains, device=self.device)
            probs = probs / probs.sum()

        # Random selection
        indices = torch.multinomial(probs, n_output_grains, replacement=True)
        selected_grains = [self._database.grains[idx] for idx in indices]

        output_tensor = self._overlap_add(selected_grains, n_frames)

        return LatentRepresentation(
            latent=output_tensor,
            sample_rate=self._database.sample_rate,
            hop_length=self._database.hop_length,
            source="random_collage",
        )

    def clear_database(self) -> None:
        """Clear the current database."""
        self._database = None
        self._grain_embeddings = None

    def database_info(self) -> Dict[str, Any]:
        """Get information about current database."""
        if not self.has_database:
            return {"status": "no_database"}

        unique_sources = list(set(self._database.sources))
        source_counts = {s: self._database.sources.count(s) for s in unique_sources}

        return {
            "n_grains": self._database.n_grains,
            "grain_size_frames": self.grain_size,
            "grain_duration_ms": self._database.grain_duration_seconds * 1000,
            "sample_rate": self._database.sample_rate,
            "sources": unique_sources,
            "source_counts": source_counts,
            "total_duration_seconds": self._database.n_grains * self._database.grain_duration_seconds,
        }


__all__ = [
    "LatentGranular",
    "GrainDatabase",
]
