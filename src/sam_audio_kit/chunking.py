"""
Audio chunking utilities for processing long audio files.

This module provides functionality to split long audio files into manageable
chunks for processing, avoiding out-of-memory errors on GPU.
"""

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
import torch

from .types import DEFAULT_CHUNK_DURATION, DEFAULT_SAMPLE_RATE


@dataclass
class ChunkingConfig:
    """Configuration for audio chunking."""

    chunk_duration: float = DEFAULT_CHUNK_DURATION  # seconds
    sample_rate: int = DEFAULT_SAMPLE_RATE
    overlap_duration: float = 0.0  # seconds of overlap between chunks
    crossfade_duration: float = 0.1  # seconds for crossfade blending

    def __post_init__(self):
        if self.chunk_duration <= 0:
            raise ValueError("chunk_duration must be positive")
        if self.overlap_duration < 0:
            raise ValueError("overlap_duration cannot be negative")
        if self.overlap_duration >= self.chunk_duration:
            raise ValueError("overlap_duration must be less than chunk_duration")
        if self.crossfade_duration < 0:
            raise ValueError("crossfade_duration cannot be negative")

    @property
    def chunk_samples(self) -> int:
        """Number of samples per chunk."""
        return int(self.chunk_duration * self.sample_rate)

    @property
    def overlap_samples(self) -> int:
        """Number of overlap samples between chunks."""
        return int(self.overlap_duration * self.sample_rate)

    @property
    def hop_samples(self) -> int:
        """Number of samples to advance between chunks."""
        return self.chunk_samples - self.overlap_samples

    @property
    def crossfade_samples(self) -> int:
        """Number of samples for crossfade blending."""
        return int(self.crossfade_duration * self.sample_rate)


@dataclass
class AudioChunk:
    """Represents a chunk of audio with metadata."""

    data: torch.Tensor  # Audio data (samples, channels) or (samples,)
    start_sample: int  # Start position in original audio
    end_sample: int  # End position in original audio
    chunk_index: int  # Index of this chunk
    total_chunks: int  # Total number of chunks
    sample_rate: int

    @property
    def duration(self) -> float:
        """Duration of this chunk in seconds."""
        return len(self.data) / self.sample_rate

    @property
    def is_first(self) -> bool:
        """Whether this is the first chunk."""
        return self.chunk_index == 0

    @property
    def is_last(self) -> bool:
        """Whether this is the last chunk."""
        return self.chunk_index == self.total_chunks - 1


class AudioChunker:
    """
    Utility class for chunking audio into smaller segments.

    Example:
        >>> chunker = AudioChunker(chunk_duration=25.0)
        >>> for chunk in chunker.chunk(audio_tensor, sample_rate=16000):
        ...     result = model.process(chunk.data)
        ...     results.append(result)
        >>> final = chunker.merge(results)
    """

    def __init__(
        self,
        chunk_duration: float = DEFAULT_CHUNK_DURATION,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        overlap_duration: float = 0.0,
        crossfade_duration: float = 0.1,
    ):
        """
        Initialize the audio chunker.

        Args:
            chunk_duration: Duration of each chunk in seconds
            sample_rate: Sample rate of the audio
            overlap_duration: Overlap between chunks in seconds
            crossfade_duration: Crossfade duration for blending in seconds
        """
        self.config = ChunkingConfig(
            chunk_duration=chunk_duration,
            sample_rate=sample_rate,
            overlap_duration=overlap_duration,
            crossfade_duration=crossfade_duration,
        )

    def needs_chunking(self, audio: torch.Tensor, sample_rate: Optional[int] = None) -> bool:
        """
        Check if audio needs to be chunked.

        Args:
            audio: Audio tensor (samples,) or (samples, channels)
            sample_rate: Sample rate (uses config default if not specified)

        Returns:
            True if audio is longer than chunk_duration
        """
        sr = sample_rate or self.config.sample_rate
        max_samples = int(self.config.chunk_duration * sr)
        num_samples = audio.shape[0] if audio.dim() == 1 else audio.shape[-1]
        return num_samples > max_samples

    def get_num_chunks(self, audio: torch.Tensor, sample_rate: Optional[int] = None) -> int:
        """
        Calculate the number of chunks for given audio.

        Args:
            audio: Audio tensor
            sample_rate: Sample rate

        Returns:
            Number of chunks needed
        """
        sr = sample_rate or self.config.sample_rate
        num_samples = audio.shape[0] if audio.dim() == 1 else audio.shape[-1]

        if num_samples <= self.config.chunk_samples:
            return 1

        # Calculate with overlap
        hop = int(self.config.chunk_duration * sr) - int(self.config.overlap_duration * sr)
        return max(1, int(np.ceil((num_samples - self.config.chunk_samples) / hop)) + 1)

    def chunk(
        self,
        audio: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> Iterator[AudioChunk]:
        """
        Split audio into chunks.

        Args:
            audio: Audio tensor (samples,) or (samples, channels) or (channels, samples)
            sample_rate: Sample rate of the audio

        Yields:
            AudioChunk objects containing chunk data and metadata
        """
        sr = sample_rate or self.config.sample_rate

        # Normalize to (samples, channels) format
        if audio.dim() == 1:
            audio = audio.unsqueeze(-1)  # (samples, 1)
        elif audio.dim() == 2 and audio.shape[0] <= 2:
            # Assume (channels, samples) format, transpose
            audio = audio.T  # (samples, channels)

        num_samples = audio.shape[0]
        chunk_samples = int(self.config.chunk_duration * sr)
        hop_samples = chunk_samples - int(self.config.overlap_duration * sr)

        # Calculate total chunks
        if num_samples <= chunk_samples:
            total_chunks = 1
        else:
            total_chunks = int(np.ceil((num_samples - chunk_samples) / hop_samples)) + 1

        # Generate chunks
        for i in range(total_chunks):
            start = i * hop_samples
            end = min(start + chunk_samples, num_samples)

            chunk_data = audio[start:end]

            yield AudioChunk(
                data=chunk_data,
                start_sample=start,
                end_sample=end,
                chunk_index=i,
                total_chunks=total_chunks,
                sample_rate=sr,
            )

    def merge(
        self,
        chunks: list[torch.Tensor],
        crossfade: bool = True,
    ) -> torch.Tensor:
        """
        Merge processed chunks back into a single audio tensor.

        Args:
            chunks: List of processed audio chunks
            crossfade: Whether to apply crossfade blending at boundaries

        Returns:
            Merged audio tensor
        """
        if not chunks:
            raise ValueError("No chunks to merge")

        if len(chunks) == 1:
            return chunks[0]

        if not crossfade or self.config.overlap_samples == 0:
            # Simple concatenation
            return torch.cat(chunks, dim=0)

        # Crossfade merging
        crossfade_samples = min(
            self.config.crossfade_samples,
            self.config.overlap_samples,
        )

        result_parts = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = result_parts[-1]
            curr_chunk = chunks[i]

            if crossfade_samples > 0 and len(prev_chunk) >= crossfade_samples:
                # Create crossfade ramps
                fade_out = torch.linspace(1.0, 0.0, crossfade_samples, device=prev_chunk.device)
                fade_in = torch.linspace(0.0, 1.0, crossfade_samples, device=curr_chunk.device)

                # Expand dims for broadcasting if needed
                if prev_chunk.dim() > 1:
                    fade_out = fade_out.unsqueeze(-1)
                    fade_in = fade_in.unsqueeze(-1)

                # Apply crossfade
                prev_end = prev_chunk[-crossfade_samples:]
                curr_start = curr_chunk[:crossfade_samples]
                blended = prev_end * fade_out + curr_start * fade_in

                # Reconstruct
                result_parts[-1] = prev_chunk[:-crossfade_samples]
                result_parts.append(blended)
                result_parts.append(curr_chunk[crossfade_samples:])
            else:
                result_parts.append(curr_chunk)

        return torch.cat(result_parts, dim=0)

    def merge_results(
        self,
        results: list[Tuple[torch.Tensor, torch.Tensor]],
        crossfade: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merge separation results (target, residual) from multiple chunks.

        Args:
            results: List of (target, residual) tuples
            crossfade: Whether to apply crossfade blending

        Returns:
            Tuple of (merged_target, merged_residual)
        """
        targets = [r[0] for r in results]
        residuals = [r[1] for r in results]

        merged_target = self.merge(targets, crossfade=crossfade)
        merged_residual = self.merge(residuals, crossfade=crossfade)

        return merged_target, merged_residual
