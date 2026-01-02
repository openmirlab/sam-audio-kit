"""
DACVAE codec wrapper for direct latent manipulation and synthesis.

This module exposes SAM-Audio's DACVAE for creative latent space operations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio

from ..types import AudioInput, DeviceType, DType, get_torch_dtype


@dataclass
class LatentRepresentation:
    """Container for latent representations with metadata."""

    latent: torch.Tensor
    """Latent tensor of shape (B, C, T) - batch, channels, time frames."""

    sample_rate: int
    """Original audio sample rate."""

    hop_length: int
    """Hop length used by codec (samples per latent frame)."""

    source: Optional[str] = None
    """Source identifier (file path, description, etc.)."""

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get latent shape."""
        return tuple(self.latent.shape)

    @property
    def n_frames(self) -> int:
        """Number of time frames."""
        return self.latent.shape[-1]

    @property
    def n_channels(self) -> int:
        """Number of latent channels."""
        return self.latent.shape[1]

    @property
    def duration_seconds(self) -> float:
        """Estimated duration in seconds."""
        return (self.n_frames * self.hop_length) / self.sample_rate

    def to(self, device: DeviceType) -> "LatentRepresentation":
        """Move to device."""
        return LatentRepresentation(
            latent=self.latent.to(device),
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            source=self.source,
        )

    def numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self.latent.cpu().numpy()

    def clone(self) -> "LatentRepresentation":
        """Create a copy."""
        return LatentRepresentation(
            latent=self.latent.clone(),
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            source=self.source,
        )

    def __getitem__(self, idx) -> "LatentRepresentation":
        """Slice latent frames."""
        if isinstance(idx, slice):
            return LatentRepresentation(
                latent=self.latent[:, :, idx],
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
                source=self.source,
            )
        elif isinstance(idx, int):
            return LatentRepresentation(
                latent=self.latent[:, :, idx:idx+1],
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
                source=self.source,
            )
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")


class DACVAECodec:
    """
    Wrapper for SAM-Audio's DACVAE codec for direct latent manipulation.

    Provides encode/decode access to the neural audio codec, enabling:
    - Audio → Latent encoding
    - Latent → Audio synthesis
    - Latent space manipulation

    Example:
        >>> from sam_audio_kit import SamAudio
        >>> from sam_audio_kit.synth import DACVAECodec
        >>>
        >>> model = SamAudio.from_pretrained("base")
        >>> codec = DACVAECodec(model)
        >>>
        >>> # Encode audio to latent
        >>> latent = codec.encode("audio.wav")
        >>> print(f"Latent shape: {latent.shape}")  # (1, 128, T)
        >>>
        >>> # Decode back to audio
        >>> audio = codec.decode(latent)
        >>> codec.save("reconstructed.wav", audio)
    """

    def __init__(
        self,
        model: Any,
        device: Optional[DeviceType] = None,
        dtype: Optional[DType] = None,
    ):
        """
        Initialize codec wrapper.

        Args:
            model: SamAudio instance or raw SAMAudio model
            device: Override device
            dtype: Override dtype
        """
        # Handle both wrapper and raw model
        if hasattr(model, "_model"):
            self._sam_model = model._model
            self._device = device or model._device
            self._dtype = dtype or model._dtype
        else:
            self._sam_model = model
            self._device = device or "cuda"
            self._dtype = dtype or "bfloat16"

        self._torch_dtype = get_torch_dtype(self._dtype)
        self._codec = self._sam_model.audio_codec

    @property
    def sample_rate(self) -> int:
        """Codec sample rate."""
        return self._codec.sample_rate

    @property
    def hop_length(self) -> int:
        """Hop length (samples per latent frame)."""
        return self._codec.hop_length

    @property
    def latent_dim(self) -> int:
        """Latent dimension (number of channels)."""
        # Get from quantizer input projection
        return self._codec.quantizer.in_proj[0].in_features

    @property
    def frame_rate(self) -> float:
        """Latent frames per second."""
        return self.sample_rate / self.hop_length

    def _load_audio(
        self,
        audio_input: AudioInput,
        target_sr: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, Optional[str]]:
        """Load and preprocess audio."""
        target_sr = target_sr or self.sample_rate
        source = None

        if isinstance(audio_input, (str, Path)):
            audio, sr = torchaudio.load(str(audio_input))
            source = str(audio_input)
        elif isinstance(audio_input, np.ndarray):
            audio = torch.from_numpy(audio_input).float()
            sr = target_sr
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
        elif isinstance(audio_input, torch.Tensor):
            audio = audio_input.float()
            sr = target_sr
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
        else:
            raise TypeError(f"Unsupported audio type: {type(audio_input)}")

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(audio)
            sr = target_sr

        return audio, sr, source

    @torch.inference_mode()
    def encode(
        self,
        audio_input: AudioInput,
        normalize: bool = False,
        chunk_duration: float = 30.0,
    ) -> LatentRepresentation:
        """
        Encode audio to latent representation.

        Args:
            audio_input: Audio file path, numpy array, or tensor
            normalize: L2 normalize latent vectors
            chunk_duration: Max duration per chunk in seconds (for memory efficiency)

        Returns:
            LatentRepresentation containing the encoded latent

        Example:
            >>> latent = codec.encode("vocals.wav")
            >>> print(f"Shape: {latent.shape}")  # (1, 128, num_frames)
            >>> print(f"Duration: {latent.duration_seconds:.2f}s")
        """
        audio, sr, source = self._load_audio(audio_input)

        # DACVAE encoder expects (B, 1, T) shape - batch, 1 channel, time samples
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)  # (T,) -> (1, 1, T)
        elif audio.dim() == 2:
            # (C, T) -> (1, 1, T) taking first channel
            audio = audio[0:1, :].unsqueeze(0)  # (1, T) -> (1, 1, T)
        elif audio.dim() == 3 and audio.shape[1] == 1:
            # Already (B, 1, T) - keep as is
            pass
        else:
            raise ValueError(f"Unexpected audio shape: {audio.shape}")

        # Move to device with matching dtype
        audio = audio.to(self._device, self._torch_dtype)

        # Chunk long audio to avoid OOM
        total_samples = audio.shape[-1]
        chunk_samples = int(chunk_duration * sr)

        if total_samples <= chunk_samples:
            # Short audio - encode directly
            latent = self._codec(audio)
        else:
            # Long audio - encode in chunks and concatenate
            latent_chunks = []
            for start in range(0, total_samples, chunk_samples):
                end = min(start + chunk_samples, total_samples)
                chunk = audio[:, :, start:end]
                chunk_latent = self._codec(chunk)
                latent_chunks.append(chunk_latent)
            latent = torch.cat(latent_chunks, dim=-1)

        if normalize:
            latent = torch.nn.functional.normalize(latent, dim=1)

        return LatentRepresentation(
            latent=latent,
            sample_rate=sr,
            hop_length=self.hop_length,
            source=source,
        )

    @torch.inference_mode()
    def decode(
        self,
        latent: Union[LatentRepresentation, torch.Tensor],
        target_length: Optional[int] = None,
        chunk_duration: float = 30.0,
    ) -> torch.Tensor:
        """
        Decode latent representation back to audio.

        Args:
            latent: LatentRepresentation or raw latent tensor
            target_length: Target audio length in samples (trim/pad if specified)
            chunk_duration: Max duration per chunk in seconds (for memory efficiency)

        Returns:
            Audio tensor of shape (samples,) or (batch, samples)

        Example:
            >>> latent = codec.encode("audio.wav")
            >>> # Modify latent...
            >>> audio = codec.decode(latent)
            >>> codec.save("output.wav", audio)
        """
        if isinstance(latent, LatentRepresentation):
            latent_tensor = latent.latent
            sr = latent.sample_rate
        else:
            latent_tensor = latent
            sr = self.sample_rate

        latent_tensor = latent_tensor.to(self._device, self._torch_dtype)

        # Calculate chunk size in latent frames
        # latent frames = audio_samples / hop_length
        chunk_samples = int(chunk_duration * sr)
        chunk_frames = chunk_samples // self.hop_length

        total_frames = latent_tensor.shape[-1]

        if total_frames <= chunk_frames:
            # Short latent - decode directly
            audio = self._codec.decode(latent_tensor)  # (B, samples)
        else:
            # Long latent - decode in chunks and concatenate
            audio_chunks = []
            for start in range(0, total_frames, chunk_frames):
                end = min(start + chunk_frames, total_frames)
                chunk = latent_tensor[:, :, start:end]
                chunk_audio = self._codec.decode(chunk)
                audio_chunks.append(chunk_audio.cpu())  # Move to CPU to save VRAM
            audio = torch.cat(audio_chunks, dim=-1)

        # Trim/pad to target length
        if target_length is not None:
            current_length = audio.shape[-1]
            if current_length > target_length:
                audio = audio[..., :target_length]
            elif current_length < target_length:
                padding = target_length - current_length
                audio = torch.nn.functional.pad(audio, (0, padding))

        # Remove batch dimension if single item
        if audio.shape[0] == 1:
            audio = audio.squeeze(0)

        if audio.is_cuda:
            audio = audio.cpu()

        return audio

    def reconstruct(
        self,
        audio_input: AudioInput,
    ) -> torch.Tensor:
        """
        Encode and decode audio (test reconstruction quality).

        Args:
            audio_input: Input audio

        Returns:
            Reconstructed audio tensor
        """
        original_audio, sr, _ = self._load_audio(audio_input)
        latent = self.encode(audio_input)
        reconstructed = self.decode(latent, target_length=original_audio.shape[-1])
        return reconstructed

    def save(
        self,
        path: Union[str, Path],
        audio: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> None:
        """
        Save audio to file.

        Args:
            path: Output file path
            audio: Audio tensor
            sample_rate: Sample rate (default: codec sample rate)
        """
        sample_rate = sample_rate or self.sample_rate

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Clamp to valid range
        audio = torch.clamp(audio, -1.0, 1.0)

        torchaudio.save(str(path), audio.cpu(), sample_rate)

    def get_segments(
        self,
        latent: Union[LatentRepresentation, torch.Tensor],
        segment_frames: int,
        stride: int = 1,
    ) -> List[torch.Tensor]:
        """
        Extract sliding window segments from latent.

        Args:
            latent: Latent representation
            segment_frames: Size of each segment in frames
            stride: Stride between segments

        Returns:
            List of segment tensors
        """
        if isinstance(latent, LatentRepresentation):
            latent_tensor = latent.latent
        else:
            latent_tensor = latent

        _, _, n_frames = latent_tensor.shape
        segments = []

        for i in range(0, n_frames - segment_frames + 1, stride):
            segment = latent_tensor[:, :, i:i + segment_frames]
            segments.append(segment)

        return segments

    def concat_latents(
        self,
        latents: List[Union[LatentRepresentation, torch.Tensor]],
    ) -> LatentRepresentation:
        """
        Concatenate multiple latents along time axis.

        Args:
            latents: List of latent representations

        Returns:
            Concatenated latent
        """
        tensors = []
        for lat in latents:
            if isinstance(lat, LatentRepresentation):
                tensors.append(lat.latent)
            else:
                tensors.append(lat)

        concatenated = torch.cat(tensors, dim=-1)

        return LatentRepresentation(
            latent=concatenated,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            source="concatenated",
        )

    def frames_to_samples(self, n_frames: int) -> int:
        """Convert number of frames to audio samples."""
        return n_frames * self.hop_length

    def samples_to_frames(self, n_samples: int) -> int:
        """Convert number of audio samples to frames."""
        return n_samples // self.hop_length

    def seconds_to_frames(self, seconds: float) -> int:
        """Convert seconds to number of frames."""
        return int(seconds * self.frame_rate)

    def frames_to_seconds(self, n_frames: int) -> float:
        """Convert frames to seconds."""
        return n_frames / self.frame_rate


__all__ = [
    "DACVAECodec",
    "LatentRepresentation",
]
