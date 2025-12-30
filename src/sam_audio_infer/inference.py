"""
Inference utilities for SAM-Audio separation.

This module provides the core inference logic for audio separation,
including single-shot and chunked processing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torchaudio

from .chunking import AudioChunker, ChunkingConfig
from .lite import LiteModelConfig, is_lite_model, get_lite_config
from .memory import cleanup_gpu_memory, log_memory_usage
from .types import AudioInput, DeviceType, DType, get_torch_dtype


@dataclass
class SeparationResult:
    """Result of audio separation."""

    target: torch.Tensor  # Extracted audio (e.g., vocals)
    residual: torch.Tensor  # Remaining audio (e.g., accompaniment)
    sample_rate: int
    description: str
    processing_time: float = 0.0  # seconds
    num_chunks: int = 1

    def save(
        self,
        target_path: Union[str, Path],
        residual_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Save separation results to files.

        Args:
            target_path: Path to save extracted audio
            residual_path: Path to save residual audio (optional)
        """
        # Ensure correct shape for torchaudio (channels, samples)
        target = self.target
        if target.dim() == 1:
            target = target.unsqueeze(0)
        elif target.dim() == 2 and target.shape[0] > target.shape[1]:
            target = target.T

        torchaudio.save(str(target_path), target.cpu(), self.sample_rate)

        if residual_path is not None:
            residual = self.residual
            if residual.dim() == 1:
                residual = residual.unsqueeze(0)
            elif residual.dim() == 2 and residual.shape[0] > residual.shape[1]:
                residual = residual.T

            torchaudio.save(str(residual_path), residual.cpu(), self.sample_rate)


def load_audio(
    audio_input: AudioInput,
    target_sample_rate: Optional[int] = None,
    device: DeviceType = "cpu",
) -> tuple[torch.Tensor, int]:
    """
    Load audio from various input types.

    Args:
        audio_input: Path to audio file, numpy array, or torch tensor
        target_sample_rate: Resample to this rate if specified
        device: Device to load audio to

    Returns:
        Tuple of (audio_tensor, sample_rate)
    """
    if isinstance(audio_input, (str, Path)):
        audio, sr = torchaudio.load(str(audio_input))
    elif isinstance(audio_input, np.ndarray):
        audio = torch.from_numpy(audio_input).float()
        sr = target_sample_rate or 16000
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
    elif isinstance(audio_input, torch.Tensor):
        audio = audio_input.float()
        sr = target_sample_rate or 16000
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

    # Resample if needed
    if target_sample_rate is not None and sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
        sr = target_sample_rate

    return audio.to(device), sr


def separate_audio(
    model: Any,
    processor: Any,
    audio_input: AudioInput,
    description: str,
    device: DeviceType = "cuda",
    dtype: DType = "bfloat16",
    chunk_duration: Optional[float] = None,
    cleanup_between_chunks: bool = True,
    verbose: bool = False,
) -> SeparationResult:
    """
    Separate audio using SAM-Audio model.

    This is the main inference function that handles both short and long audio,
    automatically chunking if necessary.

    Args:
        model: SAM-Audio model (regular or lite)
        processor: SAM-Audio processor
        audio_input: Audio file path, numpy array, or torch tensor
        description: Text description of what to extract (e.g., "vocals")
        device: Device to run inference on
        dtype: Data type for inference
        chunk_duration: Chunk duration in seconds (None for auto)
        cleanup_between_chunks: Whether to cleanup GPU memory between chunks
        verbose: Print progress information

    Returns:
        SeparationResult with target and residual audio

    Example:
        >>> result = separate_audio(
        ...     model, processor,
        ...     "song.mp3",
        ...     description="vocals",
        ...     device="cuda",
        ...     dtype="bfloat16",
        ... )
        >>> result.save("vocals.wav", "accompaniment.wav")
    """
    import time
    start_time = time.time()

    # Get model's expected sample rate (SAM-Audio uses audio_sampling_rate, not sampling_rate)
    model_sample_rate = getattr(processor, "audio_sampling_rate", None) or getattr(processor, "sampling_rate", 48000)

    # Load and prepare audio
    audio, sr = load_audio(audio_input, target_sample_rate=model_sample_rate, device="cpu")

    if verbose:
        duration = audio.shape[-1] / sr
        print(f"Audio loaded: {duration:.1f}s at {sr}Hz")

    # Determine torch dtype
    torch_dtype = get_torch_dtype(dtype)

    # Get lite config if available
    lite_config = get_lite_config(model) if is_lite_model(model) else None

    # Determine chunk duration
    if chunk_duration is None:
        chunk_duration = 25.0  # Default

    # Check if chunking is needed
    chunker = AudioChunker(
        chunk_duration=chunk_duration,
        sample_rate=sr,
        overlap_duration=0.5,  # Small overlap for smoother transitions
        crossfade_duration=0.1,
    )

    needs_chunking = chunker.needs_chunking(audio.squeeze(0))
    num_chunks = chunker.get_num_chunks(audio.squeeze(0)) if needs_chunking else 1

    if verbose:
        print(f"Processing: {num_chunks} chunk(s), lite_mode={lite_config is not None}")
        if torch.cuda.is_available():
            log_memory_usage("Before inference")

    # Prepare inference parameters
    inference_kwargs = {
        "predict_spans": False,
        "reranking_candidates": 1,
    }
    if lite_config is not None:
        inference_kwargs["predict_spans"] = lite_config.predict_spans
        inference_kwargs["reranking_candidates"] = lite_config.reranking_candidates

    if not needs_chunking:
        # Single-shot processing
        result = _process_single(
            model=model,
            processor=processor,
            audio=audio,
            description=description,
            device=device,
            dtype=torch_dtype,
            inference_kwargs=inference_kwargs,
        )
        target, residual = result
    else:
        # Chunked processing
        target_chunks = []
        residual_chunks = []

        audio_1d = audio.squeeze(0) if audio.dim() > 1 else audio

        for chunk in chunker.chunk(audio_1d, sample_rate=sr):
            if verbose:
                print(f"  Processing chunk {chunk.chunk_index + 1}/{chunk.total_chunks}")

            # Reshape chunk for processing
            chunk_audio = chunk.data
            if chunk_audio.dim() == 2:
                chunk_audio = chunk_audio.T  # (channels, samples)
            else:
                chunk_audio = chunk_audio.unsqueeze(0)  # (1, samples)

            chunk_target, chunk_residual = _process_single(
                model=model,
                processor=processor,
                audio=chunk_audio,
                description=description,
                device=device,
                dtype=torch_dtype,
                inference_kwargs=inference_kwargs,
            )

            target_chunks.append(chunk_target.cpu())
            residual_chunks.append(chunk_residual.cpu())

            if cleanup_between_chunks:
                cleanup_gpu_memory()

        # Merge chunks
        target, residual = chunker.merge_results(
            list(zip(target_chunks, residual_chunks)),
            crossfade=True,
        )

    # Clamp output to valid range
    target = torch.clamp(target, -1.0, 1.0)
    residual = torch.clamp(residual, -1.0, 1.0)

    processing_time = time.time() - start_time

    if verbose:
        print(f"Processing complete in {processing_time:.1f}s")
        if torch.cuda.is_available():
            log_memory_usage("After inference")

    return SeparationResult(
        target=target,
        residual=residual,
        sample_rate=sr,
        description=description,
        processing_time=processing_time,
        num_chunks=num_chunks,
    )


def _process_single(
    model: Any,
    processor: Any,
    audio: torch.Tensor,
    description: str,
    device: DeviceType,
    dtype: torch.dtype,
    inference_kwargs: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Process a single audio segment.

    Args:
        model: SAM-Audio model
        processor: SAM-Audio processor
        audio: Audio tensor (channels, samples)
        description: Text description
        device: Device for inference
        dtype: Data type for inference
        inference_kwargs: Additional inference parameters

    Returns:
        Tuple of (target, residual) tensors
    """
    # Prepare batch (processor expects descriptions first, then audios)
    batch = processor(
        descriptions=[description],
        audios=[audio],
    )

    # Move to device (Batch object has a to() method)
    batch = batch.to(device)

    # Run inference
    with torch.inference_mode():
        with torch.autocast(device_type=device if device != "mps" else "cpu", dtype=dtype):
            result = model.separate(batch, **inference_kwargs)

    # Extract results
    target = result.target[0] if hasattr(result, "target") else result[0][0]
    residual = result.residual[0] if hasattr(result, "residual") else result[1][0]

    return target, residual
