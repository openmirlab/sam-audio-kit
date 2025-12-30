"""
Main SAM-Audio inference model wrapper.

This module provides the SamAudioInfer class, which is the main entry point
for using SAM-Audio with optimized inference settings.
"""

import gc
from pathlib import Path
from typing import Any, Optional, Union

import torch

from .chunking import AudioChunker
from .inference import SeparationResult, separate_audio, load_audio
from .lite import LiteModelConfig, create_lite_model, is_lite_model
from .memory import cleanup_gpu_memory, get_gpu_memory_info, MemoryTracker
from .types import (
    AudioInput,
    DeviceType,
    DType,
    ModelSize,
    get_model_name,
    get_torch_dtype,
    estimate_vram,
)


class SamAudioInfer:
    """
    Optimized inference wrapper for SAM-Audio model.

    This class provides a simple interface for audio separation with built-in
    support for VRAM optimization, chunking, and memory management.

    Example:
        >>> # Basic usage
        >>> model = SamAudioInfer.from_pretrained("facebook/sam-audio-base")
        >>> result = model.separate("audio.wav", description="vocals")
        >>> result.save("vocals.wav", "accompaniment.wav")

        >>> # Lite mode for reduced VRAM
        >>> model = SamAudioInfer.from_pretrained(
        ...     "facebook/sam-audio-base",
        ...     lite_mode=True,
        ...     dtype="bfloat16",
        ... )

        >>> # With automatic chunking for long audio
        >>> result = model.separate(
        ...     "long_song.wav",
        ...     description="drums",
        ...     chunk_duration=25.0,
        ... )
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        device: DeviceType = "cuda",
        dtype: DType = "bfloat16",
        lite_mode: bool = False,
        chunk_duration: float = 25.0,
    ):
        """
        Initialize SamAudioInfer.

        Args:
            model: SAM-Audio model instance
            processor: SAM-Audio processor instance
            device: Device to run inference on
            dtype: Data type for inference
            lite_mode: Whether the model is in lite mode
            chunk_duration: Default chunk duration for long audio
        """
        self._model = model
        self._processor = processor
        self._device = device
        self._dtype = dtype
        self._lite_mode = lite_mode
        self._chunk_duration = chunk_duration

        # Move model to device
        torch_dtype = get_torch_dtype(dtype)
        self._model = self._model.to(device, torch_dtype)
        self._model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, ModelSize],
        lite_mode: bool = True,
        lite_config: Optional[LiteModelConfig] = None,
        device: DeviceType = "cuda",
        dtype: DType = "bfloat16",
        chunk_duration: float = 25.0,
        hf_token: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> "SamAudioInfer":
        """
        Load SAM-Audio model from HuggingFace Hub or local path.

        Args:
            model_name_or_path: Model name ("small", "base", "large"),
                               HuggingFace model ID, or local path
            lite_mode: Enable lite mode for reduced VRAM usage
            lite_config: Custom lite mode configuration
            device: Device to run inference on ("cuda", "cpu", "mps")
            dtype: Data type ("float32", "float16", "bfloat16")
            chunk_duration: Default chunk duration for long audio (seconds)
            hf_token: HuggingFace API token for gated models
            cache_dir: Directory to cache downloaded models
            verbose: Print loading progress

        Returns:
            SamAudioInfer instance ready for inference

        Example:
            >>> # Using model size shorthand
            >>> model = SamAudioInfer.from_pretrained("base", lite_mode=True)

            >>> # Using full model name
            >>> model = SamAudioInfer.from_pretrained(
            ...     "facebook/sam-audio-large",
            ...     lite_mode=True,
            ...     dtype="bfloat16",
            ... )
        """
        # Resolve model name
        if model_name_or_path in ("small", "base", "large"):
            model_name = get_model_name(model_name_or_path)  # type: ignore
            model_size = model_name_or_path
        else:
            model_name = model_name_or_path
            # Try to infer size from name
            if "small" in model_name.lower():
                model_size = "small"
            elif "large" in model_name.lower():
                model_size = "large"
            else:
                model_size = "base"

        # Estimate VRAM
        estimated_vram = estimate_vram(model_size, lite_mode, dtype)  # type: ignore
        if verbose:
            print(f"Loading {model_name}")
            print(f"  Lite mode: {lite_mode}")
            print(f"  Dtype: {dtype}")
            print(f"  Estimated VRAM: ~{estimated_vram:.1f} GB")

        # Check available VRAM
        if device == "cuda" and torch.cuda.is_available():
            gpu_info = get_gpu_memory_info()
            if gpu_info and gpu_info.free_gb < estimated_vram:
                print(
                    f"  Warning: Available VRAM ({gpu_info.free_gb:.1f} GB) "
                    f"may be insufficient (need ~{estimated_vram:.1f} GB)"
                )

        # Import SAM-Audio
        try:
            from sam_audio import SAMAudio, SAMAudioProcessor
        except ImportError:
            raise ImportError(
                "sam-audio is not installed. Install it with:\n"
                "  pip install git+https://github.com/facebookresearch/sam-audio.git"
            )

        # Load model and processor
        load_kwargs = {}
        if hf_token:
            load_kwargs["token"] = hf_token
        if cache_dir:
            load_kwargs["cache_dir"] = str(cache_dir)

        if verbose:
            print("  Loading model...")

        with MemoryTracker("Model Loading") if verbose and device == "cuda" else nullcontext():
            model = SAMAudio.from_pretrained(model_name, **load_kwargs)
            processor = SAMAudioProcessor.from_pretrained(model_name, **load_kwargs)

        # Apply lite mode optimizations
        if lite_mode:
            if verbose:
                print("  Applying lite mode optimizations...")

            if lite_config is None:
                lite_config = LiteModelConfig.aggressive()

            model = create_lite_model(model, lite_config)

            if verbose:
                print("    Removed: vision_encoder, rankers, span_predictor")

        # Create instance
        instance = cls(
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
            lite_mode=lite_mode,
            chunk_duration=chunk_duration,
        )

        if verbose:
            print("  Model ready!")
            if device == "cuda":
                gpu_info = get_gpu_memory_info()
                if gpu_info:
                    print(f"  Current VRAM usage: {gpu_info.allocated_gb:.2f} GB")

        return instance

    @property
    def model(self) -> Any:
        """Get the underlying SAM-Audio model."""
        return self._model

    @property
    def processor(self) -> Any:
        """Get the SAM-Audio processor."""
        return self._processor

    @property
    def device(self) -> DeviceType:
        """Get the current device."""
        return self._device

    @property
    def dtype(self) -> DType:
        """Get the current data type."""
        return self._dtype

    @property
    def is_lite(self) -> bool:
        """Check if model is in lite mode."""
        return self._lite_mode or is_lite_model(self._model)

    @property
    def sample_rate(self) -> int:
        """Get the model's expected sample rate."""
        return getattr(self._processor, "sampling_rate", 16000)

    def separate(
        self,
        audio: AudioInput,
        description: str,
        chunk_duration: Optional[float] = None,
        cleanup_between_chunks: bool = True,
        verbose: bool = False,
    ) -> SeparationResult:
        """
        Separate audio based on text description.

        Args:
            audio: Audio file path, numpy array, or torch tensor
            description: Text description of what to extract
                        (e.g., "vocals", "drums", "piano")
            chunk_duration: Override default chunk duration (seconds)
            cleanup_between_chunks: Clean GPU memory between chunks
            verbose: Print progress information

        Returns:
            SeparationResult with target and residual audio

        Example:
            >>> result = model.separate("song.wav", "vocals")
            >>> result.save("vocals.wav", "backing.wav")

            >>> # Extract multiple elements
            >>> vocals = model.separate("song.wav", "singing voice")
            >>> drums = model.separate("song.wav", "drums and percussion")
        """
        return separate_audio(
            model=self._model,
            processor=self._processor,
            audio_input=audio,
            description=description,
            device=self._device,
            dtype=self._dtype,
            chunk_duration=chunk_duration or self._chunk_duration,
            cleanup_between_chunks=cleanup_between_chunks,
            verbose=verbose,
        )

    def separate_batch(
        self,
        audio: AudioInput,
        descriptions: list[str],
        verbose: bool = False,
    ) -> list[SeparationResult]:
        """
        Separate audio into multiple stems based on descriptions.

        Args:
            audio: Audio file path, numpy array, or torch tensor
            descriptions: List of descriptions for each stem
            verbose: Print progress information

        Returns:
            List of SeparationResult objects

        Example:
            >>> results = model.separate_batch(
            ...     "song.wav",
            ...     ["vocals", "drums", "bass", "other"]
            ... )
            >>> for i, result in enumerate(results):
            ...     result.save(f"stem_{i}.wav")
        """
        results = []
        for i, desc in enumerate(descriptions):
            if verbose:
                print(f"Separating [{i+1}/{len(descriptions)}]: {desc}")
            result = self.separate(audio, desc, verbose=verbose)
            results.append(result)
            cleanup_gpu_memory()
        return results

    def to(self, device: DeviceType) -> "SamAudioInfer":
        """
        Move model to a different device.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        torch_dtype = get_torch_dtype(self._dtype)
        self._model = self._model.to(device, torch_dtype)
        self._device = device
        return self

    def unload(self) -> None:
        """Unload model from memory and cleanup GPU."""
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        cleanup_gpu_memory()

    def __repr__(self) -> str:
        return (
            f"SamAudioInfer("
            f"device={self._device}, "
            f"dtype={self._dtype}, "
            f"lite_mode={self.is_lite})"
        )


# Null context manager for when verbose is False
class nullcontext:
    """Simple null context manager for Python < 3.10 compatibility."""

    def __enter__(self):
        return None

    def __exit__(self, *args):
        pass
