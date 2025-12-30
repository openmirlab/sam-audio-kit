"""
Main SAM-Audio inference model wrapper.

This module provides the SamAudioInfer class, which is the main entry point
for using SAM-Audio with optimized inference settings.
"""

import gc
import os
from pathlib import Path
from typing import Any, Optional, Union

from dotenv import load_dotenv
import torch

# Load environment variables from .env file
# Searches in current directory and parent directories
load_dotenv()

from .chunking import AudioChunker
from .inference import SeparationResult, separate_audio, load_audio
from .lite import LiteModelConfig, create_lite_model, is_lite_model, get_config_description
from .memory import cleanup_gpu_memory, get_gpu_memory_info, MemoryTracker
from .types import (
    AudioInput,
    DeviceType,
    DType,
    ModelSize,
    get_model_name,
    get_model_size,
    get_torch_dtype,
    estimate_vram,
    MODEL_NAME_MAP,
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
        enable_text_ranker: bool = False,
        enable_span_predictor: bool = False,
        reranking_candidates: int = 3,
        device: DeviceType = "cuda",
        dtype: DType = "bfloat16",
        precision_config: Optional["PrecisionConfig"] = None,
        chunk_duration: float = 25.0,
        hf_token: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> "SamAudioInfer":
        """
        Load SAM-Audio model from HuggingFace Hub or local path.

        Args:
            model_name_or_path: Model size ("small", "base", "large"),
                HuggingFace model ID, or local path
            lite_mode: Enable lite mode for reduced VRAM usage
            lite_config: Custom lite mode configuration (overrides other lite settings)
            enable_text_ranker: Keep text ranker for better quality (adds ~2GB VRAM)
            enable_span_predictor: Keep span predictor for time segments (adds ~1-2GB VRAM)
            reranking_candidates: Number of candidates for text ranker (default 3)
            device: Device to run inference on ("cuda", "cpu", "mps")
            dtype: Data type ("float32", "float16", "bfloat16")
            precision_config: PrecisionConfig for fine-grained control over:
                - matmul_precision: "highest", "high", or "medium"
                - allow_tf32: Enable TF32 (~3x speedup on Ampere+ GPUs)
                - cudnn_benchmark: Enable cuDNN auto-tuner
                - cudnn_deterministic: Force reproducible results
            chunk_duration: Default chunk duration for long audio (seconds)
            hf_token: HuggingFace API token for gated models (or set HF_TOKEN env var)
            cache_dir: Directory to cache downloaded models
            verbose: Print loading progress

        Returns:
            SamAudioInfer instance ready for inference

        Example:
            >>> # Basic lite mode (most VRAM efficient, ~4-5GB)
            >>> model = SamAudioInfer.from_pretrained("base", lite_mode=True)

            >>> # With custom precision settings
            >>> from sam_audio_infer import PrecisionConfig
            >>> config = PrecisionConfig(
            ...     matmul_precision="medium",  # fastest
            ...     allow_tf32=True,
            ...     cudnn_benchmark=True,
            ... )
            >>> model = SamAudioInfer.from_pretrained("base", precision_config=config)

            >>> # Maximum quality (disable TF32)
            >>> config = PrecisionConfig(
            ...     matmul_precision="highest",
            ...     allow_tf32=False,
            ...     cudnn_deterministic=True,
            ... )
            >>> model = SamAudioInfer.from_pretrained("base", precision_config=config)
        """
        from .download import get_cache_dir, get_hf_token
        from .precision import apply_precision_config, PrecisionConfig

        # Load HF token from environment if not provided
        if hf_token is None:
            hf_token = get_hf_token()

        # Load cache dir from environment if not provided
        if cache_dir is None:
            cache_dir = get_cache_dir()

        # Apply precision configuration
        if precision_config is None:
            precision_config = PrecisionConfig.from_env()

        apply_precision_config(precision_config)

        if verbose:
            print(f"  Precision: matmul={precision_config.matmul_precision}, "
                  f"tf32={precision_config.allow_tf32}, "
                  f"cudnn_bench={precision_config.cudnn_benchmark}")

        # Resolve model name and size
        if model_name_or_path in MODEL_NAME_MAP:
            # Known size shorthand (e.g., "base", "large")
            model_name = get_model_name(model_name_or_path)
            model_size = model_name_or_path  # type: ignore
        else:
            # Full model name or path
            model_name = model_name_or_path
            model_size = get_model_size(model_name_or_path)

        # Estimate VRAM
        estimated_vram = estimate_vram(model_size, lite_mode, dtype)
        if verbose:
            print(f"Loading {model_name}")
            print(f"  Model size: {model_size}")
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
        model_kwargs = {}
        if hf_token:
            model_kwargs["token"] = hf_token
        if cache_dir:
            model_kwargs["cache_dir"] = str(cache_dir)
        # Note: SAMAudioProcessor doesn't accept cache_dir

        if verbose:
            print("  Loading model...")

        with MemoryTracker("Model Loading") if verbose and device == "cuda" else nullcontext():
            model = SAMAudio.from_pretrained(model_name, **model_kwargs)
            processor = SAMAudioProcessor.from_pretrained(model_name)

        # Apply lite mode optimizations
        if lite_mode:
            if verbose:
                print("  Applying lite mode optimizations...")

            # Build lite config from convenience parameters if not explicitly provided
            if lite_config is None:
                if enable_text_ranker and enable_span_predictor:
                    lite_config = LiteModelConfig.with_all_features(
                        reranking_candidates=reranking_candidates
                    )
                elif enable_text_ranker:
                    lite_config = LiteModelConfig.with_text_ranker(
                        reranking_candidates=reranking_candidates
                    )
                elif enable_span_predictor:
                    lite_config = LiteModelConfig.with_span_predictor()
                else:
                    lite_config = LiteModelConfig.aggressive()

            model = create_lite_model(model, lite_config)

            if verbose:
                config_desc = get_config_description(lite_config)
                print(f"    {config_desc}")

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
        return getattr(self._processor, "audio_sampling_rate", None) or getattr(self._processor, "sampling_rate", 48000)

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
