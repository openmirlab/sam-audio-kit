"""
SAM-Audio-Infer: Optimized inference package for Meta's SAM-Audio model.

This package provides VRAM-efficient inference for SAM-Audio with features like:
- Lite mode: Remove unused components to reduce VRAM by ~40%
- Mixed precision: bfloat16/float16 support
- Auto-chunking: Process long audio files without OOM
- Memory management: Automatic GPU cache cleanup

Example:
    >>> from sam_audio_infer import SamAudioInfer
    >>> model = SamAudioInfer.from_pretrained("facebook/sam-audio-base", lite_mode=True)
    >>> result = model.separate("audio.wav", description="vocals")

Acknowledgments:
----------------
This package builds upon:

1. SAM-Audio (Meta AI / Facebook Research)
   - Repository: https://github.com/facebookresearch/sam-audio
   - The foundational audio separation model

2. AudioGhost AI
   - Pioneered the "Lite Mode" optimization technique
   - Key insight: Vision encoder, rankers, and span predictor can be
     safely removed for audio-only separation tasks (~40% VRAM reduction)

We express our gratitude to both teams for their contributions to
open-source audio AI.
"""

__version__ = "0.1.0"

from .model import SamAudioInfer
from .lite import (
    create_lite_model,
    LiteModelConfig,
    estimate_lite_savings,
    estimate_vram_for_config,
    get_config_description,
)
from .inference import SeparationResult, separate_audio
from .chunking import AudioChunker, ChunkingConfig
from .memory import (
    cleanup_gpu_memory,
    get_gpu_memory_info,
    GPUMemoryInfo,
    MemoryTracker,
)
from .types import (
    AudioInput,
    DeviceType,
    DType,
    ModelSize,
    HUGGINGFACE_RESOURCES,
)
from .download import (
    download_model,
    download_and_warmup,
    warmup_model,
    get_cache_dir,
    list_cached_models,
    clear_cache,
)
from .precision import (
    PrecisionConfig,
    apply_precision_config,
    get_current_precision,
    print_precision_info,
)

__all__ = [
    # Main class
    "SamAudioInfer",
    # Lite mode
    "create_lite_model",
    "LiteModelConfig",
    "estimate_lite_savings",
    "estimate_vram_for_config",
    "get_config_description",
    # Inference
    "SeparationResult",
    "separate_audio",
    # Chunking
    "AudioChunker",
    "ChunkingConfig",
    # Memory
    "cleanup_gpu_memory",
    "get_gpu_memory_info",
    "GPUMemoryInfo",
    "MemoryTracker",
    # Types
    "AudioInput",
    "DeviceType",
    "DType",
    "ModelSize",
    "HUGGINGFACE_RESOURCES",
    # Download & Warmup
    "download_model",
    "download_and_warmup",
    "warmup_model",
    "get_cache_dir",
    "list_cached_models",
    "clear_cache",
    # Precision
    "PrecisionConfig",
    "apply_precision_config",
    "get_current_precision",
    "print_precision_info",
]
