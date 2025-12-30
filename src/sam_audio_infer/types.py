"""Type definitions for sam-audio-infer."""

from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch

# Device types
DeviceType = Literal["cuda", "cpu", "mps"]

# Data types for model precision
DType = Literal["float32", "float16", "bfloat16"]

# Model sizes available
ModelSize = Literal["small", "base", "large"]

# Audio input types
AudioInput = Union[str, Path, np.ndarray, torch.Tensor]

# Model name mappings
MODEL_NAME_MAP: dict[ModelSize, str] = {
    "small": "facebook/sam-audio-small",
    "base": "facebook/sam-audio-base",
    "large": "facebook/sam-audio-large",
}

# Default configurations
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_DURATION = 25.0  # seconds
DEFAULT_DTYPE: DType = "bfloat16"

# VRAM estimates (in GB) for different configurations
VRAM_ESTIMATES: dict[str, dict[str, float]] = {
    "small": {
        "full_float32": 10.0,
        "full_bfloat16": 6.0,
        "lite_float32": 6.0,
        "lite_bfloat16": 4.0,
    },
    "base": {
        "full_float32": 13.0,
        "full_bfloat16": 7.0,
        "lite_float32": 8.0,
        "lite_bfloat16": 5.0,
    },
    "large": {
        "full_float32": 20.0,
        "full_bfloat16": 10.0,
        "lite_float32": 12.0,
        "lite_bfloat16": 7.0,
    },
}


def get_torch_dtype(dtype: DType) -> torch.dtype:
    """Convert string dtype to torch.dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype]


def get_model_name(size: ModelSize) -> str:
    """Get the HuggingFace model name for a given size."""
    return MODEL_NAME_MAP[size]


def estimate_vram(size: ModelSize, lite_mode: bool, dtype: DType) -> float:
    """Estimate VRAM usage for a given configuration."""
    mode = "lite" if lite_mode else "full"
    dtype_key = "bfloat16" if dtype in ("bfloat16", "float16") else "float32"
    key = f"{mode}_{dtype_key}"
    return VRAM_ESTIMATES[size][key]
