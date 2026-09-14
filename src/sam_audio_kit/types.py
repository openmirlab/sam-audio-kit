"""Type definitions for sam-audio-kit."""

from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch

# Device types
DeviceType = str

# Data types for model precision
DType = Literal["float32", "float16", "bfloat16"]

# Model sizes available
ModelSize = Literal["small", "base", "large"]

# Audio input types
AudioInput = Union[str, Path, np.ndarray, torch.Tensor]

# Model name mappings
MODEL_NAME_MAP: dict[str, str] = {
    "small": "facebook/sam-audio-small",
    "base": "facebook/sam-audio-base",
    "large": "facebook/sam-audio-large",
}

# Related HuggingFace resources (for documentation)
HUGGINGFACE_RESOURCES = {
    # Main separation models (we support these)
    "models": {
        "facebook/sam-audio-small": "Smallest model (~4GB lite), fastest inference",
        "facebook/sam-audio-base": "Balanced model (~5GB lite), recommended",
        "facebook/sam-audio-large": "Largest model (~7GB lite), best quality",
    },
    # Visual-optimized models (NOT supported - we remove vision encoder)
    "models_visual": {
        "facebook/sam-audio-small-tv": "Requires vision encoder (not supported in lite mode)",
        "facebook/sam-audio-base-tv": "Requires vision encoder (not supported in lite mode)",
        "facebook/sam-audio-large-tv": "Requires vision encoder (not supported in lite mode)",
    },
    # Quality assessment model (used internally for re-ranking)
    "judge": {
        "facebook/sam-audio-judge": "Quality assessment (recall, precision, faithfulness)",
    },
}

# Default configurations
DEFAULT_SAMPLE_RATE = 48000
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


def resolve_device(device: DeviceType | None) -> DeviceType:
    """Resolve legacy automatic selection and validate explicit devices.

    Apple MLX backends and Torch MPS are out of scope for openmirlab
    packages (org canon, 2026-09-14): "auto" only ever resolves to "cuda"
    or "cpu", and any "mps"/"mps:N" request is rejected with a clear
    ValueError rather than silently routed to the (removed) MPS path.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if isinstance(device, torch.device):
        device = str(device)
    if device == "cpu":
        return "cpu"
    if device == "mps" or (isinstance(device, str) and device.startswith("mps:")):
        raise ValueError(
            "device 'mps' is not supported: Apple MLX/Torch MPS backends are "
            "out of scope for this package. Use 'auto', 'cpu', 'cuda', or 'cuda:N'."
        )
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is not available")
        return "cuda"
    if not isinstance(device, str) or not device.startswith("cuda:"):
        raise ValueError("device must be None, 'auto', 'cpu', 'cuda', or 'cuda:N'")
    index = device[5:]
    if not index.isdigit():
        raise ValueError("CUDA device index must be a non-negative integer")
    if torch.cuda.is_available():
        if int(index) < torch.cuda.device_count():
            return device
        raise RuntimeError(f"CUDA device index {index} is not available")
    raise RuntimeError("CUDA was explicitly requested but is not available")


def get_torch_dtype(dtype: DType) -> torch.dtype:
    """Convert string dtype to torch.dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype]


def get_model_name(size: str) -> str:
    """Get the HuggingFace model name for a given size.

    Args:
        size: Model size ("small", "base", "large") or full HuggingFace ID

    Returns:
        HuggingFace model ID (e.g., "facebook/sam-audio-base")
    """
    if size in MODEL_NAME_MAP:
        # TOML is the package runtime source for official model identifiers;
        # keep MODEL_NAME_MAP as a public compatibility fallback.
        from .checkpoints import checkpoint_info

        return checkpoint_info(size).get("model_id", MODEL_NAME_MAP[size])
    # If it's already a full model name, return as-is
    return size


def get_model_size(model_name: str) -> ModelSize:
    """Infer model size from model name.

    Args:
        model_name: Model size shorthand or full HuggingFace ID

    Returns:
        Model size ("small", "base", or "large")
    """
    if model_name in ("small", "base", "large"):
        return model_name  # type: ignore
    # Try to infer from model name
    model_lower = model_name.lower()
    if "small" in model_lower:
        return "small"
    elif "large" in model_lower:
        return "large"
    return "base"


def estimate_vram(size: str, lite_mode: bool, dtype: DType) -> float:
    """Estimate VRAM usage for a given configuration.

    Args:
        size: Model size or name
        lite_mode: Whether lite mode is enabled
        dtype: Data type for inference

    Returns:
        Estimated VRAM in GB
    """
    model_size = get_model_size(size)
    mode = "lite" if lite_mode else "full"
    dtype_key = "bfloat16" if dtype in ("bfloat16", "float16") else "float32"
    key = f"{mode}_{dtype_key}"
    return VRAM_ESTIMATES[model_size][key]
