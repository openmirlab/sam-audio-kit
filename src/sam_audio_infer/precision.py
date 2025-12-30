"""
Precision configuration for SAM-Audio inference.

This module provides fine-grained control over numerical precision settings
that affect both performance and quality.

Environment Variables (defaults):
    SAM_AUDIO_TF32: "true" or "false" (default: true)
    SAM_AUDIO_MATMUL_PRECISION: "highest", "high", or "medium" (default: high)
    SAM_AUDIO_CUDNN_BENCHMARK: "true" or "false" (default: true)

CLI Arguments (override env vars):
    --tf32 / --no-tf32
    --matmul-precision highest|high|medium
    --cudnn-benchmark / --no-cudnn-benchmark
    --deterministic
"""

import os
from dataclasses import dataclass
from typing import Literal

import torch


# Type definitions
MatmulPrecision = Literal["highest", "high", "medium"]


@dataclass
class PrecisionConfig:
    """
    Configuration for numerical precision settings.

    Attributes:
        matmul_precision: Precision for matrix multiplications.
            - "highest": Most accurate, slowest
            - "high": Good balance (default)
            - "medium": Fastest, uses TF32 internally

        allow_tf32: Enable TensorFloat-32 for matmul operations.
            TF32 provides ~3x speedup on Ampere+ GPUs (RTX 30xx, 40xx)
            with minimal quality loss. Only affects float32 operations.

        cudnn_allow_tf32: Enable TF32 for cuDNN convolution operations.

        cudnn_benchmark: Enable cuDNN auto-tuner.
            Finds fastest algorithm for your input sizes.
            Slower first run, faster subsequent runs.

        cudnn_deterministic: Force deterministic cuDNN operations.
            Ensures reproducible results but may be slower.

    Example:
        >>> # Fast inference
        >>> config = PrecisionConfig(
        ...     matmul_precision="medium",
        ...     allow_tf32=True,
        ... )
        >>> apply_precision_config(config)

        >>> # Maximum quality
        >>> config = PrecisionConfig(
        ...     matmul_precision="highest",
        ...     allow_tf32=False,
        ... )
    """

    matmul_precision: MatmulPrecision = "high"
    allow_tf32: bool = True
    cudnn_allow_tf32: bool = True
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False

    @classmethod
    def from_env(cls) -> "PrecisionConfig":
        """
        Create configuration from environment variables.

        Environment Variables:
            SAM_AUDIO_TF32: "true" or "false" (default: true)
            SAM_AUDIO_MATMUL_PRECISION: "highest", "high", or "medium" (default: high)
            SAM_AUDIO_CUDNN_BENCHMARK: "true" or "false" (default: true)
            SAM_AUDIO_DETERMINISTIC: "true" or "false" (default: false)
        """
        config = cls()

        # TF32
        tf32_env = os.getenv("SAM_AUDIO_TF32")
        if tf32_env is not None:
            config.allow_tf32 = tf32_env.lower() in ("true", "1", "yes")
            config.cudnn_allow_tf32 = config.allow_tf32

        # Matmul precision
        matmul_env = os.getenv("SAM_AUDIO_MATMUL_PRECISION")
        if matmul_env and matmul_env in ("highest", "high", "medium"):
            config.matmul_precision = matmul_env  # type: ignore

        # cuDNN benchmark
        benchmark_env = os.getenv("SAM_AUDIO_CUDNN_BENCHMARK")
        if benchmark_env is not None:
            config.cudnn_benchmark = benchmark_env.lower() in ("true", "1", "yes")

        # Deterministic
        deterministic_env = os.getenv("SAM_AUDIO_DETERMINISTIC")
        if deterministic_env is not None:
            config.cudnn_deterministic = deterministic_env.lower() in ("true", "1", "yes")

        return config


def apply_precision_config(config: PrecisionConfig) -> None:
    """
    Apply precision configuration to PyTorch.

    Args:
        config: PrecisionConfig to apply.

    Example:
        >>> config = PrecisionConfig(matmul_precision="medium", allow_tf32=True)
        >>> apply_precision_config(config)
    """
    # Set matmul precision
    torch.set_float32_matmul_precision(config.matmul_precision)

    # Set TF32 settings
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.cudnn_allow_tf32

    # Set cuDNN settings
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    torch.backends.cudnn.deterministic = config.cudnn_deterministic


def get_current_precision() -> dict:
    """
    Get current PyTorch precision settings.

    Returns:
        Dictionary of current precision settings.
    """
    try:
        matmul_precision = torch.get_float32_matmul_precision()
    except RuntimeError:
        matmul_precision = "unknown"

    return {
        "matmul_precision": matmul_precision,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


def print_precision_info() -> None:
    """Print current precision settings."""
    settings = get_current_precision()
    print("Precision Settings:")
    print(f"  matmul_precision: {settings['matmul_precision']}")
    print(f"  tf32: {settings['allow_tf32']}")
    print(f"  cudnn_benchmark: {settings['cudnn_benchmark']}")
    print(f"  deterministic: {settings['cudnn_deterministic']}")
