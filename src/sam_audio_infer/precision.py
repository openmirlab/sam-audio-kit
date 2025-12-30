"""
Precision configuration for SAM-Audio inference.

This module provides fine-grained control over numerical precision settings
that affect both performance and quality.

Precision Hierarchy (from highest to lowest quality):
1. float32 + highest matmul precision - Best quality, slowest
2. float32 + TF32 enabled - Good quality, faster on Ampere+ GPUs
3. bfloat16 - Excellent quality/speed balance (recommended)
4. float16 - Good speed, may have numerical instability
5. bfloat16 + medium matmul precision - Fastest, slightly lower quality

Environment Variables:
    SAM_AUDIO_MATMUL_PRECISION: "highest", "high", or "medium"
    SAM_AUDIO_ALLOW_TF32: "true" or "false"
    SAM_AUDIO_CUDNN_BENCHMARK: "true" or "false"
"""

import os
from dataclasses import dataclass
from typing import Literal, Optional

import torch


# Type definitions
MatmulPrecision = Literal["highest", "high", "medium"]


@dataclass
class PrecisionConfig:
    """
    Configuration for numerical precision settings.

    These settings affect the trade-off between speed and numerical accuracy.
    For most use cases, the defaults work well.

    Attributes:
        matmul_precision: Precision for matrix multiplications.
            - "highest": Most accurate (default for quality)
            - "high": Good balance
            - "medium": Fastest, uses TF32 internally

        allow_tf32: Enable TensorFloat-32 for matmul operations.
            TF32 uses 19 bits instead of 32, providing ~3x speedup on
            Ampere+ GPUs (RTX 30xx, 40xx) with minimal quality loss.
            Only affects float32 operations.

        cudnn_allow_tf32: Enable TF32 for cuDNN operations.
            Similar to allow_tf32 but for convolution operations.

        cudnn_benchmark: Enable cuDNN auto-tuner.
            Finds the fastest algorithm for your specific input sizes.
            Recommended for inference with fixed input sizes.
            May slow down first run but speeds up subsequent runs.

        cudnn_deterministic: Force deterministic cuDNN operations.
            Ensures reproducible results but may be slower.

        autocast_cache_enabled: Cache autocast dispatch decisions.
            Speeds up repeated operations with same shapes.

    Example:
        >>> # Maximum speed configuration
        >>> config = PrecisionConfig.fast()
        >>> apply_precision_config(config)

        >>> # Maximum quality configuration
        >>> config = PrecisionConfig.quality()
        >>> apply_precision_config(config)
    """

    matmul_precision: MatmulPrecision = "high"
    allow_tf32: bool = True
    cudnn_allow_tf32: bool = True
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    autocast_cache_enabled: bool = True

    @classmethod
    def default(cls) -> "PrecisionConfig":
        """
        Default configuration - balanced speed and quality.

        Good for most use cases with bfloat16 dtype.
        """
        return cls(
            matmul_precision="high",
            allow_tf32=True,
            cudnn_allow_tf32=True,
            cudnn_benchmark=True,
            cudnn_deterministic=False,
            autocast_cache_enabled=True,
        )

    @classmethod
    def fast(cls) -> "PrecisionConfig":
        """
        Maximum speed configuration.

        Uses TF32 and medium precision for fastest inference.
        Quality impact is minimal for audio separation.
        Best for: production deployment, batch processing.
        """
        return cls(
            matmul_precision="medium",
            allow_tf32=True,
            cudnn_allow_tf32=True,
            cudnn_benchmark=True,
            cudnn_deterministic=False,
            autocast_cache_enabled=True,
        )

    @classmethod
    def quality(cls) -> "PrecisionConfig":
        """
        Maximum quality configuration.

        Disables TF32 and uses highest precision.
        Best for: quality-critical applications, benchmarking.
        """
        return cls(
            matmul_precision="highest",
            allow_tf32=False,
            cudnn_allow_tf32=False,
            cudnn_benchmark=False,  # More consistent results
            cudnn_deterministic=True,
            autocast_cache_enabled=True,
        )

    @classmethod
    def reproducible(cls) -> "PrecisionConfig":
        """
        Configuration for reproducible results.

        Ensures same input always produces same output.
        May be slower due to deterministic algorithms.
        """
        return cls(
            matmul_precision="highest",
            allow_tf32=False,
            cudnn_allow_tf32=False,
            cudnn_benchmark=False,
            cudnn_deterministic=True,
            autocast_cache_enabled=True,
        )

    @classmethod
    def from_env(cls) -> "PrecisionConfig":
        """
        Create configuration from environment variables.

        Environment Variables:
            SAM_AUDIO_MATMUL_PRECISION: "highest", "high", or "medium"
            SAM_AUDIO_ALLOW_TF32: "true" or "false"
            SAM_AUDIO_CUDNN_BENCHMARK: "true" or "false"
            SAM_AUDIO_CUDNN_DETERMINISTIC: "true" or "false"
        """
        config = cls.default()

        # Matmul precision
        matmul_env = os.getenv("SAM_AUDIO_MATMUL_PRECISION")
        if matmul_env and matmul_env in ("highest", "high", "medium"):
            config.matmul_precision = matmul_env  # type: ignore

        # TF32
        tf32_env = os.getenv("SAM_AUDIO_ALLOW_TF32")
        if tf32_env:
            config.allow_tf32 = tf32_env.lower() in ("true", "1", "yes")
            config.cudnn_allow_tf32 = config.allow_tf32

        # cuDNN benchmark
        benchmark_env = os.getenv("SAM_AUDIO_CUDNN_BENCHMARK")
        if benchmark_env:
            config.cudnn_benchmark = benchmark_env.lower() in ("true", "1", "yes")

        # Deterministic
        deterministic_env = os.getenv("SAM_AUDIO_CUDNN_DETERMINISTIC")
        if deterministic_env:
            config.cudnn_deterministic = deterministic_env.lower() in ("true", "1", "yes")

        return config


def apply_precision_config(config: Optional[PrecisionConfig] = None) -> PrecisionConfig:
    """
    Apply precision configuration to PyTorch.

    Args:
        config: PrecisionConfig to apply. If None, uses default.

    Returns:
        The applied configuration.

    Example:
        >>> from sam_audio_infer.precision import apply_precision_config, PrecisionConfig
        >>> apply_precision_config(PrecisionConfig.fast())
    """
    if config is None:
        config = PrecisionConfig.default()

    # Set matmul precision
    torch.set_float32_matmul_precision(config.matmul_precision)

    # Set TF32 settings
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.cudnn_allow_tf32

    # Set cuDNN settings
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    torch.backends.cudnn.deterministic = config.cudnn_deterministic

    # Set autocast cache
    torch.set_autocast_cache_enabled(config.autocast_cache_enabled)

    return config


def get_current_precision() -> dict:
    """
    Get current PyTorch precision settings.

    Returns:
        Dictionary of current precision settings.
    """
    # Handle potential API compatibility issues
    try:
        matmul_precision = torch.get_float32_matmul_precision()
    except RuntimeError:
        # Fallback if there's a mixed API issue
        matmul_precision = "unknown"

    return {
        "matmul_precision": matmul_precision,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "autocast_cache_enabled": torch.is_autocast_cache_enabled(),
    }


def print_precision_info() -> None:
    """Print current precision settings in a readable format."""
    settings = get_current_precision()
    print("Current Precision Settings:")
    print(f"  matmul_precision: {settings['matmul_precision']}")
    print(f"  allow_tf32: {settings['allow_tf32']}")
    print(f"  cudnn_allow_tf32: {settings['cudnn_allow_tf32']}")
    print(f"  cudnn_benchmark: {settings['cudnn_benchmark']}")
    print(f"  cudnn_deterministic: {settings['cudnn_deterministic']}")
    print(f"  autocast_cache: {settings['autocast_cache_enabled']}")


# Convenience function for quick precision presets
def set_precision(preset: Literal["default", "fast", "quality", "reproducible"] = "default") -> PrecisionConfig:
    """
    Quick way to set precision using presets.

    Args:
        preset: One of "default", "fast", "quality", "reproducible"

    Returns:
        Applied PrecisionConfig

    Example:
        >>> from sam_audio_infer.precision import set_precision
        >>> set_precision("fast")  # Maximum speed
        >>> set_precision("quality")  # Maximum quality
    """
    configs = {
        "default": PrecisionConfig.default,
        "fast": PrecisionConfig.fast,
        "quality": PrecisionConfig.quality,
        "reproducible": PrecisionConfig.reproducible,
    }
    config = configs[preset]()
    return apply_precision_config(config)
