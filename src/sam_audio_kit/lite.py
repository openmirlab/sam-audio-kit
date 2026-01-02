"""
Lite mode optimizations for SAM-Audio.

This module provides functionality to create memory-optimized versions of SAM-Audio
by removing unused components that are not needed for audio-only inference.

Components removed in lite mode:
- vision_encoder: ~2GB savings (not needed for audio separation)
- visual_ranker: ~2GB savings (for visual quality ranking)
- text_ranker: ~2GB savings (for reranking results)
- span_predictor: ~1-2GB savings (for span/segment prediction)

Total VRAM reduction: ~40% compared to full model

Acknowledgments:
----------------
This optimization technique was pioneered in the AudioGhost AI project.
The key insight that SAM-Audio's vision encoder and rankers can be safely
removed for audio-only tasks was first implemented there.

References:
-----------
- SAM-Audio: https://github.com/facebookresearch/sam-audio
- AudioGhost AI: Original implementation of lite mode optimization
"""

import gc
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class LiteModelConfig:
    """Configuration for lite model creation.

    This config controls which components are removed from the model and
    how inference behaves. Use the class methods to create common configurations.

    VRAM Estimates (with bfloat16):
    - aggressive(): ~4-5 GB (base model)
    - with_text_ranker(): ~6-7 GB (better quality reranking)
    - with_span_predictor(): ~6-7 GB (time segment prediction)
    - with_all_features(): ~8-9 GB (text ranker + span predictor)
    """

    # Components to remove
    remove_vision_encoder: bool = True
    remove_visual_ranker: bool = True
    remove_text_ranker: bool = True
    remove_span_predictor: bool = True

    # Inference optimizations
    predict_spans: bool = False
    reranking_candidates: int = 1

    # Memory management
    cleanup_after_removal: bool = True

    def __post_init__(self):
        # Validate configuration
        if self.reranking_candidates < 1:
            raise ValueError("reranking_candidates must be at least 1")

        # Auto-adjust inference settings based on component availability
        if not self.remove_text_ranker and self.reranking_candidates == 1:
            # Text ranker is kept but reranking_candidates is 1 - warn user
            pass  # They may want this intentionally

        if not self.remove_span_predictor and not self.predict_spans:
            # Span predictor is kept but predict_spans is False - warn user
            pass  # They may want this intentionally

    @property
    def components_to_remove(self) -> list[str]:
        """Get list of component names to remove."""
        components = []
        if self.remove_vision_encoder:
            components.append("vision_encoder")
        if self.remove_visual_ranker:
            components.append("visual_ranker")
        if self.remove_text_ranker:
            components.append("text_ranker")
        if self.remove_span_predictor:
            components.extend(["span_predictor", "span_predictor_transform"])
        return components

    @property
    def has_text_ranker(self) -> bool:
        """Check if text ranker is enabled."""
        return not self.remove_text_ranker

    @property
    def has_span_predictor(self) -> bool:
        """Check if span predictor is enabled."""
        return not self.remove_span_predictor

    @classmethod
    def aggressive(cls) -> "LiteModelConfig":
        """Create aggressive lite config (remove all optional components).

        VRAM: ~4-5 GB (base model with bfloat16)

        This is the most memory-efficient configuration, removing all
        components not strictly needed for audio separation.
        """
        return cls(
            remove_vision_encoder=True,
            remove_visual_ranker=True,
            remove_text_ranker=True,
            remove_span_predictor=True,
            predict_spans=False,
            reranking_candidates=1,
        )

    @classmethod
    def conservative(cls) -> "LiteModelConfig":
        """Create conservative lite config (only remove vision encoder).

        VRAM: ~8-9 GB (base model with bfloat16)

        Keeps text ranker and span predictor for better quality.
        """
        return cls(
            remove_vision_encoder=True,
            remove_visual_ranker=False,
            remove_text_ranker=False,
            remove_span_predictor=False,
            predict_spans=True,
            reranking_candidates=3,
        )

    @classmethod
    def with_text_ranker(cls, reranking_candidates: int = 3) -> "LiteModelConfig":
        """Create lite config with text ranker enabled for better quality.

        VRAM: ~6-7 GB (base model with bfloat16)

        The text ranker improves separation quality by reranking multiple
        candidates. Higher reranking_candidates = better quality but slower.

        Args:
            reranking_candidates: Number of candidates to generate and rerank.
                                  Recommended: 3-5 for good quality/speed balance.
        """
        return cls(
            remove_vision_encoder=True,
            remove_visual_ranker=True,
            remove_text_ranker=False,  # Keep text ranker
            remove_span_predictor=True,
            predict_spans=False,
            reranking_candidates=reranking_candidates,
        )

    @classmethod
    def with_span_predictor(cls) -> "LiteModelConfig":
        """Create lite config with span predictor enabled.

        VRAM: ~6-7 GB (base model with bfloat16)

        The span predictor can identify time segments where the target
        sound is present. Useful for locating specific sounds in audio.
        """
        return cls(
            remove_vision_encoder=True,
            remove_visual_ranker=True,
            remove_text_ranker=True,
            remove_span_predictor=False,  # Keep span predictor
            predict_spans=True,
            reranking_candidates=1,
        )

    @classmethod
    def with_all_features(cls, reranking_candidates: int = 3) -> "LiteModelConfig":
        """Create lite config with both text ranker and span predictor.

        VRAM: ~8-9 GB (base model with bfloat16)

        This provides the best quality while still removing the vision
        encoder which is not needed for audio-only tasks.

        Args:
            reranking_candidates: Number of candidates to generate and rerank.
        """
        return cls(
            remove_vision_encoder=True,
            remove_visual_ranker=True,
            remove_text_ranker=False,  # Keep text ranker
            remove_span_predictor=False,  # Keep span predictor
            predict_spans=True,
            reranking_candidates=reranking_candidates,
        )


def _get_vision_encoder_dim(model: Any) -> int:
    """Extract the vision encoder output dimension before removal."""
    # First, try to get the dimension from align_masked_video layer
    # This is the most reliable source since it's what the model actually expects
    if hasattr(model, "align_masked_video") and model.align_masked_video is not None:
        if hasattr(model.align_masked_video, "conv"):
            # Conv1d weight shape: [out_channels, in_channels, kernel_size]
            return model.align_masked_video.conv.weight.shape[1]

    # Fallback: try vision encoder config
    if hasattr(model, "vision_encoder") and model.vision_encoder is not None:
        if hasattr(model.vision_encoder, "config"):
            if hasattr(model.vision_encoder.config, "hidden_size"):
                return model.vision_encoder.config.hidden_size

    # Default to 1024 which is the expected dimension for SAM-Audio base model
    return 1024


def _create_dummy_video_features_fn(vision_dim: int):
    """Create a replacement function for _get_video_features."""

    def _get_video_features_lite(self, video, audio_features):
        """Lite version that returns zeros instead of computing video features."""
        B, T, _ = audio_features.shape
        return audio_features.new_zeros(B, vision_dim, T)

    return _get_video_features_lite


def create_lite_model(
    model: Any,
    config: Optional[LiteModelConfig] = None,
) -> Any:
    """
    Create a memory-optimized lite version of SAM-Audio model.

    This function removes unused components from the model to reduce VRAM usage
    by approximately 40%. The modified model can still perform audio separation
    but cannot use video inputs or advanced reranking features.

    Args:
        model: The SAM-Audio model to optimize
        config: Lite model configuration (default: aggressive)

    Returns:
        The modified model with reduced memory footprint

    Example:
        >>> from sam_audio import SAMAudio
        >>> model = SAMAudio.from_pretrained("facebook/sam-audio-base")
        >>> lite_model = create_lite_model(model)
    """
    if config is None:
        config = LiteModelConfig.aggressive()

    # Store vision encoder dimension before removal
    vision_dim = _get_vision_encoder_dim(model)

    # Remove vision encoder
    if config.remove_vision_encoder and hasattr(model, "vision_encoder"):
        if model.vision_encoder is not None:
            del model.vision_encoder
            model.vision_encoder = None

            # Store dimension for the replacement function
            model._vision_encoder_dim = vision_dim

            # Replace _get_video_features method
            if hasattr(model, "_get_video_features"):
                import types
                model._get_video_features = types.MethodType(
                    _create_dummy_video_features_fn(vision_dim), model
                )

    # Remove visual ranker
    if config.remove_visual_ranker and hasattr(model, "visual_ranker"):
        if model.visual_ranker is not None:
            del model.visual_ranker
            model.visual_ranker = None

    # Remove text ranker
    if config.remove_text_ranker and hasattr(model, "text_ranker"):
        if model.text_ranker is not None:
            del model.text_ranker
            model.text_ranker = None

    # Remove span predictor
    if config.remove_span_predictor:
        if hasattr(model, "span_predictor") and model.span_predictor is not None:
            del model.span_predictor
            model.span_predictor = None

        if hasattr(model, "span_predictor_transform") and model.span_predictor_transform is not None:
            del model.span_predictor_transform
            model.span_predictor_transform = None

    # Store config for inference
    model._lite_config = config

    # Cleanup memory
    if config.cleanup_after_removal:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model


def estimate_lite_savings(model_size: str = "base") -> dict[str, float]:
    """
    Estimate VRAM savings from lite mode.

    Args:
        model_size: Model size ("small", "base", or "large")

    Returns:
        Dictionary with estimated savings per component
    """
    # Approximate savings based on model architecture
    savings_map = {
        "small": {
            "vision_encoder": 1.5,
            "visual_ranker": 1.5,
            "text_ranker": 1.5,
            "span_predictor": 0.5,
            "total": 5.0,
        },
        "base": {
            "vision_encoder": 2.0,
            "visual_ranker": 2.0,
            "text_ranker": 2.0,
            "span_predictor": 1.0,
            "total": 7.0,
        },
        "large": {
            "vision_encoder": 3.0,
            "visual_ranker": 3.0,
            "text_ranker": 3.0,
            "span_predictor": 2.0,
            "total": 11.0,
        },
    }
    return savings_map.get(model_size, savings_map["base"])


def estimate_vram_for_config(
    config: LiteModelConfig,
    model_size: str = "base",
    dtype: str = "bfloat16",
) -> float:
    """
    Estimate VRAM usage for a specific lite configuration.

    Args:
        config: LiteModelConfig instance
        model_size: Model size ("small", "base", or "large")
        dtype: Data type ("float32", "float16", "bfloat16")

    Returns:
        Estimated VRAM usage in GB

    Example:
        >>> config = LiteModelConfig.with_text_ranker()
        >>> vram = estimate_vram_for_config(config, "base", "bfloat16")
        >>> print(f"Estimated VRAM: {vram:.1f} GB")
    """
    # Base VRAM usage (full model in float32)
    base_vram = {
        "small": 10.0,
        "base": 13.0,
        "large": 20.0,
    }.get(model_size, 13.0)

    # Component sizes in float32
    component_sizes = estimate_lite_savings(model_size)

    # Calculate savings from removed components
    savings = 0.0
    if config.remove_vision_encoder:
        savings += component_sizes["vision_encoder"]
    if config.remove_visual_ranker:
        savings += component_sizes["visual_ranker"]
    if config.remove_text_ranker:
        savings += component_sizes["text_ranker"]
    if config.remove_span_predictor:
        savings += component_sizes["span_predictor"]

    # Apply dtype multiplier
    dtype_multiplier = {
        "float32": 1.0,
        "float16": 0.5,
        "bfloat16": 0.5,
    }.get(dtype, 0.5)

    estimated = (base_vram - savings) * dtype_multiplier
    return max(estimated, 1.0)  # Minimum 1 GB


def get_config_description(config: LiteModelConfig) -> str:
    """
    Get a human-readable description of what's enabled/disabled.

    Args:
        config: LiteModelConfig instance

    Returns:
        Description string
    """
    removed = config.components_to_remove
    kept = []

    if not config.remove_text_ranker:
        kept.append(f"text_ranker (reranking_candidates={config.reranking_candidates})")
    if not config.remove_span_predictor:
        kept.append(f"span_predictor (predict_spans={config.predict_spans})")

    parts = []
    if removed:
        parts.append(f"Removed: {', '.join(removed)}")
    if kept:
        parts.append(f"Kept: {', '.join(kept)}")

    return " | ".join(parts) if parts else "No changes"


def is_lite_model(model: Any) -> bool:
    """Check if a model has been converted to lite mode."""
    return hasattr(model, "_lite_config") and model._lite_config is not None


def get_lite_config(model: Any) -> Optional[LiteModelConfig]:
    """Get the lite config from a model, if available."""
    if is_lite_model(model):
        return model._lite_config
    return None
