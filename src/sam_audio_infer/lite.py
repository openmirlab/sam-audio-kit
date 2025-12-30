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
    """Configuration for lite model creation."""

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

    @classmethod
    def aggressive(cls) -> "LiteModelConfig":
        """Create aggressive lite config (remove all optional components)."""
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
        """Create conservative lite config (only remove vision encoder)."""
        return cls(
            remove_vision_encoder=True,
            remove_visual_ranker=False,
            remove_text_ranker=False,
            remove_span_predictor=False,
            predict_spans=False,
            reranking_candidates=1,
        )


def _get_vision_encoder_dim(model: Any) -> int:
    """Extract the vision encoder output dimension before removal."""
    if hasattr(model, "vision_encoder") and model.vision_encoder is not None:
        # Try to get the output dimension from the vision encoder
        if hasattr(model.vision_encoder, "config"):
            if hasattr(model.vision_encoder.config, "hidden_size"):
                return model.vision_encoder.config.hidden_size
        # Fallback to common dimensions
        return 768
    return 768  # Default dimension


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


def is_lite_model(model: Any) -> bool:
    """Check if a model has been converted to lite mode."""
    return hasattr(model, "_lite_config") and model._lite_config is not None


def get_lite_config(model: Any) -> Optional[LiteModelConfig]:
    """Get the lite config from a model, if available."""
    if is_lite_model(model):
        return model._lite_config
    return None
