"""Loading only what lite mode keeps: the skip set and the checkpoint filter."""
from sam_audio_kit.lite import LiteModelConfig, components_to_skip
from sam_audio_kit.sam_audio.model.base import drop_skipped


def test_aggressive_skips_every_optional_component():
    assert components_to_skip(LiteModelConfig.aggressive()) == frozenset(
        {"vision_encoder", "visual_ranker", "text_ranker", "span_predictor"}
    )


def test_enabled_features_are_not_skipped():
    assert "text_ranker" not in components_to_skip(LiteModelConfig.with_text_ranker())
    assert "span_predictor" not in components_to_skip(LiteModelConfig.with_span_predictor())
    both = components_to_skip(LiteModelConfig.with_all_features())
    assert both == frozenset({"vision_encoder", "visual_ranker"})


def test_drop_skipped_removes_only_those_prefixes():
    sd = {"vision_encoder.a": 1, "vision_encoder_extra.b": 2, "transformer.c": 3}
    out = drop_skipped(sd, frozenset({"vision_encoder"}))
    assert out == {"vision_encoder_extra.b": 2, "transformer.c": 3}
    assert drop_skipped(sd, frozenset()) is sd
