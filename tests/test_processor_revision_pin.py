"""Gap 1: Processor config/tokenizer fetches must honor an explicit `revision`
override (falling back to the class default otherwise), the same idiom
`sam_audio/model/base.py`'s `BaseModel._from_pretrained` already uses. Before
this fix, `Processor._get_config`/`Processor.from_pretrained` had no
per-call `revision` parameter at all, so `SAMAudioProcessor.from_pretrained`
always floated on the class-level default regardless of any pinned revision
computed by the caller."""

import json

from sam_audio_kit.sam_audio.processor import SAMAudioProcessor, SAMAudioJudgeProcessor


def _fake_config_payload():
    # `hop_length` is a derived DACVAEConfig property (prod of encoder_rates),
    # not a constructor argument -- only pass real constructor fields.
    return {
        "audio_codec": {"sample_rate": 48_000, "codebook_dim": 128},
    }


def test_get_config_uses_explicit_revision_over_class_default(tmp_path, monkeypatch):
    calls = []

    def fake_hf_hub_download(*, repo_id, filename, revision):
        calls.append({"repo_id": repo_id, "filename": filename, "revision": revision})
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(_fake_config_payload()))
        return str(config_path)

    monkeypatch.setattr(
        "sam_audio_kit.sam_audio.processor.hf_hub_download", fake_hf_hub_download
    )

    SAMAudioProcessor.from_pretrained("facebook/sam-audio-base", revision="pinned-commit")

    assert calls == [
        {
            "repo_id": "facebook/sam-audio-base",
            "filename": "config.json",
            "revision": "pinned-commit",
        }
    ]


def test_get_config_falls_back_to_class_revision_when_not_passed(tmp_path, monkeypatch):
    calls = []

    def fake_hf_hub_download(*, repo_id, filename, revision):
        calls.append(revision)
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(_fake_config_payload()))
        return str(config_path)

    monkeypatch.setattr(
        "sam_audio_kit.sam_audio.processor.hf_hub_download", fake_hf_hub_download
    )

    SAMAudioProcessor.from_pretrained("facebook/sam-audio-base")

    # SAMAudioProcessor.revision is None -- the class-level fallback -- and no
    # explicit revision was passed, so hf_hub_download must see None, not
    # silently drift to "main".
    assert calls == [None]


def test_judge_processor_from_pretrained_pins_config_and_tokenizer_fetches_explicitly(
    tmp_path, monkeypatch
):
    """An explicit revision= must reach BOTH the config fetch and the
    tokenizer fetch -- the tokenizer call was a second, separate floating
    fetch against the same repo before this fix."""
    hf_hub_calls = []
    tokenizer_calls = []

    def fake_hf_hub_download(*, repo_id, filename, revision):
        hf_hub_calls.append(revision)
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(_fake_config_payload()))
        return str(config_path)

    class _FakeTokenizer:
        pass

    def fake_tokenizer_from_pretrained(model_name_or_path, revision=None):
        tokenizer_calls.append(revision)
        return _FakeTokenizer()

    monkeypatch.setattr(
        "sam_audio_kit.sam_audio.processor.hf_hub_download", fake_hf_hub_download
    )
    monkeypatch.setattr(
        "sam_audio_kit.sam_audio.processor.AutoTokenizer.from_pretrained",
        fake_tokenizer_from_pretrained,
    )

    SAMAudioJudgeProcessor.from_pretrained(
        "facebook/sam-audio-judge", revision="explicit-pin"
    )

    assert hf_hub_calls == ["explicit-pin"]
    assert tokenizer_calls == ["explicit-pin"]


def test_judge_processor_from_pretrained_falls_back_to_class_revision_for_both_fetches(
    tmp_path, monkeypatch
):
    """Without an explicit revision=, both fetches must fall back to the SAME
    resolved class-level revision -- whatever `SAMAudioJudgeProcessor.revision`
    currently is (see the separate judge-catalog test for what that value
    should equal)."""
    hf_hub_calls = []
    tokenizer_calls = []

    def fake_hf_hub_download(*, repo_id, filename, revision):
        hf_hub_calls.append(revision)
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(_fake_config_payload()))
        return str(config_path)

    class _FakeTokenizer:
        pass

    def fake_tokenizer_from_pretrained(model_name_or_path, revision=None):
        tokenizer_calls.append(revision)
        return _FakeTokenizer()

    monkeypatch.setattr(
        "sam_audio_kit.sam_audio.processor.hf_hub_download", fake_hf_hub_download
    )
    monkeypatch.setattr(
        "sam_audio_kit.sam_audio.processor.AutoTokenizer.from_pretrained",
        fake_tokenizer_from_pretrained,
    )

    SAMAudioJudgeProcessor.from_pretrained("facebook/sam-audio-judge")

    expected = SAMAudioJudgeProcessor.revision
    assert hf_hub_calls == [expected]
    assert tokenizer_calls == [expected]


def test_model_from_pretrained_reuses_pinned_revision_for_both_model_and_processor(
    monkeypatch
):
    """SamAudio.from_pretrained's own model_kwargs['revision'] pin must be the
    exact value handed to SAMAudioProcessor.from_pretrained too -- not
    recomputed, not left unpinned."""
    import sys
    import types

    import torch

    import sam_audio_kit.model as model_module

    calls = {}

    class _FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sample_rate = 48_000
            self.dummy = torch.nn.Parameter(torch.zeros(1))

    class _FakeSAMAudio:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls["model_revision"] = kwargs.get("revision")
            return _FakeModel()

    class _FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_name, revision=None):
            calls["processor_revision"] = revision
            return object()

    monkeypatch.setattr(
        model_module, "create_lite_model", lambda model, lite_config: model
    )
    monkeypatch.setattr(
        model_module, "get_config_description", lambda lite_config: "fake"
    )

    fake_sam_audio_module = types.ModuleType("sam_audio_kit.sam_audio")
    fake_sam_audio_module.SAMAudio = _FakeSAMAudio
    fake_sam_audio_module.SAMAudioProcessor = _FakeProcessor
    monkeypatch.setitem(sys.modules, "sam_audio_kit.sam_audio", fake_sam_audio_module)

    instance = model_module.SamAudio.from_pretrained(
        "base", device="cpu", verbose=False
    )

    from sam_audio_kit.checkpoints import checkpoint_info

    pinned = checkpoint_info("base")["source_revision"]
    assert calls["model_revision"] == pinned
    assert calls["processor_revision"] == pinned
    assert instance.processor is not None
