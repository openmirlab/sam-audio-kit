"""Offline lifecycle and read-only cache contracts for SamAudioSession."""

import pytest

from sam_audio_kit import SamAudioSession
from sam_audio_kit.checkpoints import checkpoint_info
from sam_audio_kit.download import resolve_model_cache_path
from sam_audio_kit.types import get_model_name


class _Runtime:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.unloaded = 0

    def unload(self):
        self.unloaded += 1

    def separate(self, audio, description, **kwargs):
        return audio, description


def test_session_load_is_idempotent_release_reloads_and_close_is_terminal(monkeypatch):
    import sam_audio_kit.model as model_module

    built = []

    def build(*args, **kwargs):
        runtime = _Runtime(**kwargs)
        built.append(runtime)
        return runtime

    monkeypatch.setattr(model_module.SamAudio, "from_pretrained", build)
    session = SamAudioSession(device="cpu")
    with pytest.raises(RuntimeError, match="load"):
        session.infer("audio", "vocals")
    assert session.load() is session
    assert session.load() is session
    assert len(built) == 1
    assert built[0].kwargs["device"] == "cpu"
    assert session.release() is session
    assert built[0].unloaded == 1
    assert session.status == "released"
    session.load()
    assert len(built) == 2
    assert session.close() is session
    assert session.close() is session
    assert session.status == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        session.load()
    with pytest.raises(RuntimeError, match="load"):
        session.infer("audio", "vocals")


def test_session_context_and_failed_load_status(monkeypatch):
    import sam_audio_kit.model as model_module

    monkeypatch.setattr(model_module.SamAudio, "from_pretrained", lambda *args, **kwargs: _Runtime())
    with SamAudioSession(device="cpu") as session:
        assert session.status == "ready"
    assert session.status == "closed"

    def fail(*args, **kwargs):
        raise ValueError("gated failure")

    monkeypatch.setattr(model_module.SamAudio, "from_pretrained", fail)
    failed = SamAudioSession(device="cpu")
    with pytest.raises(ValueError, match="gated failure"):
        failed.load()
    assert failed.status == "failed"


def test_cache_info_is_read_only_and_uses_toml_default_or_custom_path(tmp_path):
    cache_root = tmp_path / "cache"
    default = SamAudioSession(device="cpu", cache_dir=cache_root)
    info = default.cache_info()
    pinned_revision = checkpoint_info("base")["source_revision"]
    assert info["path"] == str(resolve_model_cache_path("base", cache_root, revision=pinned_revision))
    assert info["model"] == checkpoint_info("base")["model_id"]
    assert info["exists"] is False
    assert not cache_root.exists()

    custom = SamAudioSession(
        device="cpu",
        cache_dir=cache_root,
        checkpoint_overrides={"model_id": "example/custom-sam"},
    ).cache_info()
    # model_id is overridden but "base"'s own pinned source_revision carries through.
    assert custom["path"] == str(
        cache_root / "models--example--custom-sam" / "snapshots" / pinned_revision
    )
    assert custom["model"] == "example/custom-sam"
    assert custom["exists"] is False


def test_toml_model_id_drives_runtime_resolution():
    assert get_model_name("base") == checkpoint_info("base")["model_id"]
