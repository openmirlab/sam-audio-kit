"""Gap 2: SAMAudioJudgeModel/SAMAudioJudgeProcessor must resolve their
default `revision` from the package's own checkpoint catalog
(`checkpoints.toml`'s `[models.judge].source_revision`), not the hardcoded
moving branch ref `"sam_audio"` -- and a live judge-model load must verify
its downloaded checkpoint's digest against that same catalog entry."""

import hashlib
import json

import pytest
import torch

from sam_audio_kit.checkpoints import ChecksumMismatchError, checkpoint_info
from sam_audio_kit.sam_audio.model.base import BaseModel
from sam_audio_kit.sam_audio.model.judge import SAMAudioJudgeModel
from sam_audio_kit.sam_audio.processor import SAMAudioJudgeProcessor

JUDGE_PINNED_REVISION = "d461b0b9bd4d139966a1a101fa147427eb46638d"


def test_judge_processor_revision_defaults_to_catalog_pinned_commit_not_sam_audio_branch():
    assert SAMAudioJudgeProcessor.revision == JUDGE_PINNED_REVISION
    assert SAMAudioJudgeProcessor.revision == checkpoint_info("judge")["source_revision"]
    assert SAMAudioJudgeProcessor.revision != "sam_audio"


def test_judge_model_revision_defaults_to_catalog_pinned_commit_not_sam_audio_branch():
    assert SAMAudioJudgeModel.revision == JUDGE_PINNED_REVISION
    assert SAMAudioJudgeModel.revision == checkpoint_info("judge")["source_revision"]
    assert SAMAudioJudgeModel.revision != "sam_audio"
    assert SAMAudioJudgeModel.catalog_key == "judge"


# --- live-load-time digest verification (BaseModel._from_pretrained) ---


class _FakeConfig:
    def __init__(self, **kwargs):
        pass


class _FixtureModel(BaseModel):
    """Minimal BaseModel subclass -- no transformer/codec weights, just
    enough to exercise `_from_pretrained`'s snapshot/verify/load_state_dict
    path without SAMAudioJudgeModel's heavy real submodules."""

    config_cls = _FakeConfig
    revision = "unused"
    catalog_key = "fixture"

    def __init__(self, config, skip=frozenset()):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))


class _NoCatalogKeyModel(BaseModel):
    config_cls = _FakeConfig
    revision = "unused"
    # catalog_key intentionally unset -- must stay a no-op.

    def __init__(self, config, skip=frozenset()):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))


def test_from_pretrained_raises_checksum_mismatch_for_catalog_key_class(
    tmp_path, monkeypatch
):
    # A genuinely loadable checkpoint, so that -- without the fix -- loading
    # would SUCCEED (proving the raise comes from the digest check itself,
    # not from some unrelated torch.load failure).
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "config.json").write_text(json.dumps({}))
    torch.save({"dummy": torch.zeros(1)}, snapshot_dir / "checkpoint.pt")

    fake_registry = {
        "fixture": {
            "model_id": "example/fixture",
            "artifact": "checkpoint.pt",
            # Deliberately wrong -- doesn't match the real file's sha256.
            "sha256": "f" * 64,
            "size_bytes": (snapshot_dir / "checkpoint.pt").stat().st_size,
        }
    }
    monkeypatch.setattr(
        "sam_audio_kit.download.checkpoint_info", lambda model: fake_registry[model]
    )
    monkeypatch.setattr(
        "sam_audio_kit.sam_audio.model.base.snapshot_download",
        lambda **kwargs: str(snapshot_dir),
    )

    with pytest.raises(ChecksumMismatchError):
        _FixtureModel.from_pretrained("example/fixture")


def test_from_pretrained_succeeds_when_digest_matches_catalog(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "config.json").write_text(json.dumps({}))
    torch.save({"dummy": torch.zeros(1)}, snapshot_dir / "checkpoint.pt")
    actual_sha256 = hashlib.sha256(
        (snapshot_dir / "checkpoint.pt").read_bytes()
    ).hexdigest()

    fake_registry = {
        "fixture": {
            "model_id": "example/fixture",
            "artifact": "checkpoint.pt",
            "sha256": actual_sha256,
            "size_bytes": (snapshot_dir / "checkpoint.pt").stat().st_size,
        }
    }
    monkeypatch.setattr(
        "sam_audio_kit.download.checkpoint_info", lambda model: fake_registry[model]
    )
    monkeypatch.setattr(
        "sam_audio_kit.sam_audio.model.base.snapshot_download",
        lambda **kwargs: str(snapshot_dir),
    )

    model = _FixtureModel.from_pretrained("example/fixture")
    assert isinstance(model, _FixtureModel)


def test_from_pretrained_skips_verification_for_class_without_catalog_key(
    tmp_path, monkeypatch
):
    """A class that names no `catalog_key` must load without ever calling
    `_verify_artifact` -- proves the hook is opt-in, not blanket-applied."""
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "config.json").write_text(json.dumps({}))
    torch.save({"dummy": torch.zeros(1)}, snapshot_dir / "checkpoint.pt")

    verify_calls = []

    def spy_verify_artifact(*args, **kwargs):
        verify_calls.append((args, kwargs))
        raise AssertionError("_verify_artifact must not be called")

    monkeypatch.setattr(
        "sam_audio_kit.download._verify_artifact", spy_verify_artifact
    )
    monkeypatch.setattr(
        "sam_audio_kit.sam_audio.model.base.snapshot_download",
        lambda **kwargs: str(snapshot_dir),
    )

    model = _NoCatalogKeyModel.from_pretrained("example/no-catalog-key")
    assert isinstance(model, _NoCatalogKeyModel)
    assert verify_calls == []
