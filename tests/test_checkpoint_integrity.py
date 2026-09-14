"""Checkpoint-catalog integrity contract: every registry entry is pinned and
verifiable, and a downloaded artifact's digest is actually checked."""
import hashlib
import re

import pytest

from sam_audio_kit.checkpoints import (
    ChecksumMismatchError,
    load_checkpoint_config,
)
from sam_audio_kit.download import _verify_artifact, resolve_model_cache_path

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def test_every_catalog_entry_has_a_digest_or_declared_unavailable():
    config = load_checkpoint_config()
    for name, entry in config["models"].items():
        sha256 = entry.get("sha256")
        integrity = entry.get("integrity")
        assert sha256 is not None or integrity == "unavailable", (
            f"models.{name} has neither a sha256 nor integrity='unavailable'"
        )
        if sha256 is not None:
            assert SHA256_RE.match(sha256), f"models.{name}.sha256 is not lowercase 64-hex: {sha256!r}"


def test_every_catalog_entry_pins_a_commit_revision_not_a_floating_branch():
    config = load_checkpoint_config()
    for name, entry in config["models"].items():
        revision = entry.get("source_revision")
        assert revision is not None and revision != "main", (
            f"models.{name}.source_revision must be a pinned commit, not {revision!r}"
        )


def test_verify_artifact_raises_checksum_mismatch_error_offline(tmp_path, monkeypatch):
    fake_config = {
        "models": {
            "fixture": {
                "model_id": "example/fixture",
                "artifact": "checkpoint.pt",
                "sha256": "0" * 64,
                "size_bytes": 4,
            }
        }
    }
    monkeypatch.setattr(
        "sam_audio_kit.download.checkpoint_info",
        lambda model: fake_config["models"][model],
    )
    artifact_path = tmp_path / "checkpoint.pt"
    artifact_path.write_bytes(b"not the right bytes")

    with pytest.raises(ChecksumMismatchError) as exc_info:
        _verify_artifact("fixture", tmp_path, verbose=False)
    assert exc_info.value.model == "fixture"
    assert exc_info.value.artifact == "checkpoint.pt"
    assert exc_info.value.expected == "0" * 64


def test_verify_artifact_passes_on_matching_digest_offline(tmp_path, monkeypatch):
    payload = b"deterministic fixture bytes"
    expected = hashlib.sha256(payload).hexdigest()
    fake_config = {
        "models": {
            "fixture": {
                "model_id": "example/fixture",
                "artifact": "checkpoint.pt",
                "sha256": expected,
                "size_bytes": len(payload),
            }
        }
    }
    monkeypatch.setattr(
        "sam_audio_kit.download.checkpoint_info",
        lambda model: fake_config["models"][model],
    )
    artifact_path = tmp_path / "checkpoint.pt"
    artifact_path.write_bytes(payload)

    _verify_artifact("fixture", tmp_path, verbose=False)  # must not raise


def test_verify_artifact_skips_when_integrity_marked_unavailable(tmp_path, monkeypatch, capsys):
    fake_config = {
        "models": {
            "fixture": {
                "model_id": "example/fixture",
                "artifact": "checkpoint.pt",
                "integrity": "unavailable",
            }
        }
    }
    monkeypatch.setattr(
        "sam_audio_kit.download.checkpoint_info",
        lambda model: fake_config["models"][model],
    )
    # No artifact file at all -- must still not raise, since verification is skipped.
    _verify_artifact("fixture", tmp_path, verbose=True)
    assert "unavailable" in capsys.readouterr().out


def test_download_model_wires_pinned_revision_and_verifies_checksum_offline(tmp_path, monkeypatch):
    """End-to-end (offline): download_model must pass the catalog's source_revision
    to snapshot_download and verify the resulting artifact -- no network, no real HF call."""
    import sam_audio_kit.download as download_module

    payload = b"fake small checkpoint payload"
    expected_sha256 = hashlib.sha256(payload).hexdigest()

    fake_registry = {
        "small": {
            "model_id": "facebook/sam-audio-small",
            "source_revision": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            "artifact": "checkpoint.pt",
            "sha256": expected_sha256,
            "size_bytes": len(payload),
        }
    }
    monkeypatch.setattr(download_module, "checkpoint_info", lambda model: fake_registry[model])

    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "checkpoint.pt").write_bytes(payload)

    calls = []

    def fake_snapshot_download(*, repo_id, cache_dir, token, revision):
        calls.append({"repo_id": repo_id, "revision": revision})
        return str(snapshot_dir)

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", fake_snapshot_download
    )

    result = download_module.download_model("small", cache_dir=tmp_path, verbose=False)
    assert result == tmp_path
    assert calls == [
        {"repo_id": "facebook/sam-audio-small", "revision": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"}
    ]


def test_download_model_raises_on_mismatch_offline(tmp_path, monkeypatch):
    import sam_audio_kit.download as download_module

    fake_registry = {
        "small": {
            "model_id": "facebook/sam-audio-small",
            "source_revision": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            "artifact": "checkpoint.pt",
            "sha256": "f" * 64,
            "size_bytes": 4,
        }
    }
    monkeypatch.setattr(download_module, "checkpoint_info", lambda model: fake_registry[model])

    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "checkpoint.pt").write_bytes(b"wrong bytes here")

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: str(snapshot_dir),
    )

    with pytest.raises(ChecksumMismatchError):
        download_module.download_model("small", cache_dir=tmp_path, verbose=False)


def test_resolve_model_cache_path_with_and_without_revision(tmp_path):
    bare = resolve_model_cache_path("base", tmp_path)
    assert bare == tmp_path / "models--facebook--sam-audio-base"

    pinned = resolve_model_cache_path("base", tmp_path, revision="abc123")
    assert pinned == tmp_path / "models--facebook--sam-audio-base" / "snapshots" / "abc123"

    # default (no revision passed at all) matches today's exact bare-root behaviour
    assert resolve_model_cache_path("base", tmp_path) == bare


@pytest.mark.network
def test_pinned_digests_still_match_the_hub_live(monkeypatch):
    """Cheap (metadata-only, no multi-GB download) drift check against the live Hub."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    config = load_checkpoint_config()
    api = huggingface_hub.HfApi()
    try:
        api.whoami()
    except Exception:
        pytest.skip("no HF network/token available")

    for name, entry in config["models"].items():
        model_id = entry["model_id"]
        revision = entry["source_revision"]
        artifact = entry.get("artifact")
        expected_sha256 = entry.get("sha256")
        if artifact is None or expected_sha256 is None:
            continue
        try:
            info = api.model_info(model_id, revision=revision, files_metadata=True)
        except Exception as exc:
            pytest.skip(f"could not reach Hub for {model_id}: {exc}")
        siblings = {s.rfilename: s for s in info.siblings}
        sibling = siblings.get(artifact)
        assert sibling is not None, f"{model_id}@{revision} no longer has {artifact}"
        actual_sha256 = getattr(sibling, "lfs", None)
        actual_sha256 = actual_sha256["sha256"] if isinstance(actual_sha256, dict) else None
        if actual_sha256 is None:
            pytest.skip(f"{model_id}@{revision}/{artifact} has no LFS sha256 in this record")
        assert actual_sha256 == expected_sha256, (
            f"{model_id}@{revision}/{artifact}: catalog sha256 {expected_sha256} != "
            f"Hub sha256 {actual_sha256} -- weights may have rotated"
        )
