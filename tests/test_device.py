"""Tests for the "auto" device sentinel and its resolution."""

import pytest

# sam_audio_kit/__init__.py unconditionally imports the SAM-Audio model
# chain, which requires the `dacvae` codec (no PyPI release; not a declared
# dependency -- see pyproject.toml and README.md "Installing the audio codec
# (dacvae)"). Skip this module cleanly in environments without it (e.g. CI)
# instead of hard-failing collection.
pytest.importorskip("dacvae")

import torch

from sam_audio_kit.model import SamAudio
from sam_audio_kit.types import resolve_device


class _FakeModel:
    """Minimal stand-in for a SAM-Audio model: only .to()/.eval() are used by __init__."""

    def to(self, device, dtype):
        return self

    def eval(self):
        return self


class TestResolveDevice:
    """Tests for resolve_device."""

    def test_auto_resolves_to_a_concrete_device(self):
        resolved = resolve_device("auto")
        assert resolved != "auto"
        if torch.cuda.is_available():
            assert resolved == "cuda"
        else:
            assert resolved == "cpu"

    def test_none_preserves_automatic_cpu_fallback(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert resolve_device(None) == "cpu"

    def test_auto_never_selects_mps_even_when_available(self, monkeypatch):
        """Apple MLX/Torch MPS backends are out of scope (org canon,
        2026-09-14): "auto" must resolve to cuda-else-cpu even on a
        machine where torch reports MPS as available."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert resolve_device("auto") == "cpu"
        assert resolve_device(None) == "cpu"

    def test_explicit_cpu_passes_through_unchanged(self):
        assert resolve_device("cpu") == "cpu"

    def test_explicit_cuda_index_is_preserved(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        assert resolve_device("cuda") == "cuda"
        assert resolve_device("cuda:1") == "cuda:1"

    def test_unavailable_explicit_cuda_raises(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda")

    def test_explicit_mps_always_raises_value_error(self, monkeypatch):
        """mps must be rejected outright, even when torch reports it as
        available -- there is no code path that should ever return it."""
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        with pytest.raises(ValueError, match="mps"):
            resolve_device("mps")
        with pytest.raises(ValueError, match="mps"):
            resolve_device("mps:0")

    def test_invalid_device_string_raises_value_error(self):
        with pytest.raises(ValueError):
            resolve_device("metal")


class TestSamAudioInitResolvesAutoDevice:
    """SamAudio.__init__ is the funnel every code path converges on before
    the model is moved to a device; it must never leave self._device == "auto"."""

    def test_init_resolves_auto_device(self):
        instance = SamAudio(model=_FakeModel(), processor=object(), device="auto")
        assert instance._device == resolve_device("auto")
        assert instance._device != "auto"
