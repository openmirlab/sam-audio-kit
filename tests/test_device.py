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
        elif torch.backends.mps.is_available():
            assert resolved == "mps"
        else:
            assert resolved == "cpu"

    def test_explicit_devices_pass_through_unchanged(self):
        assert resolve_device("cpu") == "cpu"
        assert resolve_device("cuda") == "cuda"
        assert resolve_device("mps") == "mps"


class TestSamAudioInitResolvesAutoDevice:
    """SamAudio.__init__ is the funnel every code path converges on before
    the model is moved to a device; it must never leave self._device == "auto"."""

    def test_init_resolves_auto_device(self):
        instance = SamAudio(model=_FakeModel(), processor=object(), device="auto")
        assert instance._device == resolve_device("auto")
        assert instance._device != "auto"
