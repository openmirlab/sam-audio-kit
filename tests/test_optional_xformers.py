"""Keep the default SDPA path independent of xformers without changing its output.

The fixture was captured before moving the xformers import, on Linux x86_64
with Torch 2.9.1+cu128, CPU tensors, and seed 7401. It contains the input,
weights and all output floats rather than a digest or a serialized audio file.
Reads: fixtures/sdpa_pre_xformers_split.npz, sam_audio_kit.core.transformer.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from sam_audio_kit.core.transformer import Attention

FIXTURE = Path(__file__).parent / "fixtures" / "sdpa_pre_xformers_split.npz"


def _attention_from_fixture(source: np.lib.npyio.NpzFile) -> Attention:
    model = Attention(dim=16, head_dim=8, n_heads=2, n_kv_heads=2, rope_theta=10000).eval()
    with torch.no_grad():
        for name in ("wq", "wk", "wv", "wo"):
            getattr(model, name).weight.copy_(torch.from_numpy(source[name]))
    return model


def test_public_import_does_not_need_xformers() -> None:
    script = "import sys; sys.modules['xformers'] = None; import sam_audio_kit; assert sam_audio_kit.SamAudioSession"
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_default_sdpa_matches_pre_split_output() -> None:
    if torch.__version__ != "2.9.1+cu128" or (platform.system(), platform.machine()) != (
        "Linux", "x86_64"
    ):
        pytest.skip("exact float fixture requires recording Torch 2.9.1+cu128 on Linux x86_64")
    with np.load(FIXTURE) as source, torch.no_grad():
        model = _attention_from_fixture(source)
        output = model(
            torch.from_numpy(source["x"]), torch.from_numpy(source["freq"]), attn_impl="sdpa"
        )
        np.testing.assert_array_equal(output.numpy(), source["y"])


def test_optional_fmha_has_actionable_install_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "xformers", None)
    monkeypatch.setitem(sys.modules, "xformers.ops", None)
    with np.load(FIXTURE) as source, torch.no_grad():
        model = _attention_from_fixture(source)
        with pytest.raises(ImportError, match="xformers==0.0.33.post2"):
            model(
                torch.from_numpy(source["x"]),
                torch.from_numpy(source["freq"]),
                attn_impl="fmha",
            )
