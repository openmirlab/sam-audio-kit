# sam-audio-kit

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Mixed](https://img.shields.io/badge/License-Mixed%20(see%20LICENSING.md)-orange.svg)](LICENSING.md)

**GitHub source/release only (PyPI held).** Public GitHub releases and GitHub
installation are allowed under the mixed-license terms documented in
[LICENSING.md](LICENSING.md). PyPI publication remains on hold as a separate
distribution-channel decision; this is policy guidance, not legal advice.

Inference-only package for [SAM-Audio](https://github.com/facebookresearch/sam-audio) (Segment Anything for Audio) by Meta AI.

---

## Why this exists

[SAM-Audio](https://github.com/facebookresearch/sam-audio) ("Segment Anything for
Audio") is Meta AI's foundation model for language/visual/temporal-prompted audio
source separation, introduced in *SAM Audio: Segment Anything in Audio*
(Shi et al., 2025 — see Citation below). The
[original repository](https://github.com/facebookresearch/sam-audio) is a full
research codebase: training, evaluation, and inference code bundled together,
with a heavier dependency footprint and no VRAM-conscious inference path. That
makes it awkward to drop into a lightweight production or creative-tooling
pipeline where you only ever want to *run* the model.

**sam-audio-kit** is a from-scratch, inference-only repackaging. It loads the
same upstream model weights (downloaded from Meta's gated HuggingFace
checkpoints — see [Prerequisites](#prerequisites) and
["What this project will NEVER bundle"](#what-this-project-will-never-bundle)
below) and runs the same architecture, but strips training code and adds a
VRAM-efficient "Lite Mode" (62–78% VRAM reduction), auto-chunking for long
audio, mixed-precision inference, model caching, and a small Python API + CLI.
For training or the full research codebase, use the
[original SAM-Audio repository](https://github.com/facebookresearch/sam-audio).

## Acknowledgments

sam-audio-kit is an independent, unofficial repackaging built on top of other
people's research and tooling. None of the following are affiliated with or
endorse this repackaging.

- **Upstream model & research**: [SAM-Audio](https://github.com/facebookresearch/sam-audio)
  ("Segment Anything for Audio") was developed by **Meta AI (FAIR)**. The
  paper's authors are Bowen Shi, Andros Tjandra, John Hoffman, Helin Wang,
  Yi-Chiao Wu, Luya Gao, Julius Richter, Matt Le, Apoorv Vyas, Sanyuan Chen,
  Christoph Feichtenhofer, Piotr Dollár, Wei-Ning Hsu, and Ann Lee — verified
  against the paper's own page at
  [arXiv:2512.18099](https://arxiv.org/abs/2512.18099) (title and full author
  list match; checked 2026-07-12). See [Citation](#citation) below.
- **Source repository**: [github.com/facebookresearch/sam-audio](https://github.com/facebookresearch/sam-audio)
- **Model weights host**: Hugging Face, gated under Meta's SAM License —
  [facebook/sam-audio-base](https://huggingface.co/facebook/sam-audio-base),
  [facebook/sam-audio-large](https://huggingface.co/facebook/sam-audio-large)
- **Perception Encoders** (vendored vision/audio-visual encoder components,
  Apache-2.0): also Meta AI Research — see [`LICENSE.PE`](LICENSE.PE)
- **`dacvae` neural audio codec** (Apache-2.0, not vendored, installed
  separately): [github.com/facebookresearch/dacvae](https://github.com/facebookresearch/dacvae),
  Meta AI Research
- **Lite Mode VRAM optimization technique**: inspired by the memory-reduction
  approach used in [AudioGhost AI](https://github.com/0x0funky/audioghost-ai)
  by [0x0funky](https://github.com/0x0funky) — a "memory-optimized SAM-Audio
  with modern UI" project whose README documents removing the same unused
  components (vision encoder, visual/text rankers) for VRAM savings. No code
  was copied, approach only. (Verified via the GitHub API that both the repo
  and the account exist, and independently confirmed the README's described
  technique, on 2026-07-12.)
- **Latent granular synthesis technique** (inspiration only, no vendored
  code): credited to Naotokui's public work — see [`LICENSING.md`](LICENSING.md)

## Citation

If you use SAM Audio in your research, please cite the original paper:

```bibtex
@article{shi2025samaudio,
    title={SAM Audio: Segment Anything in Audio},
    author={Bowen Shi and Andros Tjandra and John Hoffman and Helin Wang and Yi-Chiao Wu and Luya Gao and Julius Richter and Matt Le and Apoorv Vyas and Sanyuan Chen and Christoph Feichtenhofer and Piotr Doll{\'a}r and Wei-Ning Hsu and Ann Lee},
    year={2025},
    url={https://arxiv.org/abs/2512.18099}
}
```

*(Verified against [arXiv:2512.18099](https://arxiv.org/abs/2512.18099) on
2026-07-12: title and full author list match what is printed above.)*

---

## Features

- **Inference-Only**: Optimized for inference with `torch.inference_mode()` (no grad overhead)
- **Lite Mode**: Reduce VRAM usage by 62-78% by removing unused components
- **Mixed Precision**: Support for bfloat16/float16 inference
- **48kHz Audio**: Native high-quality audio processing at 48kHz sample rate
- **Auto-Chunking**: Process long audio files without OOM errors
- **Model Caching**: Configurable cache directory with environment variable support
- **Warmup Support**: Pre-compile CUDA kernels for faster first inference
- **Simple API**: Easy-to-use Python API and CLI
- **Creative Synthesis**: Iterative subtractive synthesis using semantic filtering

## Scope

**In scope:**
- Inference-only SAM-Audio: load a pretrained checkpoint, run separation/synthesis
- VRAM-efficient "Lite Mode", mixed precision, auto-chunking for long audio
- Python API + CLI for separation and iterative subtractive synthesis
- Model download/caching helpers for the gated HuggingFace checkpoints

**Out of scope, forever:**
- Training or fine-tuning SAM-Audio — use the
  [original repository](https://github.com/facebookresearch/sam-audio) for that
- Redistributing SAM-Audio model weights — they remain gated on HuggingFace
  under Meta's SAM License; this package only automates *your own*
  authenticated download, and never bundles or mirrors the weights itself
  (see ["What this project will NEVER bundle"](#what-this-project-will-never-bundle))
- Reimplementing or modifying the SAM-Audio model architecture itself —
  `src/sam_audio_kit/sam_audio/**` is a thin extraction of the upstream model,
  not a redesign

## Installation

This package is intentionally **not published to PyPI**. Install the public
source directly from GitHub (or clone it first); this keeps the distribution
channel explicit while the PyPI decision remains open.

```bash
# Direct GitHub install
pip install "sam-audio-kit @ git+https://github.com/openmirlab/sam-audio-kit.git"

# Or clone for development
git clone https://github.com/openmirlab/sam-audio-kit.git
cd sam-audio-kit

# Using uv (recommended)
uv sync

# Or using pip from the clone
pip install -e .

# Optional: Initialize Claude Code for AI-assisted development
claude init
```

### Prerequisites

This repository combines OpenMIRLab MIT wrapper code, Apache-2.0 Perception
Encoders, and Meta's non-OSI SAM License for the SAM-Audio model code. Keep
all three license files when redistributing source or GitHub-built artifacts;
the Meta terms include trade-control/ITAR, military/weapons, and
reverse-engineering restrictions. Review [LICENSING.md](LICENSING.md) before
use; this documentation is not legal advice.

**HuggingFace Access Required**: SAM-Audio models are gated.

1. Request access to the model checkpoints:
   - [facebook/sam-audio-base](https://huggingface.co/facebook/sam-audio-base)
   - [facebook/sam-audio-large](https://huggingface.co/facebook/sam-audio-large)
2. Once accepted, authenticate with HuggingFace:
   ```bash
   # Generate token at https://huggingface.co/settings/tokens
   huggingface-cli login
   # Or set environment variable
   export HF_TOKEN=hf_your_token_here
   ```

### Installing the audio codec (dacvae)

The core SAM-Audio model uses Meta's [`dacvae`](https://github.com/facebookresearch/dacvae)
neural audio codec (Apache-2.0) to encode/decode waveforms. It is **not** a
declared dependency of this package -- `dacvae` has no PyPI release, and
PyPI's upload validation rejects any package whose metadata contains a direct
git/URL dependency (even under an optional extra), so declaring it here would
make sam-audio-kit permanently unpublishable. Install it manually before
using `SamAudio`:

```bash
pip install "dacvae @ git+https://github.com/facebookresearch/dacvae"
```

If it's missing, `sam_audio_kit.sam_audio.model.codec` raises a clear
`ImportError` with this same instruction rather than failing with a bare
"module not found".

## Quick Start

### Managed lifecycle (recommended for services)

`SamAudioSession` owns one live model while the package-owned checkpoint
registry (`src/sam_audio_kit/config/checkpoints.toml`) records official model
URLs and provenance. The registry contains no weights and accepts explicit
metadata overrides for mirrors or deployment manifests.

Every registry entry pins a `source_revision` (a commit, never a floating
`"main"`) and a `sha256`/`size_bytes` for its primary artifact; `download_model()`
downloads at that pinned revision and verifies the downloaded artifact's
digest, raising `ChecksumMismatchError` on a mismatch. These digests were
read from the Hub's own git-LFS-recorded metadata
(`HfApi().model_info(..., files_metadata=True)`), **not** independently
re-hashed against downloaded bytes — see
["What this project will NEVER bundle"](#what-this-project-will-never-bundle)
for that limitation.

```python
from sam_audio_kit import SamAudioSession

with SamAudioSession(model="base", device="cuda") as session:
    result = session.infer("song.wav", "vocals")
    result.save("vocals.wav")
```

`load()` is idempotent, `infer()` is ready-only, `release()` drops live model
memory while allowing a later reload, and `close()` permanently and
idempotently ends the session. `cache_info()` resolves the exact Hugging Face
cache repository path without downloading or creating it. Devices support
legacy `auto` selection plus explicit `cpu`, `cuda`, `cuda:N`, and `mps`;
unavailable explicit accelerators raise. The legacy
`SamAudio.from_pretrained(...).separate(...)` API remains available for
compatibility.

### Python API

```python
from sam_audio_kit import SamAudio

# Load model (recommended settings, ~3 GB VRAM)
model = SamAudio.from_pretrained(
    "base",                      # Model size: "small", "base", or "large"
    dtype="bfloat16",            # Mixed precision (~50% VRAM savings)
    enable_text_ranker=False,    # +3 GB VRAM if enabled
    enable_span_predictor=False, # +3 GB VRAM if enabled
)

# Separate audio
result = model.separate("song.wav", description="vocals")
result.save("vocals.wav", "accompaniment.wav")
```

### Iterative Subtractive Synthesis (New!)

Use SAM-Audio as a creative semantic synthesizer:

```python
from sam_audio_kit import SamAudio
import torch

model = SamAudio.from_pretrained("base", dtype="bfloat16")

# Start with white noise
white_noise = torch.randn(1, 48000 * 10)  # 10 seconds

# Iteratively sculpt sound by removing components
descriptions = [
    "harsh high frequencies",
    "muddy low end",
    "noisy artifacts"
]

current = white_noise
for desc in descriptions:
    result = model.separate(current, desc)
    current = result.residual  # Keep what's left

# Save sculpted sound
torchaudio.save("sculpted.wav", current, 48000)
```

See [Iterative Subtractive Synthesis](docs/subtractive_synthesis.md) for detailed guide and examples.

### Command Line

```bash
# Basic separation
sam-audio-kit separate song.wav -d "vocals" -o vocals.wav

# With residual output
sam-audio-kit separate song.wav -d "drums" -o drums.wav --residual other.wav

# Download model with warmup
sam-audio-kit download --model base --warmup
```

## VRAM Usage

| Model | Full Mode | Lite Mode | Reduction |
|-------|-----------|-----------|-----------|
| Base | 12.73 GB | **2.84 GB** | **78%** |
| Large | 16.18 GB | **6.15 GB** | **62%** |

*Tested on RTX 4090 with bfloat16 precision*

## Documentation

- [CLI Reference](docs/cli.md) - Command line interface
- [Python API](docs/api.md) - Python API reference
- [Configuration](docs/configuration.md) - Models, precision, lite mode settings
- [Architecture](docs/architecture.md) - How it works and optimization techniques
- [Iterative Subtractive Synthesis](docs/subtractive_synthesis.md) - Creative semantic filtering
- [Benchmarks](docs/benchmarks.md) - VRAM and performance benchmarks
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

## Requirements

- Python >= 3.11
- PyTorch >= 2.0.0
- torchaudio >= 2.0.0
- CUDA-capable GPU with at least 4GB VRAM (lite + bfloat16)

## What this project will NEVER bundle

sam-audio-kit downloads SAM-Audio's pretrained checkpoints
(`facebook/sam-audio-base`, `facebook/sam-audio-large`, and optionally
`facebook/sam-audio-judge`) at runtime via `huggingface_hub.snapshot_download`
(see `src/sam_audio_kit/download.py`). This is the highest license-sensitivity
part of this repo, so read this section fully before you rely on it in a
product:

- **Gated on HuggingFace.** You must personally request and be granted access
  to each model page before `download_model()` / `sam-audio-kit download`
  will succeed. An unauthenticated or not-yet-approved request fails with a
  HuggingFace access-denied error — there is no silent fallback and no bundled
  copy to fall back to.
- **Licensed under Meta's SAM License** (`LICENSE.SAM-AUDIO`), a custom,
  source-available, **non-OSI** license with its own redistribution and
  use-restriction terms (Trade Controls/ITAR compliance, no
  military/weapons use, no reverse-engineering the model internals — see
  [`LICENSING.md`](LICENSING.md) for the full map). This is materially more
  restrictive than the MIT license covering sam-audio-kit's own wrapper code,
  and the two must not be conflated.
- **Never committed to this git repository, never included in the PyPI or
  GitHub release artifact (sdist/wheel), and never mirrored to any host this
  project controls.** Every checkpoint is fetched directly from Meta's own
  HuggingFace org into *your own* cache directory (`SAM_AUDIO_CACHE_DIR`,
  default `~/.cache/sam-audio-kit`), authenticated with *your own*
  HuggingFace token (`HF_TOKEN` / `HUGGINGFACE_TOKEN`) that must already have
  gated access approved by Meta.
- **This project has no mechanism to grant, bypass, or share gated access.**
  If your HuggingFace account doesn't have approved access to
  `facebook/sam-audio-base` / `-large` / `-judge`, no flag or environment
  variable in this package will get you the weights — access must come from
  Meta.
- **Digest verification is metadata-only, not byte-level, today.** The
  `sha256`/`size_bytes` pinned in `checkpoints.toml` come from one
  authoritative source — the Hub's own git-LFS-recorded object ID, read via
  `HfApi().model_info(repo, files_metadata=True)` — confirmed through two
  separate `huggingface_hub` code paths, but not yet cross-checked against an
  independently downloaded and re-hashed copy of the file: gated raw-content
  resolve (`get_hf_file_metadata`) currently 403s for the account used to
  read this metadata. `download_model()`'s post-download check still compares
  every real download against this same digest and will refuse a mismatch —
  it just means the *reference* value itself awaits a from-bytes
  confirmation, which should happen the first time someone with approved
  gated access downloads these weights for real.
- **The same applies to `dacvae`** (see
  ["Installing the audio codec"](#installing-the-audio-codec-dacvae) above):
  it is never vendored or bundled, precisely so this package's own
  distribution never becomes a redistribution vector for someone else's
  license terms.
- **Training, fine-tuning, evaluation, dataset, and experiment tooling are
  never shipped.** This repository is an inference-only runtime; use the
  upstream research repository for those workflows.

If you redistribute anything built with this package — fine-tuned weights, a
derivative checkpoint, model outputs bundled into another product, etc. —
**read `LICENSE.SAM-AUDIO` yourself first**. Its terms travel with the
weights and any derivative of them, independent of sam-audio-kit's own MIT
license, and this README does not summarize all of its terms.

## Development

```bash
git clone https://github.com/openmirlab/sam-audio-kit.git
cd sam-audio-kit
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Lint / format / type-check
uv run ruff check src/ tests/
uv run black src/ tests/ examples/
uv run mypy src/
```

See [`CLAUDE.md`](CLAUDE.md) for architecture notes, current release status,
and orientation for AI coding agents working in this repo.

## License

sam-audio-kit is **mixed-license**, not plain MIT:

- `src/sam_audio_kit/sam_audio/**` (the SAM-Audio model itself) is under Meta's
  custom **SAM License** -- source-available, but not OSI-approved, with its
  own redistribution and use-restriction terms. See [`LICENSE.SAM-AUDIO`](LICENSE.SAM-AUDIO).
- `src/sam_audio_kit/core/vision_encoder/**` and `core/audio_visual_encoder/**`
  (Perception Encoders) are **Apache-2.0**. See [`LICENSE.PE`](LICENSE.PE).
- The OpenMIRLab wrapper (inference orchestration, lite mode, chunking,
  synthesis extras, CLI, etc.) is **MIT**. See [`LICENSE`](LICENSE).

See [`LICENSING.md`](LICENSING.md) for the full component-by-component map.
**Read `LICENSE.SAM-AUDIO` yourself before redistributing this package or any
model output** -- it is more restrictive than MIT and this README does not
summarize all of its terms.

## Support

- **Issues**: [github.com/openmirlab/sam-audio-kit/issues](https://github.com/openmirlab/sam-audio-kit/issues)
- **Documentation**: see the [Documentation](#documentation) section above
  (CLI, Python API, configuration, architecture, benchmarks, troubleshooting)
- **Licensing questions**: [`LICENSING.md`](LICENSING.md) maps every
  component to its governing license; read `LICENSE.SAM-AUDIO` yourself
  before any redistribution decision
