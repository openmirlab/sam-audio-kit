# sam-audio-kit

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Mixed](https://img.shields.io/badge/License-Mixed%20(see%20LICENSING.md)-orange.svg)](LICENSING.md)

**Not yet published to PyPI** — release is gated on a pending decision about
redistributing code under Meta's SAM License (see [LICENSING.md](LICENSING.md)
and `CHANGELOG.md`). Install from a local clone in the meantime.

Inference-only package for [SAM-Audio](https://github.com/facebookresearch/sam-audio) (Segment Anything for Audio) by Meta AI.

This is a lightweight, dependency-minimal repackaging focused solely on inference with VRAM-efficient lite mode. For training and the full research codebase, please visit the [original SAM-Audio repository](https://github.com/facebookresearch/sam-audio).

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

## Installation

This package is **not published to PyPI** (see the note above). Install it
locally from a clone of this repo.

```bash
git clone https://github.com/openmirlab/sam-audio-kit.git
cd sam-audio-kit

# Using uv (recommended)
uv sync

# Or using pip
pip install -e .

# Optional: Initialize Claude Code for AI-assisted development
claude init
```

### Prerequisites

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

## Quick Start

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

## Acknowledgments

This package stands on the shoulders of two important projects.

### Original Research by Meta AI / Facebook Research

**SAM-Audio** (Segment Anything for Audio) is developed by Meta AI Research.

- **Repository**: [github.com/facebookresearch/sam-audio](https://github.com/facebookresearch/sam-audio)
- **Paper**: *Segment Anything for Audio*
- **HuggingFace**: [facebook/sam-audio-base](https://huggingface.co/facebook/sam-audio-base)

### Lite Mode Optimization

The **Lite Mode VRAM optimization technique** used in this package is inspired by [AudioGhost AI](https://github.com/0x0funky/audioghost-ai).

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

## Installing the audio codec (dacvae)

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
