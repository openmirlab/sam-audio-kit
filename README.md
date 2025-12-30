# sam-audio-infer

[![PyPI](https://img.shields.io/pypi/v/sam-audio-infer)](https://pypi.org/project/sam-audio-infer/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**PyPI**: https://pypi.org/project/sam-audio-infer/

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

## Installation

```bash
# Using uv (recommended)
uv add sam-audio-infer
uv pip install git+https://github.com/facebookresearch/sam-audio.git

# Or using pip
pip install sam-audio-infer
pip install git+https://github.com/facebookresearch/sam-audio.git
```

For development:
```bash
git clone https://github.com/openmirlab/sam-audio-infer.git
cd sam-audio-infer
uv sync
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
from sam_audio_infer import SamAudioInfer

# Load model (recommended settings, ~3 GB VRAM)
model = SamAudioInfer.from_pretrained(
    "base",                      # Model size: "small", "base", or "large"
    dtype="bfloat16",            # Mixed precision (~50% VRAM savings)
    enable_text_ranker=False,    # +3 GB VRAM if enabled
    enable_span_predictor=False, # +3 GB VRAM if enabled
)

# Separate audio
result = model.separate("song.wav", description="vocals")
result.save("vocals.wav", "accompaniment.wav")
```

### Command Line

```bash
# Basic separation
sam-audio-infer separate song.wav -d "vocals" -o vocals.wav

# With residual output
sam-audio-infer separate song.wav -d "drums" -o drums.wav --residual other.wav

# Download model with warmup
sam-audio-infer download --model base --warmup
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

MIT License

**Note**: The underlying SAM-Audio model has its own license terms. Please refer to the [official SAM-Audio repository](https://github.com/facebookresearch/sam-audio) for model usage terms.

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
