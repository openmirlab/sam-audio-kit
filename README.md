# SAM-Audio-Infer

Optimized inference package for Meta's SAM-Audio model with VRAM-efficient lite mode.

## Features

- **Lite Mode**: Reduce VRAM usage by ~40% by removing unused components
- **Mixed Precision**: Support for bfloat16/float16 inference (~50% additional savings)
- **48kHz Audio**: Native high-quality audio processing at 48kHz sample rate
- **Auto-Chunking**: Process long audio files without OOM errors
- **Model Caching**: Configurable cache directory with environment variable support
- **Warmup Support**: Pre-compile CUDA kernels for faster inference
- **Simple API**: Easy-to-use Python API and CLI

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Download & Caching](#model-download--caching)
- [CLI Reference](#cli-reference)
- [Python API](#python-api)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Benchmarks](#benchmarks)
- [Requirements](#requirements)
- [Acknowledgments](#acknowledgments)

---

## Installation

### Prerequisites

**HuggingFace Access Required**: SAM-Audio models are gated.

1. Create a HuggingFace account at [huggingface.co](https://huggingface.co)
2. Request access to [facebook/sam-audio-base](https://huggingface.co/facebook/sam-audio-base)
3. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Using uv (Recommended)

```bash
# Clone and install
git clone https://github.com/your-username/sam-audio-infer.git
cd sam-audio-infer
uv sync

# Or add to existing project
uv add sam-audio-infer --path /path/to/sam-audio-infer
```

### Using pip

```bash
git clone https://github.com/your-username/sam-audio-infer.git
cd sam-audio-infer
pip install -e .
```

---

## Quick Start

### Environment Setup

Create a `.env` file with your HuggingFace token:

```bash
# Required for model download
HF_TOKEN=hf_your_token_here

# Optional: Custom cache directory (default: ~/.cache/sam-audio-infer)
SAM_AUDIO_CACHE_DIR=/path/to/cache
```

### Python API

```python
from sam_audio_infer import SamAudioInfer

# Load model with lite mode (recommended, ~5 GB VRAM)
model = SamAudioInfer.from_pretrained(
    "base",              # Model size: "small", "base", or "large"
    lite_mode=True,      # Remove unused components (~40% VRAM savings)
    dtype="bfloat16",    # Mixed precision (~50% additional savings)
)

# Separate audio
result = model.separate("song.wav", description="vocals")
result.save("vocals.wav", "accompaniment.wav")

print(f"Processing time: {result.processing_time:.1f}s")
print(f"Sample rate: {result.sample_rate} Hz")  # 48000 Hz
```

### Command Line

```bash
# Basic separation
sam-audio-infer separate song.wav -d "vocals" -o vocals.wav

# With residual output
sam-audio-infer separate song.wav -d "drums" -o drums.wav --residual other.wav

# Verbose output
sam-audio-infer separate song.wav -d "piano" -o piano.wav -v
```

---

## Model Download & Caching

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SAM_AUDIO_CACHE_DIR` | Model cache directory | `~/.cache/sam-audio-infer` |
| `HF_TOKEN` | HuggingFace API token | None |
| `HUGGINGFACE_TOKEN` | Alternative HF token | None |

### Download Models

#### Using CLI

```bash
# Download base model
sam-audio-infer download --model base

# Download to custom directory
sam-audio-infer download --model base --cache-dir /models/sam-audio

# Download and warmup (recommended for production)
sam-audio-infer download --model base --warmup

# Download with warmup settings
sam-audio-infer download --model base --warmup --warmup-duration 2.0

# Also download the judge model (for quality assessment)
sam-audio-infer download --model base --include-judge
```

#### Using Python

```python
from sam_audio_infer import download_model, download_and_warmup, warmup_model

# Download only
download_model("base")

# Download to custom directory
download_model("base", cache_dir="/models/sam-audio")

# Download, load, and warmup (recommended for production)
model = download_and_warmup(
    model_size="base",
    lite_mode=True,
    warmup_duration=1.0,  # seconds of dummy audio to process
)

# Warmup existing model
from sam_audio_infer import SamAudioInfer
model = SamAudioInfer.from_pretrained("base")
warmup_model(model, duration_seconds=1.0)
```

### Manage Cache

#### List Cached Models

```bash
sam-audio-infer list
```

```python
from sam_audio_infer import list_cached_models

for model in list_cached_models():
    print(f"{model['model_id']}: {model['size_gb']:.2f} GB")
```

#### Clear Cache

```bash
# Clear specific model
sam-audio-infer clear --model base

# Clear all models
sam-audio-infer clear -y

# Clear from custom directory
sam-audio-infer clear --cache-dir /models/sam-audio -y
```

```python
from sam_audio_infer import clear_cache

clear_cache(model_size="base")  # Clear specific model
clear_cache()  # Clear all models
```

---

## CLI Reference

### Commands Overview

```bash
sam-audio-infer --help
```

| Command | Description |
|---------|-------------|
| `separate` | Separate audio based on text description |
| `download` | Download and cache model files |
| `list` | List cached models |
| `clear` | Clear cached models |

### separate

Separate audio based on text description.

```bash
sam-audio-infer separate <input> -d <description> -o <output> [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `<input>` | Input audio file | Required |
| `-d, --description` | What to extract (e.g., "vocals") | Required |
| `-o, --output` | Output file path | Required |
| `--residual` | Output path for residual audio | None |
| `--model` | Model size: small, base, large | base |
| `--lite / --no-lite` | Enable/disable lite mode | enabled |
| `--dtype` | float32, float16, bfloat16 | bfloat16 |
| `--device` | cuda, cpu, mps | cuda |
| `--chunk-duration` | Chunk size in seconds | 25.0 |
| `--cache-dir` | Model cache directory | env/default |
| `--hf-token` | HuggingFace token | env |
| `--warmup` | Run warmup before processing | False |
| `-v, --verbose` | Verbose output | False |

**Examples:**

```bash
# Basic usage
sam-audio-infer separate song.wav -d "vocals" -o vocals.wav

# Extract drums with residual
sam-audio-infer separate song.wav -d "drums" -o drums.wav --residual other.wav

# Use large model with warmup
sam-audio-infer separate song.wav -d "bass" -o bass.wav --model large --warmup -v

# Custom cache directory
sam-audio-infer separate song.wav -d "piano" -o piano.wav --cache-dir /models
```

### download

Download and cache model files.

```bash
sam-audio-infer download [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model size: small, base, large | base |
| `--cache-dir` | Cache directory | env/default |
| `--hf-token` | HuggingFace token | env |
| `--warmup` | Also run warmup inference | False |
| `--warmup-duration` | Warmup audio duration (seconds) | 1.0 |
| `--no-lite` | Disable lite mode for warmup | False |
| `--device` | Device for warmup | cuda |
| `--dtype` | Data type for warmup | bfloat16 |
| `--include-judge` | Also download judge model | False |

**Examples:**

```bash
# Download base model
sam-audio-infer download --model base

# Download and warmup for production deployment
sam-audio-infer download --model base --warmup

# Download all sizes
sam-audio-infer download --model small
sam-audio-infer download --model base
sam-audio-infer download --model large
```

### list

List cached models.

```bash
sam-audio-infer list [--cache-dir DIR]
```

### clear

Clear cached models.

```bash
sam-audio-infer clear [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Specific model to clear | all |
| `--cache-dir` | Cache directory | env/default |
| `-y, --yes` | Skip confirmation | False |

---

## Python API

### SamAudioInfer

Main class for audio separation.

```python
from sam_audio_infer import SamAudioInfer

# Load model
model = SamAudioInfer.from_pretrained(
    model_name_or_path="base",  # "small", "base", "large", or HuggingFace ID
    lite_mode=True,             # Remove unused components
    lite_config=None,           # Custom LiteModelConfig (overrides lite params)
    enable_text_ranker=False,   # Keep text ranker (+~2GB VRAM)
    enable_span_predictor=False,# Keep span predictor (+~1-2GB VRAM)
    reranking_candidates=3,     # Candidates for text ranker
    device="cuda",              # "cuda", "cpu", "mps"
    dtype="bfloat16",           # "float32", "float16", "bfloat16"
    chunk_duration=25.0,        # Default chunk size (seconds)
    hf_token=None,              # HuggingFace token (or use env)
    cache_dir=None,             # Cache directory (or use env)
    verbose=True,               # Print loading progress
)

# Properties
model.sample_rate  # 48000
model.device       # "cuda"
model.dtype        # "bfloat16"
model.is_lite      # True

# Separate single audio
result = model.separate(
    audio="song.wav",           # Path, numpy array, or torch tensor
    description="vocals",       # What to extract
    chunk_duration=None,        # Override default chunk size
    cleanup_between_chunks=True,
    verbose=False,
)

# Separate multiple stems
results = model.separate_batch(
    audio="song.wav",
    descriptions=["vocals", "drums", "bass", "other"],
    verbose=False,
)

# Move to different device
model.to("cpu")

# Unload model
model.unload()
```

### SeparationResult

Result from audio separation.

```python
result = model.separate("song.wav", "vocals")

# Properties
result.target           # torch.Tensor - extracted audio
result.residual         # torch.Tensor - remaining audio
result.sample_rate      # 48000
result.description      # "vocals"
result.processing_time  # seconds
result.num_chunks       # number of chunks processed

# Save to files
result.save("vocals.wav", "accompaniment.wav")
result.save("vocals.wav")  # Only save target
```

### Download Functions

```python
from sam_audio_infer import (
    download_model,
    download_and_warmup,
    warmup_model,
    get_cache_dir,
    list_cached_models,
    clear_cache,
)

# Get cache directory (from env or default)
cache_dir = get_cache_dir()  # Path object

# Download model files
download_model(
    model_size="base",      # "small", "base", "large"
    cache_dir=None,         # Use env/default
    hf_token=None,          # Use env
    include_judge=False,    # Also download judge model
    verbose=True,
)

# Download, load, and warmup
model = download_and_warmup(
    model_size="base",
    lite_mode=True,
    device="cuda",
    dtype="bfloat16",
    cache_dir=None,
    hf_token=None,
    warmup_duration=1.0,
    verbose=True,
)

# Warmup existing model
warmup_time = warmup_model(
    model,                  # SamAudioInfer instance
    duration_seconds=1.0,
    verbose=True,
)

# List cached models
models = list_cached_models(cache_dir=None)
# Returns: [{"model_id": "facebook/sam-audio-base", "size_gb": 7.19, ...}]

# Clear cache
clear_cache(cache_dir=None, model_size=None, verbose=True)
```

### LiteModelConfig

Configure which components to remove/keep.

```python
from sam_audio_infer import LiteModelConfig

# Pre-built configurations
config = LiteModelConfig.aggressive()        # ~4-5 GB - maximum savings
config = LiteModelConfig.with_text_ranker(reranking_candidates=5)  # ~6-7 GB
config = LiteModelConfig.with_span_predictor()  # ~6-7 GB
config = LiteModelConfig.with_all_features(reranking_candidates=3)  # ~8-9 GB

# Custom configuration
config = LiteModelConfig(
    remove_vision_encoder=True,   # Always True for audio-only
    remove_visual_ranker=True,    # Always True for audio-only
    remove_text_ranker=False,     # Keep for quality
    remove_span_predictor=True,   # Remove to save VRAM
    reranking_candidates=5,       # Number of candidates
    predict_spans=False,
)

# Use with model
model = SamAudioInfer.from_pretrained("base", lite_config=config)
```

### Memory Management

```python
from sam_audio_infer import (
    cleanup_gpu_memory,
    get_gpu_memory_info,
    GPUMemoryInfo,
    MemoryTracker,
)

# Get GPU memory info
info = get_gpu_memory_info()
print(f"Total: {info.total_gb:.1f} GB")
print(f"Allocated: {info.allocated_gb:.1f} GB")
print(f"Free: {info.free_gb:.1f} GB")

# Track memory during operations
with MemoryTracker("Separation"):
    result = model.separate("song.wav", "vocals")

# Manual cleanup
cleanup_gpu_memory()
```

---

## Configuration

### Available Models

| Model | HuggingFace ID | VRAM (Lite) | Use Case |
|-------|----------------|-------------|----------|
| `small` | `facebook/sam-audio-small` | **~4 GB** | Fast inference, limited VRAM |
| `base` | `facebook/sam-audio-base` | **~5 GB** | **Recommended** for most use cases |
| `large` | `facebook/sam-audio-large` | **~7 GB** | Best quality |

### VRAM by Configuration (Base Model + bfloat16)

| Configuration | VRAM | Features |
|--------------|------|----------|
| `LiteModelConfig.aggressive()` | ~4-5 GB | Basic separation |
| `LiteModelConfig.with_text_ranker()` | ~6-7 GB | + Quality reranking |
| `LiteModelConfig.with_span_predictor()` | ~6-7 GB | + Time segments |
| `LiteModelConfig.with_all_features()` | ~8-9 GB | + Both features |

### Quality Scores (Subjective 1-5)

| Category | Small | Base | Large |
|----------|-------|------|-------|
| General SFX | 3.62 | 3.28 | 3.50 |
| Speech | 3.99 | **4.25** | 4.03 |
| Music | 4.11 | 3.87 | **4.22** |
| Instruments (pro) | 4.24 | 4.27 | **4.49** |

### Precision Settings

Fine-grained control over numerical precision for speed vs quality trade-offs.

#### Options

| Setting | Values | Description |
|---------|--------|-------------|
| `matmul_precision` | `highest`, `high`, `medium` | Internal precision for matrix multiplications |
| `tf32` | `true`, `false` | Enable TensorFloat-32 (~3x speedup on Ampere+ GPUs) |
| `cudnn_benchmark` | `true`, `false` | Auto-tune convolution algorithms |
| `deterministic` | `true`, `false` | Force reproducible results (slower) |

#### What is `matmul_precision`?

Controls the internal precision PyTorch uses for matrix multiplications:

| Value | Internal Precision | Speed | Quality |
|-------|-------------------|-------|---------|
| `highest` | Full float32 (32-bit) | Slowest | Best |
| `high` | TF32 (19-bit mantissa) | ~3x faster | Very good |
| `medium` | Lower precision | Fastest | Good |

**Note:** There is no `low` option. `medium` is the fastest available.

For `dtype=bfloat16`, the impact is smaller since tensors are already in reduced precision.

#### Environment Variables (Defaults)

Set these to configure defaults across all runs:

```bash
export SAM_AUDIO_TF32=true                    # default: true
export SAM_AUDIO_MATMUL_PRECISION=high        # default: high
export SAM_AUDIO_CUDNN_BENCHMARK=true         # default: true
export SAM_AUDIO_DETERMINISTIC=false          # default: false
```

#### CLI Flags (Override Per-Run)

```bash
# Fast inference (max speed)
sam-audio-infer separate audio.wav -d "vocals" -o out.wav \
    --matmul-precision medium

# Maximum quality
sam-audio-infer separate audio.wav -d "vocals" -o out.wav \
    --no-tf32 --matmul-precision highest

# Reproducible results
sam-audio-infer separate audio.wav -d "vocals" -o out.wav \
    --deterministic --no-cudnn-benchmark
```

| Flag | Description |
|------|-------------|
| `--tf32` | Enable TF32 (default) |
| `--no-tf32` | Disable TF32 for maximum precision |
| `--matmul-precision` | `highest`, `high`, `medium` |
| `--cudnn-benchmark` | Enable cuDNN auto-tuner (default) |
| `--no-cudnn-benchmark` | Disable cuDNN auto-tuner |
| `--deterministic` | Force reproducible results |

#### Python API

```python
from sam_audio_infer import SamAudioInfer, PrecisionConfig

# Fast inference
config = PrecisionConfig(
    matmul_precision="medium",
    allow_tf32=True,
    cudnn_benchmark=True,
)
model = SamAudioInfer.from_pretrained("base", precision_config=config)

# Maximum quality
config = PrecisionConfig(
    matmul_precision="highest",
    allow_tf32=False,
    cudnn_deterministic=True,
)
model = SamAudioInfer.from_pretrained("base", precision_config=config)

# Load from environment variables
config = PrecisionConfig.from_env()
```

#### Recommendations

| Use Case | Settings |
|----------|----------|
| **Production** (fastest) | `matmul=medium`, `tf32=true`, `cudnn_benchmark=true` |
| **Balanced** (default) | `matmul=high`, `tf32=true`, `cudnn_benchmark=true` |
| **Quality-critical** | `matmul=highest`, `tf32=false` |
| **Reproducible** | `matmul=highest`, `tf32=false`, `deterministic=true` |

---

## How It Works

SAM-Audio is a multimodal model designed for both audio AND video inputs. For audio-only separation, several components are **never used** - this is the key insight behind our optimizations.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SAM-Audio Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐   │
│  │  Audio Encoder   │  │  Vision Encoder  │  │ Text Encoder │   │
│  │                  │  │                  │  │              │   │
│  │  (REQUIRED)      │  │  (~2GB) ❌       │  │ (REQUIRED)   │   │
│  └────────┬─────────┘  └────────┬─────────┘  └──────┬───────┘   │
│           │                     │                    │           │
│           └──────────┬──────────┴────────────────────┘           │
│                      ▼                                           │
│           ┌──────────────────────┐                               │
│           │   Fusion / Decoder   │                               │
│           │      (REQUIRED)      │                               │
│           └──────────┬───────────┘                               │
│                      │                                           │
│    ┌─────────────────┼─────────────────┐                         │
│    ▼                 ▼                 ▼                         │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐                │
│  │ Visual   │  │    Text      │  │    Span      │                │
│  │ Ranker   │  │   Ranker     │  │  Predictor   │                │
│  │ (~2GB) ❌ │  │  (~2GB) ⚠️   │  │ (~1-2GB) ⚠️  │                │
│  └──────────┘  └──────────────┘  └──────────────┘                │
│                                                                  │
│  ❌ = Always removed (not needed for audio-only)                 │
│  ⚠️ = Optionally kept (improves quality)                         │
└──────────────────────────────────────────────────────────────────┘
```

### Optimization Techniques

#### 1. Lite Mode (~40% VRAM savings)

Remove unused components for audio-only tasks:

| Component | VRAM Saved | Removed by Default |
|-----------|------------|-------------------|
| `vision_encoder` | ~2 GB | ✅ Always |
| `visual_ranker` | ~2 GB | ✅ Always |
| `text_ranker` | ~2 GB | ✅ (optional keep) |
| `span_predictor` | ~1-2 GB | ✅ (optional keep) |

#### 2. Mixed Precision (~50% additional savings)

| Precision | VRAM Usage | Quality |
|-----------|------------|---------|
| float32 | 100% | Best |
| float16 | ~50% | Good (can be unstable) |
| **bfloat16** | **~50%** | **Excellent** (recommended) |

#### 3. Audio Chunking

Process long audio in segments to avoid OOM:

```python
# Automatic chunking (default: 25 seconds)
result = model.separate("long_song.wav", description="vocals")

# Custom chunk duration
result = model.separate("long_song.wav", "vocals", chunk_duration=30.0)
```

#### 4. Warmup (First Prediction Caching)

The first inference is slower due to CUDA kernel compilation. Warmup pre-compiles these:

```python
from sam_audio_infer import warmup_model

# After loading model
warmup_model(model, duration_seconds=1.0)
# Subsequent inferences will be faster
```

---

## Benchmarks

### VRAM Usage

| Model | Original | Lite + bfloat16 | Reduction |
|-------|----------|-----------------|-----------|
| Small | ~10 GB | **~4 GB** | **60%** |
| Base | ~13 GB | **~5 GB** | **62%** |
| Large | ~20 GB | **~7 GB** | **65%** |

### GPU Compatibility

| GPU | VRAM | Small | Base | Large |
|-----|------|-------|------|-------|
| RTX 3060 | 6 GB | ✅ | ⚠️ | ❌ |
| RTX 3070 | 8 GB | ✅ | ✅ | ❌ |
| RTX 3080 | 10 GB | ✅ | ✅ | ⚠️ |
| RTX 4070 | 12 GB | ✅ | ✅ | ✅ |
| RTX 4090 | 24 GB | ✅ | ✅ | ✅ |

✅ Comfortable | ⚠️ Tight (use smaller chunks) | ❌ Not recommended

---

## Requirements

- Python >= 3.11
- PyTorch >= 2.0.0
- torchaudio >= 2.0.0
- CUDA-capable GPU with at least 4GB VRAM (lite + bfloat16)
- Recommended: 8GB+ VRAM

---

## Troubleshooting

### Common Issues

**Model download fails:**
```bash
# Check your HuggingFace token
echo $HF_TOKEN

# Or set it explicitly
export HF_TOKEN=hf_your_token_here
```

**Out of memory:**
```python
# Use smaller chunks
result = model.separate("song.wav", "vocals", chunk_duration=15.0)

# Or use smaller model
model = SamAudioInfer.from_pretrained("small", lite_mode=True)
```

**Slow first inference:**
```python
# Use warmup
from sam_audio_infer import warmup_model
warmup_model(model, duration_seconds=1.0)
```

---

## Acknowledgments

### SAM-Audio (Meta AI / Facebook Research)

This package is built upon **SAM-Audio** (Segment Anything for Audio), developed by Meta AI.

- **Repository**: [github.com/facebookresearch/sam-audio](https://github.com/facebookresearch/sam-audio)
- **Paper**: *Segment Anything for Audio*

### AudioGhost AI

The **Lite Mode optimization technique** - the core innovation that reduces VRAM by ~40% - was pioneered in the **AudioGhost AI** project.

- **Key Innovation**: Discovery that SAM-Audio's vision encoder, rankers, and span predictor can be safely removed for audio-only separation tasks

---

## License

MIT License

**Note**: The underlying SAM-Audio model has its own license terms. Please refer to the [official SAM-Audio repository](https://github.com/facebookresearch/sam-audio) for model usage terms.

---

## Citation

```bibtex
@software{sam_audio_infer,
  title = {SAM-Audio-Infer: Memory-Efficient SAM-Audio Inference},
  year = {2024},
  note = {Lite mode optimization based on AudioGhost AI}
}
```
