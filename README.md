# SAM-Audio-Infer

Optimized inference package for Meta's SAM-Audio model with VRAM-efficient lite mode.

## Features

- **Multiple Model Sizes**: Support for `small`, `base`, and `large` models
- **Lite Mode**: Reduce VRAM usage by ~40% by removing unused components
- **Mixed Precision**: Support for bfloat16/float16 inference
- **Auto-Chunking**: Process long audio files without OOM errors
- **Memory Management**: Automatic GPU cache cleanup between operations
- **Simple API**: Easy-to-use interface for audio separation

## Available Models

### Supported Models

| Model | HuggingFace ID | VRAM (Lite) | Use Case |
|-------|----------------|-------------|----------|
| `small` | `facebook/sam-audio-small` | **~4 GB** | Fast inference, limited VRAM |
| `base` | `facebook/sam-audio-base` | **~5 GB** | **Recommended** for most use cases |
| `large` | `facebook/sam-audio-large` | **~7 GB** | Best quality |

### Quality Scores (Subjective 1-5)

| Category | Small | Base | Large |
|----------|-------|------|-------|
| General SFX | 3.62 | 3.28 | 3.50 |
| Speech | 3.99 | **4.25** | 4.03 |
| Speaker | 3.12 | 3.57 | **3.60** |
| Music | 4.11 | 3.87 | **4.22** |
| Instruments (wild) | 3.56 | **3.66** | 3.66 |
| Instruments (pro) | 4.24 | 4.27 | **4.49** |

### Related HuggingFace Resources

| Resource | Type | Purpose |
|----------|------|---------|
| `facebook/sam-audio-judge` | Model | Quality assessment (used internally for re-ranking) |
| `facebook/sam-audio-bench` | Dataset | Evaluation benchmark |
| `facebook/sam-audio-musdb18hq-test` | Dataset | Music separation evaluation |

> **Note**: Visual-optimized models (`-tv` variants like `facebook/sam-audio-base-tv`) are **not supported** because this package removes the vision encoder to reduce VRAM. For video-based separation, use the original [sam-audio](https://github.com/facebookresearch/sam-audio) package.

### VRAM by Configuration (Base Model + bfloat16)

| Configuration | VRAM | Features | Use Case |
|--------------|------|----------|----------|
| `aggressive()` | ~4-5 GB | Basic separation | Maximum VRAM savings |
| `with_text_ranker()` | ~6-7 GB | + Reranking | Better quality |
| `with_span_predictor()` | ~6-7 GB | + Time segments | Locate sounds in audio |
| `with_all_features()` | ~8-9 GB | + Both | Best quality |

## Prerequisites

### HuggingFace Access

SAM-Audio models are gated. Before using this package:

1. **Create a HuggingFace account** at [huggingface.co](https://huggingface.co)
2. **Request access** to the model at [facebook/sam-audio-base](https://huggingface.co/facebook/sam-audio-base)
3. **Generate an access token** at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Environment Setup

Create a `.env` file in your project root (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` with your HuggingFace token:

```env
HF_TOKEN=hf_your_token_here
```

The package automatically loads this token - no need to pass it in code.

Alternatively, you can:
- Set the environment variable directly: `export HF_TOKEN=hf_your_token_here`
- Use HuggingFace CLI: `huggingface-cli login`
- Pass token explicitly: `SamAudioInfer.from_pretrained("base", hf_token="...")`

## Installation

### Using uv (Recommended)

```bash
# Add to your project
uv add sam-audio-infer --path /path/to/sam-audio-infer

# Or install from git (when published)
uv add git+https://github.com/your-username/sam-audio-infer.git
```

### Using pip

```bash
git clone https://github.com/your-username/sam-audio-infer.git
cd sam-audio-infer
pip install -e .
```

## Quick Start

### Python API

```python
from sam_audio_infer import SamAudioInfer

# Load model with lite mode (recommended, ~4-5 GB)
model = SamAudioInfer.from_pretrained(
    "base",              # Model size: "small", "base", or "large"
    lite_mode=True,      # Remove unused components (~40% VRAM savings)
    dtype="bfloat16",    # Mixed precision (~50% additional savings)
)

# Separate vocals
result = model.separate("song.wav", description="vocals")
result.save("vocals.wav", "accompaniment.wav")

# Separate multiple stems
results = model.separate_batch(
    "song.wav",
    descriptions=["vocals", "drums", "bass", "other"]
)
```

#### Optional Features for Better Quality

```python
# With Text Ranker for improved quality (~6-7 GB)
# Generates multiple candidates and selects the best one
model = SamAudioInfer.from_pretrained(
    "base",
    lite_mode=True,
    enable_text_ranker=True,     # Keep text ranker
    reranking_candidates=5,       # Generate 5 candidates (higher = better but slower)
)

# With Span Predictor for time segment detection (~6-7 GB)
# Automatically predicts where the target sound occurs
model = SamAudioInfer.from_pretrained(
    "base",
    lite_mode=True,
    enable_span_predictor=True,   # Keep span predictor
)

# With both features for best quality (~8-9 GB)
model = SamAudioInfer.from_pretrained(
    "base",
    lite_mode=True,
    enable_text_ranker=True,
    enable_span_predictor=True,
    reranking_candidates=5,
)
```

#### Using LiteModelConfig for Full Control

```python
from sam_audio_infer import SamAudioInfer, LiteModelConfig

# Pre-built configurations
config = LiteModelConfig.aggressive()        # Most VRAM efficient
config = LiteModelConfig.with_text_ranker(reranking_candidates=5)
config = LiteModelConfig.with_span_predictor()
config = LiteModelConfig.with_all_features(reranking_candidates=3)

# Custom configuration
config = LiteModelConfig(
    remove_vision_encoder=True,   # Always remove for audio-only
    remove_visual_ranker=True,    # Always remove for audio-only
    remove_text_ranker=False,     # Keep for quality
    remove_span_predictor=True,   # Remove to save VRAM
    reranking_candidates=5,
    predict_spans=False,
)

model = SamAudioInfer.from_pretrained("base", lite_config=config)
```

### Command Line

```bash
# Basic usage
sam-audio-infer input.wav --description "vocals" --output vocals.wav

# Use large model for better quality
sam-audio-infer input.wav -d "vocals" -o vocals.wav --model large

# With all options
sam-audio-infer input.wav \
    --description "drums" \
    --output drums.wav \
    --model large \
    --lite \
    --dtype bfloat16 \
    --verbose
```

---

## How It Works

SAM-Audio is a multimodal model designed for both audio AND video inputs. For audio-only separation, several components are **never used** - this is the key insight behind our optimizations.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SAM-Audio Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  Audio Encoder   │  │  Vision Encoder  │  │ Text Encoder │  │
│  │                  │  │                  │  │              │  │
│  │  (REQUIRED)      │  │  (~2GB) ❌       │  │ (REQUIRED)   │  │
│  └────────┬─────────┘  └────────┬─────────┘  └──────┬───────┘  │
│           │                     │                    │          │
│           └──────────┬──────────┴────────────────────┘          │
│                      ▼                                          │
│           ┌──────────────────────┐                              │
│           │   Fusion / Decoder   │                              │
│           │      (REQUIRED)      │                              │
│           └──────────┬───────────┘                              │
│                      │                                          │
│    ┌─────────────────┼─────────────────┐                        │
│    ▼                 ▼                 ▼                        │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Visual   │  │    Text      │  │    Span      │               │
│  │ Ranker   │  │   Ranker     │  │  Predictor   │               │
│  │ (~2GB) ❌ │  │  (~2GB) ❌   │  │ (~1-2GB) ❌  │               │
│  └──────────┘  └──────────────┘  └──────────────┘               │
│                                                                 │
│  ❌ = Removed in Lite Mode (not needed for audio-only)          │
└─────────────────────────────────────────────────────────────────┘
```

### Optimization Techniques

We apply **5 key optimizations** to reduce VRAM by 60-65%:

#### 1. Lite Mode - Component Removal (~40% savings)

Remove unused components for audio-only tasks:

| Component | VRAM Saved | Function | Recommendation |
|-----------|------------|----------|----------------|
| `vision_encoder` | ~2 GB | Video input processing | **Always remove** (audio-only) |
| `visual_ranker` | ~2 GB | Visual quality ranking | **Always remove** (audio-only) |
| `text_ranker` | ~2 GB | Candidate reranking | Optional (improves quality) |
| `span_predictor` | ~1-2 GB | Time segment detection | Optional (locates sounds) |

##### Text Ranker (Optional, +~2 GB)

The text ranker improves separation quality by:
1. Generating multiple separation candidates (`reranking_candidates`)
2. Scoring each using CLAP (audio-text similarity) and Judge models
3. Selecting the best result

```python
# Enable text ranker for better quality
model = SamAudioInfer.from_pretrained(
    "base",
    lite_mode=True,
    enable_text_ranker=True,
    reranking_candidates=5,  # Higher = better quality, slower
)
```

##### Span Predictor (Optional, +~1-2 GB)

The span predictor automatically identifies **when** the target sound occurs:
1. Predicts time segments where the target sound is present
2. Focuses separation on those segments
3. Especially useful for non-ambient sounds (e.g., "horn honking", "dog barking")

```python
# Enable span predictor for time-aware separation
model = SamAudioInfer.from_pretrained(
    "base",
    lite_mode=True,
    enable_span_predictor=True,
)
```

##### Configuration Presets

```python
from sam_audio_infer import LiteModelConfig

# Aggressive - Maximum VRAM savings (~4-5 GB)
config = LiteModelConfig.aggressive()

# With Text Ranker - Better quality (~6-7 GB)
config = LiteModelConfig.with_text_ranker(reranking_candidates=5)

# With Span Predictor - Time-aware (~6-7 GB)
config = LiteModelConfig.with_span_predictor()

# With All Features - Best quality (~8-9 GB)
config = LiteModelConfig.with_all_features(reranking_candidates=5)

model = SamAudioInfer.from_pretrained("base", lite_config=config)
```

#### 2. Mixed Precision - bfloat16 (~50% additional savings)

| Precision | VRAM Usage | Quality |
|-----------|------------|---------|
| float32 | 100% | Best |
| float16 | ~50% | Good (can be unstable) |
| **bfloat16** | **~50%** | **Excellent** (recommended) |

```python
model = SamAudioInfer.from_pretrained("base", dtype="bfloat16")
```

#### 3. Audio Chunking

Process long audio in segments to avoid OOM:

```python
# Automatic chunking (default: 25 seconds)
result = model.separate("long_song.wav", description="vocals")

# Custom chunk duration
result = model.separate("long_song.wav", "vocals", chunk_duration=30.0)
```

#### 4. Memory Management

```python
from sam_audio_infer import cleanup_gpu_memory, get_gpu_memory_info, MemoryTracker

# Check GPU memory
info = get_gpu_memory_info()
print(f"Available: {info.free_gb:.1f} GB")

# Track memory during operations
with MemoryTracker("Separation"):
    result = model.separate("song.wav", "vocals")

# Manual cleanup
cleanup_gpu_memory()
```

#### 5. Inference Optimizations

- `torch.inference_mode()` - Disable gradient computation
- `predict_spans=False` - Skip span prediction
- `reranking_candidates=1` - Don't rerank results

---

## Benchmarks

### VRAM Usage

| Model | Original | Lite + bfloat16 | Reduction |
|-------|----------|-----------------|-----------|
| Small | ~10 GB | **~4 GB** | **60%** |
| Base | ~13 GB | **~5 GB** | **62%** |
| Large | ~20 GB | **~7 GB** | **65%** |

### Processing Speed (RTX 4090, 4:26 audio)

| Model | First Run | Cached | Realtime Factor |
|-------|-----------|--------|-----------------|
| Small | ~78s | ~25s | ~10x |
| Base | ~100s | ~29s | ~9x |
| Large | ~130s | ~41s | ~6.5x |

### GPU Compatibility

| GPU | VRAM | Small | Base | Large |
|-----|------|-------|------|-------|
| RTX 3060 | 6 GB | ✅ | ⚠️ | ❌ |
| RTX 3070 | 8 GB | ✅ | ✅ | ❌ |
| RTX 3080 | 10 GB | ✅ | ✅ | ⚠️ |
| RTX 4070 | 12 GB | ✅ | ✅ | ✅ |
| RTX 4090 | 24 GB | ✅ | ✅ | ✅ |

✅ Comfortable | ⚠️ Tight (smaller chunks) | ❌ Not recommended

---

## API Reference

### SamAudioInfer

```python
class SamAudioInfer:
    @classmethod
    def from_pretrained(
        model_name_or_path: str,  # "small", "base", "large", or HuggingFace ID
        lite_mode: bool = True,
        lite_config: LiteModelConfig = None,  # Custom config (overrides below)
        enable_text_ranker: bool = False,     # Keep for better quality (+~2GB)
        enable_span_predictor: bool = False,  # Keep for time segments (+~1-2GB)
        reranking_candidates: int = 3,        # Candidates for text ranker
        device: str = "cuda",
        dtype: str = "bfloat16",
        chunk_duration: float = 25.0,
        hf_token: str = None,                 # Or set HF_TOKEN in .env
        verbose: bool = True,
    ) -> SamAudioInfer: ...

    def separate(
        audio: AudioInput,
        description: str,
        chunk_duration: float = None,
        verbose: bool = False,
    ) -> SeparationResult: ...

    def separate_batch(
        audio: AudioInput,
        descriptions: list[str],
        verbose: bool = False,
    ) -> list[SeparationResult]: ...
```

### LiteModelConfig

```python
@dataclass
class LiteModelConfig:
    # Components to remove
    remove_vision_encoder: bool = True   # Always True for audio-only
    remove_visual_ranker: bool = True    # Always True for audio-only
    remove_text_ranker: bool = True      # False to keep for quality
    remove_span_predictor: bool = True   # False to keep for time segments

    # Inference settings
    predict_spans: bool = False          # True when using span predictor
    reranking_candidates: int = 1        # >1 when using text ranker

    # Class methods for common configurations
    @classmethod
    def aggressive(cls) -> LiteModelConfig: ...       # ~4-5 GB
    @classmethod
    def with_text_ranker(cls, reranking_candidates=3) -> LiteModelConfig: ...  # ~6-7 GB
    @classmethod
    def with_span_predictor(cls) -> LiteModelConfig: ...  # ~6-7 GB
    @classmethod
    def with_all_features(cls, reranking_candidates=3) -> LiteModelConfig: ...  # ~8-9 GB
```

### SeparationResult

```python
@dataclass
class SeparationResult:
    target: torch.Tensor      # Extracted audio
    residual: torch.Tensor    # Remaining audio
    sample_rate: int
    description: str
    processing_time: float
    num_chunks: int

    def save(target_path: str, residual_path: str = None) -> None: ...
```

---

## Requirements

- Python >= 3.11
- PyTorch >= 2.0.0
- torchaudio >= 2.0.0
- CUDA-capable GPU with at least 4GB VRAM (lite + bfloat16)
- Recommended: 8GB+ VRAM

---

## Acknowledgments

### SAM-Audio (Meta AI / Facebook Research)

This package is built upon **SAM-Audio** (Segment Anything for Audio), developed by Meta AI.

- **Repository**: [github.com/facebookresearch/sam-audio](https://github.com/facebookresearch/sam-audio)
- **Paper**: *Segment Anything for Audio*

We express our sincere gratitude to the Meta AI team for open-sourcing this powerful model.

### AudioGhost AI

The **Lite Mode optimization technique** - the core innovation that reduces VRAM by ~40% - was pioneered in the **AudioGhost AI** project.

- **Key Innovation**: Discovery that SAM-Audio's vision encoder, rankers, and span predictor can be safely removed for audio-only separation tasks
- **Techniques Contributed**:
  - Component removal strategy
  - Memory management patterns
  - Chunking implementation for long audio

We acknowledge and thank the AudioGhost AI developers for their groundbreaking optimization work that made consumer-GPU deployment possible.

### Technical References

1. **Mixed Precision Training** - Micikevicius et al., 2018
2. **PyTorch Memory Management** - [pytorch.org/docs/stable/notes/cuda.html](https://pytorch.org/docs/stable/notes/cuda.html)

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
