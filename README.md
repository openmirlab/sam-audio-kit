# SAM-Audio-Infer

Optimized inference package for Meta's SAM-Audio model with VRAM-efficient lite mode.

## Features

- **Multiple Model Sizes**: Support for `small`, `base`, and `large` models
- **Lite Mode**: Reduce VRAM usage by ~40% by removing unused components
- **Mixed Precision**: Support for bfloat16/float16 inference
- **Auto-Chunking**: Process long audio files without OOM errors
- **Memory Management**: Automatic GPU cache cleanup between operations
- **Simple API**: Easy-to-use interface for audio separation

## Model Sizes

| Model | Full VRAM | Lite + bfloat16 | Quality | Use Case |
|-------|-----------|-----------------|---------|----------|
| `small` | ~10 GB | **~4 GB** | Good | Quick prototyping, limited VRAM |
| `base` | ~13 GB | **~5 GB** | Better | Recommended for most use cases |
| `large` | ~20 GB | **~7 GB** | Best | Highest quality, more VRAM |

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

# Load model with lite mode (recommended)
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

| Component | VRAM Saved | Why Not Needed |
|-----------|------------|----------------|
| `vision_encoder` | ~2 GB | Only for video input |
| `visual_ranker` | ~2 GB | Ranks by visual quality |
| `text_ranker` | ~2 GB | Reranks candidates |
| `span_predictor` | ~1-2 GB | Time segmentation |
| **Total** | **~7-8 GB** | |

```python
# Enable lite mode
model = SamAudioInfer.from_pretrained("base", lite_mode=True)

# Custom configuration
from sam_audio_infer import LiteModelConfig

config = LiteModelConfig(
    remove_vision_encoder=True,
    remove_visual_ranker=True,
    remove_text_ranker=False,  # Keep for better quality
    remove_span_predictor=True,
)
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
        model_name_or_path: str,  # "small", "base", "large", or HF model ID
        lite_mode: bool = True,
        lite_config: LiteModelConfig = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        chunk_duration: float = 25.0,
        hf_token: str = None,
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

- Python >= 3.10
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
