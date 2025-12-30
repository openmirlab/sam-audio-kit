# SAM-Audio-Infer

Optimized inference package for Meta's SAM-Audio model with VRAM-efficient lite mode.

## Features

- **Multiple Model Sizes**: Support for `small`, `base`, and `large` models
- **Lite Mode**: Reduce VRAM usage by ~40% by removing unused components
- **Mixed Precision**: Support for bfloat16/float16 inference
- **Auto-Chunking**: Process long audio files without OOM errors
- **Memory Management**: Automatic GPU cache cleanup between operations
- **Simple API**: Easy-to-use interface for audio separation

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
# Clone and install
git clone https://github.com/your-username/sam-audio-infer.git
cd sam-audio-infer
pip install -e .
```

### Development Installation

```bash
uv add sam-audio-infer --path /path/to/sam-audio-infer --dev
# or
pip install -e ".[dev]"
```

## Model Sizes

| Model | Full VRAM | Lite + bfloat16 | Quality | Use Case |
|-------|-----------|-----------------|---------|----------|
| `small` | ~10 GB | **~4 GB** | Good | Quick prototyping, limited VRAM |
| `base` | ~13 GB | **~5 GB** | Better | Recommended for most use cases |
| `large` | ~20 GB | **~7 GB** | Best | Highest quality, more VRAM |

## Quick Start

### Python API

```python
from sam_audio_infer import SamAudioInfer

# Load BASE model with lite mode (recommended for most users)
model = SamAudioInfer.from_pretrained(
    "base",              # Model size: "small", "base", or "large"
    lite_mode=True,      # Remove unused components (~40% VRAM savings)
    dtype="bfloat16",    # Mixed precision (~50% additional savings)
)

# Separate vocals
result = model.separate("song.wav", description="vocals")
result.save("vocals.wav", "accompaniment.wav")

# Load LARGE model for highest quality
model_large = SamAudioInfer.from_pretrained(
    "large",             # Use large model
    lite_mode=True,      # Still use lite mode to save VRAM
    dtype="bfloat16",
)

# Separate multiple stems
results = model.separate_batch(
    "song.wav",
    descriptions=["vocals", "drums", "bass", "other"]
)
```

### Command Line

```bash
# Basic usage (uses base model by default)
sam-audio-infer input.wav --description "vocals" --output vocals.wav

# Use large model for better quality
sam-audio-infer input.wav -d "vocals" -o vocals.wav --model large

# Use small model for faster processing / limited VRAM
sam-audio-infer input.wav -d "vocals" -o vocals.wav --model small

# With all options
sam-audio-infer input.wav \
    --description "drums" \
    --output drums.wav \
    --residual other.wav \
    --model large \
    --lite \
    --dtype bfloat16 \
    --chunk-duration 25 \
    --verbose
```

## VRAM Usage

| Model | Full float32 | Full bfloat16 | Lite float32 | Lite bfloat16 |
|-------|--------------|---------------|--------------|---------------|
| Small | ~10 GB | ~6 GB | ~6 GB | **~4 GB** |
| Base | ~13 GB | ~7 GB | ~8 GB | **~5 GB** |
| Large | ~20 GB | ~10 GB | ~12 GB | **~7 GB** |

## Lite Mode

Lite mode removes components that are not needed for audio-only separation:

| Component | VRAM Saved | Purpose (removed) |
|-----------|------------|-------------------|
| Vision Encoder | ~2 GB | Video input processing |
| Visual Ranker | ~2 GB | Visual quality ranking |
| Text Ranker | ~2 GB | Result reranking |
| Span Predictor | ~1-2 GB | Segment prediction |

### Enabling Lite Mode

```python
# Automatic (aggressive - removes all optional components)
model = SamAudioInfer.from_pretrained("base", lite_mode=True)

# Custom configuration
from sam_audio_infer import LiteModelConfig

config = LiteModelConfig(
    remove_vision_encoder=True,
    remove_visual_ranker=True,
    remove_text_ranker=False,  # Keep text ranker
    remove_span_predictor=True,
)
model = SamAudioInfer.from_pretrained("base", lite_mode=True, lite_config=config)
```

## Audio Chunking

Long audio files are automatically processed in chunks to avoid OOM errors:

```python
# Default: 25 second chunks
result = model.separate("long_song.wav", description="vocals")

# Custom chunk duration
result = model.separate(
    "long_song.wav",
    description="vocals",
    chunk_duration=30.0,  # seconds
)
```

## Memory Management

```python
from sam_audio_infer import cleanup_gpu_memory, get_gpu_memory_info, MemoryTracker

# Check GPU memory
info = get_gpu_memory_info()
print(f"Available: {info.free_gb:.1f} GB")

# Manual cleanup
cleanup_gpu_memory()

# Track memory during operations
with MemoryTracker("Separation"):
    result = model.separate("song.wav", "vocals")
# Prints: GPU Memory [Separation] - Before: 5.00GB, After: 5.50GB, Delta: +0.50GB
```

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

    def save(
        target_path: str,
        residual_path: str = None,
    ) -> None: ...
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- torchaudio >= 2.0.0
- sam-audio (from GitHub)

### GPU Requirements

- CUDA-capable GPU with at least 4GB VRAM (lite mode + bfloat16)
- Recommended: 8GB+ VRAM for comfortable operation

## Documentation

- **[VRAM Optimization Guide](docs/VRAM_OPTIMIZATION.md)** - Comprehensive guide on how we achieve memory efficiency

## License

MIT License

**Note**: The underlying SAM-Audio model has its own license terms. Please refer to the [official SAM-Audio repository](https://github.com/facebookresearch/sam-audio) for model usage terms.

## Acknowledgments

### SAM-Audio (Meta AI / Facebook Research)

This package is built upon **SAM-Audio** (Segment Anything for Audio), an incredible model developed by Meta AI that enables natural language-based audio separation.

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

## How It Works

See our [VRAM Optimization Guide](docs/VRAM_OPTIMIZATION.md) for a detailed technical explanation of:

1. **Lite Mode** - Removing unused model components (~40% VRAM reduction)
2. **Mixed Precision** - Using bfloat16 (~50% additional reduction)
3. **Audio Chunking** - Processing long files without OOM
4. **Memory Management** - Aggressive cleanup strategies

## Citation

If you use this package in your research, please cite both SAM-Audio and acknowledge the optimization contributions:

```bibtex
@software{sam_audio_infer,
  title = {SAM-Audio-Infer: Memory-Efficient SAM-Audio Inference},
  year = {2024},
  note = {Lite mode optimization based on AudioGhost AI}
}
```
