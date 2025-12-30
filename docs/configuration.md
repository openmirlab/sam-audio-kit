# Configuration

## Available Models

| Model | HuggingFace ID | VRAM (Lite) | Use Case |
|-------|----------------|-------------|----------|
| `small` | `facebook/sam-audio-small` | **~4 GB** | Fast inference, limited VRAM |
| `base` | `facebook/sam-audio-base` | **~5 GB** | **Recommended** for most use cases |
| `large` | `facebook/sam-audio-large` | **~7 GB** | Best quality |

## VRAM by Configuration (Base Model + bfloat16)

| text_ranker | span_predictor | VRAM | Features |
|-------------|----------------|------|----------|
| `False` | `False` | **~3 GB** | Basic separation (recommended) |
| `True` | `False` | ~6 GB | + Quality reranking |
| `False` | `True` | ~6 GB | + Time segments |
| `True` | `True` | ~9 GB | + Both features |

## Quality Scores (Subjective 1-5)

| Category | Small | Base | Large |
|----------|-------|------|-------|
| General SFX | 3.62 | 3.28 | 3.50 |
| Speech | 3.99 | **4.25** | 4.03 |
| Music | 4.11 | 3.87 | **4.22** |
| Instruments (pro) | 4.24 | 4.27 | **4.49** |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SAM_AUDIO_CACHE_DIR` | Model cache directory | `~/.cache/sam-audio-infer` |
| `HF_TOKEN` | HuggingFace API token | None |
| `HUGGINGFACE_TOKEN` | Alternative HF token | None |
| `SAM_AUDIO_TF32` | Enable TF32 | `true` |
| `SAM_AUDIO_MATMUL_PRECISION` | Matrix multiplication precision | `high` |
| `SAM_AUDIO_CUDNN_BENCHMARK` | Enable cuDNN benchmark | `true` |
| `SAM_AUDIO_DETERMINISTIC` | Force deterministic mode | `false` |

## Precision Settings

Fine-grained control over numerical precision for speed vs quality trade-offs.

### Options

| Setting | Values | Description |
|---------|--------|-------------|
| `matmul_precision` | `highest`, `high`, `medium` | Internal precision for matrix multiplications |
| `tf32` | `true`, `false` | Enable TensorFloat-32 (~3x speedup on Ampere+ GPUs) |
| `cudnn_benchmark` | `true`, `false` | Auto-tune convolution algorithms |
| `deterministic` | `true`, `false` | Force reproducible results (slower) |

### What is `matmul_precision`?

Controls the internal precision PyTorch uses for matrix multiplications:

| Value | Internal Precision | Speed | Quality |
|-------|-------------------|-------|---------|
| `highest` | Full float32 (32-bit) | Slowest | Best |
| `high` | TF32 (19-bit mantissa) | ~3x faster | Very good |
| `medium` | Lower precision | Fastest | Good |

**Note:** There is no `low` option. `medium` is the fastest available.

For `dtype=bfloat16`, the impact is smaller since tensors are already in reduced precision.

### Environment Variables (Defaults)

Set these to configure defaults across all runs:

```bash
export SAM_AUDIO_TF32=true                    # default: true
export SAM_AUDIO_MATMUL_PRECISION=high        # default: high
export SAM_AUDIO_CUDNN_BENCHMARK=true         # default: true
export SAM_AUDIO_DETERMINISTIC=false          # default: false
```

### CLI Flags (Override Per-Run)

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

### Recommendations

| Use Case | Settings |
|----------|----------|
| **Production** (fastest) | `matmul=medium`, `tf32=true`, `cudnn_benchmark=true` |
| **Balanced** (default) | `matmul=high`, `tf32=true`, `cudnn_benchmark=true` |
| **Quality-critical** | `matmul=highest`, `tf32=false` |
| **Reproducible** | `matmul=highest`, `tf32=false`, `deterministic=true` |
