# Architecture & How It Works

SAM-Audio is a multimodal model designed for both audio AND video inputs. For audio-only separation, several components are **never used** - this is the key insight behind our optimizations.

## Model Architecture

```
+-----------------------------------------------------------------+
|                    SAM-Audio Architecture                        |
+-----------------------------------------------------------------+
|                                                                  |
|  +------------------+  +------------------+  +--------------+   |
|  |  Audio Encoder   |  |  Vision Encoder  |  | Text Encoder |   |
|  |                  |  |                  |  |              |   |
|  |  (REQUIRED)      |  |  (~2GB) X        |  | (REQUIRED)   |   |
|  +--------+---------+  +--------+---------+  +------+-------+   |
|           |                     |                    |           |
|           +----------+----------+--------------------+           |
|                      v                                           |
|           +----------------------+                               |
|           |   Fusion / Decoder   |                               |
|           |      (REQUIRED)      |                               |
|           +----------+-----------+                               |
|                      |                                           |
|    +-----------------+-----------------+                         |
|    v                 v                 v                         |
|  +----------+  +--------------+  +--------------+                |
|  | Visual   |  |    Text      |  |    Span      |                |
|  | Ranker   |  |   Ranker     |  |  Predictor   |                |
|  | (~2GB) X |  |  (~2GB) ?    |  | (~1-2GB) ?   |                |
|  +----------+  +--------------+  +--------------+                |
|                                                                  |
|  X = Always removed (not needed for audio-only)                  |
|  ? = Optionally kept (improves quality)                          |
+-----------------------------------------------------------------+
```

## Optimization Techniques

### 1. Lite Mode (~40% VRAM savings)

Remove unused components for audio-only tasks:

| Component | VRAM Saved | Removed by Default |
|-----------|------------|-------------------|
| `vision_encoder` | ~2 GB | Always |
| `visual_ranker` | ~2 GB | Always |
| `text_ranker` | ~2 GB | (optional keep) |
| `span_predictor` | ~1-2 GB | (optional keep) |

### 2. Mixed Precision (~50% additional savings)

| Precision | VRAM Usage | Quality |
|-----------|------------|---------|
| float32 | 100% | Best |
| float16 | ~50% | Good (can be unstable) |
| **bfloat16** | **~50%** | **Excellent** (recommended) |

### 3. Audio Chunking

Process long audio in segments to avoid OOM:

```python
# Automatic chunking (default: 25 seconds)
result = model.separate("long_song.wav", description="vocals")

# Custom chunk duration
result = model.separate("long_song.wav", "vocals", chunk_duration=30.0)
```

### 4. Warmup (First Prediction Caching)

The first inference is slower due to CUDA kernel compilation. Warmup pre-compiles these:

```python
from sam_audio_infer import warmup_model

# After loading model
warmup_model(model, duration_seconds=1.0)
# Subsequent inferences will be faster
```

## Why Lite Mode Works

SAM-Audio was designed as a multimodal model supporting:
- Audio-only separation (using text descriptions)
- Audio-visual separation (using video frames)
- Time-span prediction (finding when sounds occur)

For audio-only tasks:
- **Vision encoder**: Never processes any input (no video frames)
- **Visual ranker**: Never ranks anything (no visual candidates)
- **Text ranker**: Optional quality improvement (reranks output candidates)
- **Span predictor**: Optional time segment prediction

By removing the unused vision components, we achieve significant VRAM savings without any loss in audio-only separation quality.
