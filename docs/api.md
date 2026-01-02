# Python API Reference

## SamAudioInfer

Main class for audio separation.

```python
from sam_audio_kit import SamAudioInfer

# Load model
model = SamAudioInfer.from_pretrained(
    model_name_or_path="base",  # "small", "base", "large", or HuggingFace ID
    dtype="bfloat16",           # "float32", "float16", "bfloat16"
    enable_text_ranker=False,   # Enable text ranker (+~3GB VRAM)
    enable_span_predictor=False,# Enable span predictor (+~3GB VRAM)
    device="cuda",              # "cuda", "cpu", "mps"
    chunk_duration=25.0,        # Default chunk size (seconds)
    hf_token=None,              # HuggingFace token (or use env)
    cache_dir=None,             # Cache directory (or use env)
    verbose=True,               # Print loading progress
)

# Properties
model.sample_rate  # 48000
model.device       # "cuda"
model.dtype        # "bfloat16"

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

## SeparationResult

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

## Download Functions

```python
from sam_audio_kit import (
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

## Memory Management

```python
from sam_audio_kit import (
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

## PrecisionConfig

Configure numerical precision settings.

```python
from sam_audio_kit import PrecisionConfig

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
