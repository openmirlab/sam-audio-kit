# Troubleshooting

## Common Issues

### Model download fails

SAM-Audio models are gated and require access approval.

1. Request access to the model checkpoints:
   - [facebook/sam-audio-base](https://huggingface.co/facebook/sam-audio-base)
   - [facebook/sam-audio-large](https://huggingface.co/facebook/sam-audio-large)

2. Once accepted, authenticate with HuggingFace:
   ```bash
   # Option 1: CLI login (recommended)
   huggingface-cli login

   # Option 2: Environment variable
   export HF_TOKEN=hf_your_token_here
   ```

3. Verify authentication:
   ```bash
   huggingface-cli whoami
   ```

### Out of memory (OOM)

```python
# Use smaller chunks
result = model.separate("song.wav", "vocals", chunk_duration=15.0)

# Or use smaller model
model = SamAudioInfer.from_pretrained("small", lite_mode=True)

# Or use lite mode with aggressive config
model = SamAudioInfer.from_pretrained("base", lite_mode=True)
```

### Slow first inference

The first inference is slower due to CUDA kernel compilation:

```python
# Use warmup
from sam_audio_infer import warmup_model
warmup_model(model, duration_seconds=1.0)
```

Or via CLI:

```bash
sam-audio-infer download --model base --warmup
```

### CUDA not available

```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())  # Should be >= 1
```

If CUDA is not available:
1. Ensure you have an NVIDIA GPU
2. Install CUDA toolkit
3. Reinstall PyTorch with CUDA support: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### Permission denied for cache directory

```bash
# Set a custom cache directory with write access
export SAM_AUDIO_CACHE_DIR=/path/to/writable/directory
```

### Model outputs are silent or corrupted

1. Check input audio sample rate (should work with any rate, resampled internally to 48kHz)
2. Ensure input audio is not empty or corrupted
3. Try a different description
4. Check if the input contains the requested content

### Reproducibility issues

For reproducible results:

```bash
sam-audio-infer separate audio.wav -d "vocals" -o out.wav \
    --deterministic --no-cudnn-benchmark --no-tf32 --matmul-precision highest
```

Or in Python:

```python
from sam_audio_infer import SamAudioInfer, PrecisionConfig

config = PrecisionConfig(
    matmul_precision="highest",
    allow_tf32=False,
    cudnn_benchmark=False,
    cudnn_deterministic=True,
)
model = SamAudioInfer.from_pretrained("base", precision_config=config)
```
