# CLI Reference

## Commands Overview

```bash
sam-audio-kit --help
```

| Command | Description |
|---------|-------------|
| `separate` | Separate audio based on text description |
| `download` | Download and cache model files |
| `list` | List cached models |
| `clear` | Clear cached models |

## separate

Separate audio based on text description.

```bash
sam-audio-kit separate <input> -d <description> -o <output> [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `<input>` | Input audio file | Required |
| `-d, --description` | What to extract (e.g., "vocals") | Required |
| `-o, --output` | Output file path | Required |
| `--residual` | Output path for residual audio | None |
| `--model` | Model size: small, base, large | base |
| `--dtype` | float32, float16, bfloat16 | bfloat16 |
| `--enable-text-ranker` | Enable text ranker (+~3GB VRAM) | False |
| `--enable-span-predictor` | Enable span predictor (+~3GB VRAM) | False |
| `--device` | cuda, cpu, mps | cuda |
| `--chunk-duration` | Chunk size in seconds | 25.0 |
| `--cache-dir` | Model cache directory | env/default |
| `--hf-token` | HuggingFace token | env |
| `--warmup` | Run warmup before processing | False |
| `-v, --verbose` | Verbose output | False |

### Precision Options

| Flag | Description |
|------|-------------|
| `--tf32` | Enable TF32 (default) |
| `--no-tf32` | Disable TF32 for maximum precision |
| `--matmul-precision` | `highest`, `high`, `medium` |
| `--cudnn-benchmark` | Enable cuDNN auto-tuner (default) |
| `--no-cudnn-benchmark` | Disable cuDNN auto-tuner |
| `--deterministic` | Force reproducible results |

### Examples

```bash
# Basic usage (~3 GB VRAM)
sam-audio-kit separate song.wav -d "vocals" -o vocals.wav

# Extract drums with residual
sam-audio-kit separate song.wav -d "drums" -o drums.wav --residual other.wav

# Use large model with warmup
sam-audio-kit separate song.wav -d "bass" -o bass.wav --model large --warmup -v

# Enable text ranker for better quality (+3 GB VRAM)
sam-audio-kit separate song.wav -d "vocals" -o vocals.wav --enable-text-ranker

# Fast inference
sam-audio-kit separate audio.wav -d "vocals" -o out.wav --matmul-precision medium

# Maximum quality
sam-audio-kit separate audio.wav -d "vocals" -o out.wav --no-tf32 --matmul-precision highest
```

## download

Download and cache model files.

```bash
sam-audio-kit download [options]
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

### Examples

```bash
# Download base model
sam-audio-kit download --model base

# Download and warmup for production deployment
sam-audio-kit download --model base --warmup

# Download all sizes
sam-audio-kit download --model small
sam-audio-kit download --model base
sam-audio-kit download --model large
```

## list

List cached models.

```bash
sam-audio-kit list [--cache-dir DIR]
```

## clear

Clear cached models.

```bash
sam-audio-kit clear [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Specific model to clear | all |
| `--cache-dir` | Cache directory | env/default |
| `-y, --yes` | Skip confirmation | False |
