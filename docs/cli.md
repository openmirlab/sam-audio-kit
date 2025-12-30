# CLI Reference

## Commands Overview

```bash
sam-audio-infer --help
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
# Basic usage
sam-audio-infer separate song.wav -d "vocals" -o vocals.wav

# Extract drums with residual
sam-audio-infer separate song.wav -d "drums" -o drums.wav --residual other.wav

# Use large model with warmup
sam-audio-infer separate song.wav -d "bass" -o bass.wav --model large --warmup -v

# Fast inference
sam-audio-infer separate audio.wav -d "vocals" -o out.wav --matmul-precision medium

# Maximum quality
sam-audio-infer separate audio.wav -d "vocals" -o out.wav --no-tf32 --matmul-precision highest
```

## download

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

### Examples

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

## list

List cached models.

```bash
sam-audio-infer list [--cache-dir DIR]
```

## clear

Clear cached models.

```bash
sam-audio-infer clear [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Specific model to clear | all |
| `--cache-dir` | Cache directory | env/default |
| `-y, --yes` | Skip confirmation | False |
