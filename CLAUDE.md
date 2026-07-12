# CLAUDE.md - Project Guide for Claude Code

## Project Overview

**sam-audio-kit** is a creative audio toolkit built on Meta AI's SAM-Audio model. It provides VRAM-efficient audio separation using natural language descriptions, along with creative synthesis and effects in latent space.

- **Package**: `sam-audio-kit` (v0.2.0)
- **Python**: 3.10+
- **License**: `LicenseRef-Meta-SAM-License AND MIT AND Apache-2.0` (see LICENSING.md for the per-component map -- this is not a plain-MIT package)

## Release status

Reviewed 2026-07-12. `LICENSE.SAM-AUDIO` section 1.a explicitly grants
redistribution rights ("use, reproduce, distribute, copy, create derivative
works of, and make modifications"), conditioned on bundling the license
text with any distribution (satisfied via `license-files` in
`pyproject.toml`) and complying with the use restrictions in LICENSING.md
(Trade Controls/ITAR, no military/weapons use, no reverse-engineering the
model internals). Redistribution is not legally blocked.

Decision: publish full GitHub releases; **hold PyPI** as the more
conservative distribution channel for now (a `pip install` is lower-friction
and less likely to be read before running than a deliberate git-clone
install, given the license's non-OSI use restrictions). `.github/workflows/publish.yml`'s
actual "Publish to PyPI" step stays commented out until a future decision
to lift that hold -- re-enabling it does not require a further license
review, only a fresh decision on distribution channel.

- PyTorch 2.0+ with torchaudio
- Transformers (HuggingFace)
- DACVAE audio codec
- LAION-CLAP for text ranking

## Project Structure

```
src/sam_audio_kit/
├── model.py          # SamAudioInfer - main inference wrapper
├── inference.py      # Core inference logic
├── lite.py           # Lite mode VRAM optimizations
├── chunking.py       # Audio chunking for long files
├── memory.py         # GPU memory management
├── download.py       # Model downloading & caching
├── precision.py      # Precision configuration
├── cli.py            # Command-line interface
├── sam_audio/        # Extracted SAM-Audio model components
└── synth/            # Creative synthesis module
    ├── codec.py      # DACVAE audio codec
    ├── effects.py    # Latent space effects
    └── granular.py   # Granular synthesis
```

## Development Setup

```bash
# Install dependencies (using uv)
uv sync

# Or with pip
pip install -e ".[dev]"

# Set HuggingFace token for gated models
export HF_TOKEN="your_token"
```

## Common Commands

```bash
# Run tests
pytest tests/

# Format code
black src/ tests/ examples/

# Lint
ruff check src/ tests/

# Type check
mypy src/

# CLI usage
sam-audio-kit separate audio.wav -d "vocals" -o output.wav
sam-audio-kit download --model base --warmup
```

## Code Style

- **Line length**: 100 characters (Black)
- **Linting**: Ruff (E, F, I, N, W, UP rules)
- **Type hints**: MyPy with strict mode
- **Docstrings**: Google style

## Key Classes

- `SamAudioInfer` (`model.py`): Main inference wrapper with unified API
- `LiteModelConfig` (`lite.py`): VRAM optimization configurations
- `AudioChunker` (`chunking.py`): Auto-chunking for long audio (25s default)
- `LatentSynthesizer` (`synth/__init__.py`): Creative latent space manipulation
- `MemoryTracker` (`memory.py`): GPU memory monitoring

## Important Notes

- Models require HuggingFace token acceptance: `facebook/sam-audio-base`, `facebook/sam-audio-large`
- Lite mode removes vision_encoder and visual_ranker for 62-78% VRAM savings
- Audio is processed at 48kHz native sample rate
- Long audio is automatically chunked with overlap and crossfade

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=sam_audio_kit
```

## Environment Variables

- `HF_TOKEN` or `HUGGINGFACE_TOKEN`: Required for gated model access
- See `.env.example` for template
