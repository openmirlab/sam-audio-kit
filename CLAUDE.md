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

Decision: publish public GitHub source/releases and support GitHub installation;
**hold PyPI** as the more
conservative distribution channel for now (a `pip install` is lower-friction
and less likely to be read before running than a deliberate git-clone
install, given the license's non-OSI use restrictions). `.github/workflows/publish.yml`'s
actual "Publish to PyPI" step stays commented out until a future decision
to lift that hold -- re-enabling it does not require a further license
review, only a fresh decision on distribution channel. GitHub distribution
must retain `LICENSE`, `LICENSE.SAM-AUDIO`, and `LICENSE.PE`; gated checkpoints
remain user-authenticated and are never bundled or mirrored. This is policy
documentation, not legal advice.

## Inference-only boundary evidence

The shipped package contains model, preprocessing/postprocessing, download,
cache, and inference runtime only. Training, evaluation, dataset, experiment,
and debug residue is excluded from runtime imports and package metadata. The
2026-07-15 cleanup removed the unused debug probe hook and evaluation-dataset
resource map; see `docs/NAV_AUDIT.md` for the scan and audit evidence.

## Public inference contract

`SamAudioSession` is the independent lifecycle facade: call `load()` before
ready-only `infer()`, then `release()` or `close()`; `status`, `cache_info()`,
and context-manager use are supported. `config/checkpoints.toml` is the
package-owned source for official gated model URLs and provenance. It never
bundles or mirrors weights, and callers may provide explicit metadata
overrides. Keep the legacy `SamAudio` one-shot API lazy and compatible.

**Checkpoint-catalog contract** (org `checkpoint-catalog.md`, article 4):
every `[models.*]` entry in `checkpoints.toml` must carry a pinned commit
`source_revision` (never a floating branch) plus a lowercase `sha256` +
`size_bytes` for its primary artifact, or an explicit
`integrity = "unavailable"` with a documented reason — no silent gap.
`download.py`'s `download_model()` downloads at that pinned revision and
verifies the digest post-download, raising `ChecksumMismatchError` on a
mismatch; `resolve_model_cache_path(..., revision=...)` and
`session.py`'s `cache_info()` resolve the *pinned* snapshot path, not just
any cached snapshot. `facebook/sam-audio-judge`'s catalog entry pins
`checkpoint.pt` (not `model.safetensors`) because
`sam_audio/model/base.py`'s `BaseModel._from_pretrained` hardcodes that
filename for every `BaseModel` subclass, judge included — confirmed by
reading that loader, not assumed. `model.py`'s `SamAudio.from_pretrained()`
threads the catalog's pinned revision through to the underlying
`SAMAudio.from_pretrained(..., revision=...)` call too, which required
fixing a real pre-existing bug in `_from_pretrained` that silently
discarded any passed `revision` in favor of the class-level `cls.revision`
fallback.

`sam_audio/processor.py`'s `Processor`/`SAMAudioProcessor`/
`SAMAudioJudgeProcessor.from_pretrained` (and `_get_config`) also accept an
explicit `revision=` now, same idiom as `BaseModel` (an explicit call wins,
`cls.revision` is the fallback) -- previously they had no such parameter at
all, so `model.py`'s processor load always floated on `cls.revision`
regardless of the pinned revision computed right next to it for the model
load. `model.py`'s `SamAudio.from_pretrained()` now reuses that same
`pinned_revision` value for both the model and processor calls, rather than
computing it once and leaving the processor call unpinned.

## Documentation conformance

README reviewed 2026-07-12 against the org's documentation-conformance shape
(Why-this-exists -> Acknowledgments -> Citation -> Features -> Scope ->
Install -> Quick Start -> ... -> "will NEVER bundle" -> Development ->
License -> Support). Two attribution/citation items were specifically
verified against primary sources rather than trusted as-is:

- **Citation** (`arXiv:2512.18099`, "SAM Audio: Segment Anything in Audio",
  14 authors): fetched the arXiv abstract page directly -- title and full
  author list match what's printed in the README. No changes needed.
- **AudioGhost AI attribution** (Lite Mode credited to
  `github.com/0x0funky/audioghost-ai`): checked via `gh api
  repos/0x0funky/audioghost-ai` and `gh api users/0x0funky` -- both the repo
  (401 stars, created 2025-12-22) and the account are real, and the repo's
  own README independently documents the same "remove unused SAM-Audio
  components for VRAM savings" technique. This is *not* a fabricated
  attribution (unlike a superficially similar case previously found and
  removed in `larsnet-infer`) -- kept as-is.

- PyTorch 2.0+ with torchaudio
- Transformers (HuggingFace)
- DACVAE audio codec
- LAION-CLAP for text ranking

The optional `ClapRanker` imports LAION-CLAP's legacy preprocessing module inside a temporary
argv-isolation boundary. LAION-CLAP parses process arguments at import time; host applications
must not have server flags such as `--port` or `--device` consumed by that dependency.

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
uv run pytest tests/

# Format code
uv run black src/ tests/ examples/

# Lint
uv run ruff check src/ tests/

# Type check
uv run mypy src/

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
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=sam_audio_kit
```

## Environment Variables

- `HF_TOKEN` or `HUGGINGFACE_TOKEN`: Required for gated model access
- See `.env.example` for template
