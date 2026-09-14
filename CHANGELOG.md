# Changelog

All notable changes to sam-audio-kit are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- Pinned every `checkpoints.toml` registry entry (`small`/`base`/`large`, and
  a new `judge` entry) to a specific commit `source_revision` and a lowercase
  `sha256`/`size_bytes` for its primary artifact, read from the Hub's
  git-LFS-recorded metadata (`HfApi().model_info(..., files_metadata=True)`);
  the toml documents this as metadata-only, not yet byte-level-verified
  against a downloaded file (gated raw-content resolve currently 403s for the
  checking account). `download_model()` now downloads at the pinned revision
  and verifies the downloaded artifact's digest, raising the new
  `ChecksumMismatchError` on a mismatch (or a `verbose`-gated skip warning
  for a catalog entry explicitly marked `integrity = "unavailable"`).
  `resolve_model_cache_path()` accepts an optional `revision=` to name a
  pinned snapshot specifically, and `SamAudioSession.cache_info()` now uses
  it so `exists` reflects the *pinned* revision rather than any cached
  snapshot. `SamAudio.from_pretrained()` threads the same pinned revision
  into the underlying `SAMAudio.from_pretrained(..., revision=...)` call.
- Fixed `sam_audio/model/base.py`'s `BaseModel._from_pretrained`, which
  silently discarded any `revision` argument in favor of the class-level
  `cls.revision` fallback -- found while wiring the checkpoint-pinning work
  above; a passed `revision` now takes priority, `cls.revision` remains the
  default when none is given.
- Added `SamAudioSession`, an independent load/infer/release/close lifecycle
  facade with status, cache inspection, and context-manager support.
- Added package-owned `config/checkpoints.toml` metadata for gated official
  model repositories and generic caller overrides; no weights are bundled or
  mirrored.
- Added strict explicit device validation for `cpu`, `cuda`, `cuda:N`, and
  `mps`, preserving legacy automatic selection.

### Changed

- `SamAudio.from_pretrained` no longer builds the components lite mode deletes
  (vision encoder, rankers, span predictor), reads the checkpoint through
  `mmap`, and skips random parameter initialisation. Base model on a 4090:
  load 33 s -> 5 s, host RAM peak 22 GB -> 5.8 GB, output bit-identical.
- Isolated the optional LAION-CLAP preprocessing import from host process CLI
  arguments. Server applications can now load the text ranker without
  `laion_clap` consuming flags such as `--port` or `--device` at import time.
- Clarified that public GitHub source/releases and GitHub installation are
  allowed under the mixed-license terms, while PyPI remains held as a
  separate distribution-channel decision.
- Removed an unused debug probe hook and evaluation-dataset resource metadata
  from the shipped inference surface. Existing tests remain green; the
  optional external CLAP preprocessing import is documented as a future
  adapter candidate.
- `SamAudioSession.release()` is reloadable and `close()` is terminal and
  idempotent. `cache_info()` now reports the same read-only Hugging Face repo
  path the loader uses, without contacting gated Hugging Face endpoints.
- Official shorthand model IDs are now resolved through packaged checkpoint
  TOML at runtime; `MODEL_NAME_MAP` remains a public compatibility fallback.

## [0.2.0] - 2026-07-12

**Release status: full GitHub release, PyPI held (conservative, by user
decision).** The Meta SAM License question was reviewed: section 1.a of
`LICENSE.SAM-AUDIO` explicitly grants the right to "use, reproduce,
distribute, copy, create derivative works of, and make modifications" to
the SAM Materials, subject to bundling the license text with any
redistribution (already satisfied -- see `license-files` below) and the
use restrictions in LICENSING.md (Trade Controls/ITAR, no
military/weapons use). Redistribution is legally permitted; PyPI
publication is deliberately deferred anyway as the more conservative
choice (GitHub-only distribution requires a deliberate git clone/install
step, so a user sees the README's licensing section before anything is
running, versus `pip install`'s lower-friction, less-read path). See
`CLAUDE.md`'s "Release status" section for the decision record. The
uncommitted differentiable-API feature previously noted below has since
landed (`feat/adopt-constitution`, merged).

### Fixed

- **License metadata now tells the truth.** `pyproject.toml` `license` was
  `"MIT AND Apache-2.0"`, which omitted that most of the code
  (`src/sam_audio_kit/sam_audio/**`) is under Meta's custom, non-OSI **SAM
  License**. It is now the SPDX expression
  `LicenseRef-Meta-SAM-License AND MIT AND Apache-2.0`, with
  `license-files = ["LICENSE", "LICENSE.SAM-AUDIO", "LICENSE.PE"]` and the
  misleading `License :: OSI Approved :: MIT License` classifier replaced
  with `License :: Other/Proprietary License`. Added `LICENSING.md` mapping
  every component to its actual license. No Meta term was reinterpreted or
  weakened -- this is a labeling fix only.
- **`dacvae` git dependency removed from package metadata (art.3).** It had
  no PyPI release, and PyPI's upload validation rejects *any* direct
  git/URL dependency in a package's metadata -- including under an optional
  extra -- so as a declared dependency it permanently blocked a PyPI
  release regardless of the license question. It is no longer declared
  anywhere in `pyproject.toml`. `sam_audio_kit/sam_audio/model/codec.py`
  now raises a clear, actionable `ImportError` (with the manual install
  command) if `dacvae` is missing, instead of a bare
  `ModuleNotFoundError`. Manual install instructions are documented in
  README.md ("Installing the audio codec (dacvae)"). Vendoring was
  considered and rejected for now: `dacvae.DACVAE` inherits from
  `audiotools.ml.BaseModel`, so a faithful vendor would also have to fork
  that inheritance away from `descript-audiotools` -- a nontrivial rewrite
  of code the core model depends on, better done deliberately later than as
  part of this cleanup.

### Added

- **`.github/workflows/test.yml`**: a push/PR-triggered CI gate, closing the
  gap where the only CI (`publish.yml`) built/tested a single pinned Python
  (3.11) and only at release time. A `test` job matrixes over every Python
  version `pyproject.toml`'s classifiers claim (3.10, 3.11, 3.12) plus 3.13
  (verified passing locally; not yet in the classifiers). A `build` job adds
  the wheel-from-sdist install smoke test required by org art.7
  (`python -m build`, install the wheel into a clean venv, import the
  package and touch a public symbol) -- adapted for this repo's one real
  constraint: `sam_audio_kit/__init__.py` unconditionally imports the
  SAM-Audio chain, which needs the un-publishable `dacvae` codec (not
  installed in CI), so the smoke test accepts either a real import (touching
  `SamAudio`) or the one specific, documented `ImportError` -- and fails on
  anything else (e.g. the bare `ModuleNotFoundError` an empty/broken wheel
  would produce instead).

### Fixed

- **`tests/test_chunking.py` and `tests/test_memory.py` no longer hard-fail
  collection without `dacvae`.** Both import from `sam_audio_kit`, whose
  `__init__.py` unconditionally pulls in the SAM-Audio chain requiring
  `dacvae` -- intentionally not installed in CI (no PyPI release; see
  "Fixed" above from the previous entry). Each file now starts with
  `pytest.importorskip("dacvae")`, so an environment without it gets a clean
  `SKIP` instead of a collection `ERROR`. Verified both ways: skips cleanly
  with `dacvae` absent, and all 15 tests still pass with it present.
- **Transitive dependency floors for `librosa`/`numba`/`llvmlite`.** With
  only `laion-clap>=1.1.0` declared, `pip`/`uv` could resolve to an ancient
  `librosa`/`numba`/`llvmlite` chain that builds from source and fails on
  Python 3.12+ (`numba`'s `setup.py` imports the removed
  `numpy.distutils`; old `llvmlite` hard-refuses Python>=3.10) -- meaning the
  classifiers' claimed 3.12 support did not actually install cleanly.  Added
  explicit floors (`librosa>=0.10.0`, `numba>=0.61.0`, `llvmlite>=0.44.0`)
  that steer the resolver to versions with prebuilt wheels for 3.10-3.13.
  sam-audio-kit doesn't call these packages directly; the floors exist only
  to fix installability. Verified with a clean-venv install (both `uv pip`
  and plain `pip`) on 3.10, 3.11, 3.12, and 3.13.

### Changed

- `.github/workflows/publish.yml`: added a `test` job that `build` now
  `needs:`, so a release build can no longer proceed without tests passing
  (org art.7). The final `Publish to PyPI` step is commented out with a
  rationale comment -- deliberate: the CI gate is live, but publishing
  itself stays disabled pending the license review above.

### Notes

- The differentiable-API feature noted as in-flight/uncommitted in earlier
  drafts of this entry has since been committed and merged into `main`
  ahead of this release.
