# Changelog

All notable changes to sam-audio-kit are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

**Release status: BLOCKED.** These changes prepare the safe, unambiguous
subset of a licensing/dependency cleanup. They do NOT themselves clear the
package for a PyPI release -- see `CLAUDE.md` ("Release status") for what
still requires a human decision.

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

### Changed

- `.github/workflows/publish.yml`: added a `test` job that `build` now
  `needs:`, so a release build can no longer proceed without tests passing
  (org art.7). The final `Publish to PyPI` step is commented out with a
  rationale comment -- deliberate: the CI gate is live, but publishing
  itself stays disabled pending the license review above.

### Notes

- The user has an **in-flight, uncommitted differentiable-API feature** on
  `main` (`src/sam_audio_kit/{__init__,inference,model}.py` and
  `src/sam_audio_kit/sam_audio/{__init__,model/__init__,model/base,model/model}.py`).
  This branch (`feat/adopt-constitution`) was created from a clean worktree
  and does not touch any of those files, specifically to avoid colliding
  with that WIP. Expect a merge/rebase step when both land.
