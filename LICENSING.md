# Licensing map

sam-audio-kit is **not** a single-license package. `pyproject.toml` declares
the SPDX expression `LicenseRef-Meta-SAM-License AND MIT AND Apache-2.0` to
say so truthfully. This document maps each part of the tree to the license
that actually governs it, so nobody has to reverse-engineer that from source
comments.

| Component | Path | License | Source text |
|---|---|---|---|
| SAM-Audio model (transformer, vision/audio-visual encoders' PE integration glue, text encoder, ranking, processor, DACVAE wrapper) | `src/sam_audio_kit/sam_audio/**` | **Meta SAM License** (custom, source-available, **not OSI-approved**) | `LICENSE.SAM-AUDIO` |
| Perception Encoders (vendored vision/audio-visual encoder implementation) | `src/sam_audio_kit/core/vision_encoder/**`, `src/sam_audio_kit/core/audio_visual_encoder/**` | **Apache-2.0** | `LICENSE.PE` |
| OpenMIRLab wrapper: inference orchestration, lite mode, chunking, memory management, precision config, download/caching, CLI, types | `src/sam_audio_kit/model.py`, `inference.py`, `lite.py`, `chunking.py`, `memory.py`, `precision.py`, `download.py`, `cli.py`, `types.py`, `__init__.py` | **MIT** | `LICENSE` |
| Creative synthesis extras (granular synthesis, latent effects, DACVAE-based codec wrapper for creative use) | `src/sam_audio_kit/synth/**` | **MIT**, built *on top of* SAM-Audio's DACVAE codec (see below) | `LICENSE` |
| Latent granular synthesis technique (inspiration only, no vendored code) | `src/sam_audio_kit/synth/granular.py` | N/A — approach credited to Naotokui's public work; no third-party source is copied | `LICENSE` (attribution note) |

## Why this isn't just "MIT"

The wrapper code OpenMIRLab wrote is MIT. But `sam_audio/**` is a derivative
of Meta's SAM-Audio release and stays under Meta's **SAM License**, which is
source-available but imposes its own redistribution, use-restriction
(Trade Controls, ITAR, prohibited military/weapons use, etc.), and
termination terms — it is **not** OSI-approved and is **more restrictive**
than MIT. Slapping "MIT AND Apache-2.0" on the whole package (the previous
metadata) understated those restrictions. The corrected SPDX expression adds
`LicenseRef-Meta-SAM-License` (SPDX's mechanism for referencing a
non-standard license with no registered identifier) so the metadata matches
reality without weakening or reinterpreting any Meta term.

## External dependency not vendored: dacvae

`src/sam_audio_kit/sam_audio/model/codec.py` imports the external `dacvae`
package (github.com/facebookresearch/dacvae, Apache-2.0). It is **not
vendored** into this repo and **not declared** as a pip dependency (see
`pyproject.toml` and README.md "Installing the audio codec (dacvae)" for why
and how to install it manually). Its Apache-2.0 license is unaffected either
way; it simply isn't part of this repository's own SPDX expression, since it
isn't distributed with this package.

## What this document is not

This is a licensing *map*, not legal advice, and it does not itself decide
whether/how sam-audio-kit may be published or redistributed. See the
"Release status" section in `CLAUDE.md` for the current release-blocked
status pending a human decision on the SAM License question.
