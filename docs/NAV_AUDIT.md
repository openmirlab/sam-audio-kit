# Navigation and inference-only audit

Audit date: 2026-07-15  
Branch: `adopt/sam-github-inference-cleanup`  
Scope: `src/sam_audio_kit`, packaging metadata, tests, and user-facing release docs.

## Inference-only gate

The shipped source was scanned for training, evaluation, dataset, experiment,
debug, Lightning, W&B, and TensorBoard surfaces. No training loop, dataset
loader, evaluation harness, experiment runner, or related dependency is
exposed by the package. The unused `core/probe.py` debug hook and the
evaluation-dataset entries in `types.py` were removed in this branch. Mentions
of training in model comments/docstrings describe how pretrained weights were
created; they are not executable training surfaces.

One optional inference feature, `ClapRanker`, uses the upstream LAION-CLAP
audio-feature helper located in that dependency's `laion_clap.training.data`
module. This is an external preprocessing import used only when the optional
text ranker is explicitly enabled; no LAION training code is vendored or
declared as an OpenMIRLab runtime module. It remains a documented non-blocking
warning for a future adapter extraction, because replacing it now would risk
changing ranking numerics without a golden fixture.

## Evidence

- `rg` scan of `src/` and `pyproject.toml` for training/evaluation/dataset/
  experiment tooling: only pretrained-model comments, inference-time
  `self.training` flags, and the external CLAP helper noted above remain.
- Baseline from `main` (`0935925`): `.venv/bin/python -m pytest -q` → 15
  passed. The post-cleanup run is also 15 passed, preserving the existing
  tested behavior.
- `python3 -m compileall -q src tests`: passed.
- `python3` import smoke is environment-blocked: optional runtime dependencies
  (`torch`, `torchaudio`, `transformers`, `xformers`, and friends) are not
  installed in the audit environment.
- `pytest`: environment-blocked because `pytest` is not installed.
- Package-data review: `LICENSE`, `LICENSE.SAM-AUDIO`, and `LICENSE.PE` remain
  included; no checkpoint or weight file is tracked by the repository.
- `git diff --check`: passed after the changes in this branch.
- `openmirlab-skills` was inspected on 2026-07-15; it has no `sam-audio-kit`
  task-to-package entry, so no public skills edit was warranted in this phase.

## Eight-principle nav audit

### ✓ Working

1. **Information hiding** — `model.py` and `download.py` are the narrow user
   entry points over model construction and Hugging Face cache details.
2. **Interface-first** — package exports are concentrated in `__init__.py`,
   while advanced model internals remain under `sam_audio/`.
3. **Explicit dependencies** — model/device/precision choices are passed into
   constructors and helpers; no global model manager was introduced.
4. **Right grain** — most domain modules have one clear role; synthesis,
   chunking, precision, and download are separate concerns.
5. **Framework fit** — the implementation uses ordinary PyTorch modules and
   `torch.inference_mode()` for inference paths.
6. **Rearrange, don't rewrite** — this phase only removed an unused no-op probe
   call and non-runtime evaluation metadata; model math was untouched.
7. **Confidence discipline** — blocked runtime tests are reported explicitly
   instead of being treated as passing.
8. **Agent navigability** — top-level entry points and domain names make the
   main inference path discoverable.

### ⚠ Non-blocking warnings

- Several extracted upstream model files exceed 500 lines (`core/vision_encoder/pe.py`,
  `core/transformer.py`, and synthesis modules). They are legitimate model or
  DSP implementations, so splitting them before golden fixtures would be a
  behavior-risking refactor; revisit during the structural phase.
- A number of extracted files lack standardized nav headers. Add headers in a
  dedicated `nav-sync` pass rather than mixing documentation-only edits with
  model behavior changes.
- `ClapRanker` retains the external LAION-CLAP preprocessing import described
  above. Extracting that helper is a phase-2 candidate once ranking fixtures
  are available.

### ❌ Errors

None found for the approved GitHub-only/inference-only cleanup phase.

## Phase decision

The inference-only and release-policy gates are passed for this branch. The
repository is ready for a separate phase-2 clean API, package-owned checkpoint
configuration, and lifecycle-session adoption, after a deterministic baseline
fixture is recorded. Those API changes are intentionally not included here.
