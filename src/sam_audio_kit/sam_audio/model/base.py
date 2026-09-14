# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import json
import os
from contextlib import contextmanager
from typing import Callable, Dict, Optional, Union

import torch
from huggingface_hub import ModelHubMixin, snapshot_download


class BaseModel(torch.nn.Module, ModelHubMixin):
    config_cls: Callable

    def device(self):
        return next(self.parameters()).device

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        cache_dir: str = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Union[str, bool, None] = None,
        map_location: str = "cpu",
        strict: bool = True,
        revision: Optional[str] = None,
        skip: frozenset = frozenset(),
        **model_kwargs,
    ):
        """Build the model and load its checkpoint.

        `skip` names top-level components to leave unbuilt (see `SAMAudio.SKIPPABLE`);
        their checkpoint entries are dropped before `load_state_dict` so strict loading
        still holds for everything that was built.
        """
        if os.path.isdir(model_id):
            cached_model_dir = model_id
        else:
            # A caller-supplied `revision` (e.g. sam_audio_kit's own catalog
            # pin) wins; `cls.revision` is only the class-level fallback
            # (e.g. `SAMAudioJudgeModel.revision`, which itself now resolves
            # from the checkpoint catalog's `[models.judge].source_revision`
            # rather than a hardcoded ref -- see `judge.py`). Fixed here
            # 2026-09-14: this previously discarded the `revision` parameter
            # unconditionally, so passing one through `from_pretrained()` was
            # silently a no-op -- confirmed by reading this call directly.
            cached_model_dir = snapshot_download(
                repo_id=model_id,
                revision=revision if revision is not None else cls.revision,
                cache_dir=cache_dir,
                force_download=force_download,
                token=token,
                local_files_only=local_files_only,
            )
            # Verify the downloaded checkpoint's sha256 against the
            # package's own catalog, same as `download_model()`'s own
            # explicit pre-caching path -- only for classes that name a
            # `catalog_key` (see e.g. `SAMAudioJudgeModel`); a class with
            # none set (or a repo outside the small/base/large/judge
            # registry) is left unverified, matching `_verify_artifact`'s
            # own no-op behavior for untracked models.
            catalog_key = getattr(cls, "catalog_key", None)
            if catalog_key is not None:
                from ...download import _verify_artifact

                _verify_artifact(catalog_key, cached_model_dir, verbose=False)

        with open(os.path.join(cached_model_dir, "config.json")) as fin:
            config = json.load(fin)

        for key, value in model_kwargs.items():
            if key in config:
                config[key] = value

        config = cls.config_cls(**config)
        with no_random_init():
            model = cls(config, skip=skip) if skip else cls(config)
        # mmap: the file is paged in lazily, so tensors for skipped components are never
        # read at all, and nothing is copied twice (page cache -> copy into the module).
        state_dict = torch.load(
            os.path.join(cached_model_dir, "checkpoint.pt"),
            weights_only=True,
            map_location=map_location,
            mmap=True,
        )
        model.load_state_dict(drop_skipped(state_dict, skip), strict=strict)
        return model


_INIT_FNS = (
    "uniform_", "normal_", "trunc_normal_", "constant_", "ones_", "zeros_", "eye_",
    "dirac_", "xavier_uniform_", "xavier_normal_", "kaiming_uniform_", "kaiming_normal_",
    "orthogonal_", "sparse_",
)


@contextmanager
def no_random_init():
    """Make `torch.nn.init.*` no-ops while a model is built.

    Every parameter the constructor creates is overwritten by the checkpoint a moment
    later (strict loading guarantees it), so the random initialisation is pure waste:
    ~3 s of CPU on the 1.4 B-parameter base model. Sub-models that load their own
    weights (T5 through transformers, the CLAP ranker) do not go through these
    functions and are unaffected.
    """
    saved = {name: getattr(torch.nn.init, name) for name in _INIT_FNS}
    try:
        for name in _INIT_FNS:
            setattr(torch.nn.init, name, lambda tensor, *a, **k: tensor)
        yield
    finally:
        for name, fn in saved.items():
            setattr(torch.nn.init, name, fn)


def drop_skipped(state_dict: dict[str, torch.Tensor], skip) -> dict[str, torch.Tensor]:
    """Return `state_dict` without the entries belonging to the top-level modules in `skip`."""
    if not skip:
        return state_dict
    prefixes = tuple(f"{name}." for name in skip)
    return {k: v for k, v in state_dict.items() if not k.startswith(prefixes)}
