# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# Extracted from facebook/sam-audio - See LICENSE.SAM-AUDIO

from ..model.config import (
    ClapRankerConfig,
    EnsembleRankerConfig,
    ImageBindRankerConfig,
    JudgeRankerConfig,
)
from .clap import ClapRanker
from .ranker import EnsembleRanker


def create_ranker(config):
    if isinstance(config, ImageBindRankerConfig):
        # ImageBind not included in lite extraction - requires imagebind dependency
        raise NotImplementedError(
            "ImageBindRanker not available in extracted version. "
            "Use ClapRanker (text_ranker) instead."
        )
    elif isinstance(config, ClapRankerConfig):
        return ClapRanker(config)
    elif isinstance(config, JudgeRankerConfig):
        # Judge ranker not included in lite extraction
        raise NotImplementedError(
            "JudgeRanker not available in extracted version. "
            "Use ClapRanker (text_ranker) instead."
        )
    elif isinstance(config, EnsembleRankerConfig):
        ranker_cfgs, weights = zip(*config.rankers.values(), strict=False)
        return EnsembleRanker(
            rankers=[create_ranker(cfg) for cfg in ranker_cfgs],
            weights=weights,
        )
    else:
        assert config is None
        return None
