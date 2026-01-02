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
        # Return None since lite mode will remove visual_ranker anyway
        return None
    elif isinstance(config, ClapRankerConfig):
        return ClapRanker(config)
    elif isinstance(config, JudgeRankerConfig):
        # Judge ranker not included in lite extraction
        # Return None since lite mode will remove it anyway
        return None
    elif isinstance(config, EnsembleRankerConfig):
        ranker_cfgs, weights = zip(*config.rankers.values(), strict=False)
        rankers = [create_ranker(cfg) for cfg in ranker_cfgs]
        # Filter out None rankers
        valid = [(r, w) for r, w in zip(rankers, weights) if r is not None]
        if not valid:
            return None
        rankers, weights = zip(*valid)
        return EnsembleRanker(rankers=list(rankers), weights=list(weights))
    else:
        assert config is None
        return None
