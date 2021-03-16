#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""

from models.slowfast.config.defaults import get_cfg
from models.slowfast.models import video_model_builder
from models.slowfast.utils import checkpoint

def load_config():
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    cfg_file = 'models/SLOWFAST_8x8_R50.yaml'
    if cfg_file is not None:
        cfg.merge_from_file(cfg_file)
    # # Load config from command line, overwrite config from opts.
    # if args.opts is not None:
    #     cfg.merge_from_list(args.opts)

    return cfg



def model_build():
    cfg = load_config()
    model = video_model_builder.SlowFastModel(cfg)
    # print(model)
    checkpoint.load_checkpoint('pre_trained/SLOWFAST_8x8_R50.pkl', model, False, convert_from_caffe2=True)
    print('loaded slowfast')
    return model
