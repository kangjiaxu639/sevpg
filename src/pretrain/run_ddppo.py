#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import random
import os
from datetime import datetime

import numpy as np
import torch
import yaml

import habitat
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config_simple


# from detectron2
def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.
    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
        )
        seed = int(seed % 1e4)
    # np.random.seed(seed)
    # random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def main():
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    '''
    run_exp()


def run_exp() -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    run_type = "train"
    exp_config = "/home/ros/kjx/sevpg/configs/pretrain/crl_alp.yaml"
    opts = None
    config = get_config_simple(exp_config, opts)

    seed = config.ENV_CONFIG.SIMULATOR.SEED
    if seed == -1:
        seed = seed_all_rng()

    config.defrost()
    config.ENV_CONFIG.SIMULATOR.SEED = seed
    config.DATE = datetime.today().strftime('%Y-%m-%d')
    config.freeze()

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
