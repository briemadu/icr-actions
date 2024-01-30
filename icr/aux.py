#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary functions used in various scripts.
"""

import json
import os
import re
import warnings
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from pytorch_lightning.loggers import Logger
from torch import Tensor
from torch.utils.data import Dataset

from icr import constants
from icr.structs.dataconf import file2obj


def filter_config(params: Namespace, keys: list) -> Namespace:
    """Filter the parameters by keys and return a Namespace with the subset."""
    subset = {key: value for key, value in vars(params).items() if key in keys}
    return Namespace(**subset)


def split_config(params: Namespace) -> List[Namespace]:
    """Split arguments into groups for each component of the experiment."""
    data_config = filter_config(params, constants.DATA_CONFIG)
    comet_config = filter_config(params, constants.COMET_CONFIG)
    train_config = filter_config(params, constants.TRAINER_CONFIG)
    model_config = filter_config(params, constants.MODEL_CONFIG)
    exp_config = filter_config(params, constants.EXPERIMENT_CONFIG)
    return data_config, comet_config, train_config, model_config, exp_config


def check_params_consistency(params):
    """Raises error if invalid combinations of hyperparameters is detected."""

    # either pure ablation only with state, or at least one source of input
    if not params.random_baseline:
        assert (not params.no_instruction
                or params.use_scene_after
                or params.use_scene_before), "At least one input is needed!"
    if params.random_baseline:
        assert params.actions_for_icr == 'none', "No input allowed for random base."
        assert params.no_instruction, "No input allowed for random base."
        assert not params.use_scene_after, "No input allowed for random base."
        assert not params.use_scene_before, "No input allowed for random base."

    # at least one prediction
    assert (not params.dont_make_actions
            or params.predict_icrs_clipart
            or params.predict_icrs_turn), "At least one prediction is needed!"

    assert not (params.predict_icrs_clipart
                and params.predict_icrs_turn), "Only one type of iCR prediction is possible!"

    if params.use_scene_after and not params.dont_make_actions:
        warn_msg = 'Actions will be *detected* from scenes, not *predicted*!'
        warnings.warn(warn_msg)

    # logits can only be used is actions are being predicted
    if params.actions_for_icr == 'logits':
        assert not params.dont_make_actions

    # set boundaries of the CoDraw game score
    assert 0 <= params.score_threshold <= 5

    if params.unfreeze_resnet or params.dont_preprocess_scenes:
        assert params.use_scene_before or params.use_scene_after


def log_all(logger: Logger, params: Namespace, datasets: Dict[str, Dataset],
            clipmap: Dict[str, int], monitored_metric: str) -> Path:
    """Log all hyperparameters and metadata to comet and to local folder."""

    # create directory where experiment results and metadata will be saved
    path_name = Path(f'{params.outputs_path}{logger.version}')
    os.mkdir(path_name)

    # log all to comet
    params_dic = {k: v for k, v in vars(params).items() if '_key' not in k}
    logger.experiment.log_parameters(params_dic)
    logger.experiment.log_code('main.py')
    logger.experiment.log_code(folder='icr/')
    logger.experiment.log_others(datasets['train'].stats)
    logger.experiment.log_others(datasets['val'].stats)
    logger.experiment.log_others(datasets['test'].stats)
    logger.experiment.log_other('monitored', monitored_metric)
    if params.comet_tag:
        logger.experiment.add_tag(params.comet_tag)

    # also log hyperparameter configuration and metadata to local
    for split in constants.SPLITS:
        with open(path_name / f'{split}_stats.json', 'w') as file:
            json.dump(datasets['train'].stats, file)

    with open(path_name / 'config.json', 'w') as file:
        json.dump(params_dic, file)

    with open(path_name / 'clipmap.json', 'w') as file:
        json.dump(clipmap, file)

    with open(path_name / 'meta.txt', 'w') as file:
        file.write(f'Comet name: {logger.experiment.get_name()}\n')
        file.write(f'Monitored: {monitored_metric}\n')
        file.write(f'Tag: {params.comet_tag}')

    datasets['val'].save_structure(path_name)
    datasets['test'].save_structure(path_name)
    print('\n')
    return path_name


def log_final_state(logger: Logger, log_dir: Path, ckpt_name: str) -> None:
    """Log the best epoch to the the local directory."""
    begin = re.search('epoch=', ckpt_name).span()[1]
    end = re.search('.ckpt', ckpt_name).span()[0]
    epoch = int(ckpt_name[begin: end])

    with open(log_dir / 'meta.txt', 'a') as file:
        file.write(f'Best epoch: {epoch}')
    logger.experiment.log_other('best_epoch', epoch)
    logger.experiment.log_asset_folder(log_dir)
    logger.experiment.log_model('best-model.ckpt', ckpt_name)


def get_mentioned_cliparts(row: pd.Series) -> List[str]:
    """Make list of mentioned cliparts in an iCR."""
    mentioned = [row.clipart_1, row.clipart_2, row.clipart_3, row.clipart_4,
                 row.clipart_5]
    return [clip.replace('_', ' ') for clip in mentioned if not pd.isna(clip)]


def parse_id(name: str) -> int:
    """Extract the id from the {train, val, test}_id game name as int."""
    return int(name.split('_')[1])


def get_pose_face(name: str) -> Tuple[int, int]:
    """Get an id for the pose and face, if applicable, or return a dummy id."""
    if 'boy' in name or 'girl' in name:
        _, face, pose = name.split()
        return [constants.POSES[pose], constants.FACES[face]]
    return constants.POSES['n/a'], constants.POSES['n/a']


def get_attributes(cliplist: Dict[str, Any], attribute: str) -> Tensor:
    """Return a tensor with the attribute value for all cliparts in scene."""
    return torch.tensor([clip[attribute] for clip in cliplist])


def define_cliparts(dont_merge_persons: bool) -> Dict[str, int]:
    """Create a mapping from cliparts to integers."""
    if dont_merge_persons:
        clipmap = {clip: i for i, clip in enumerate(file2obj.values())}
    else:
        # merge all boys into one and all girls into generic categories
        short_clips = [x for x in file2obj.values()
                       if 'boy' not in x and 'girl' not in x]
        short_clips += ['boy', 'girl']
        clipmap = {clip: i for i, clip in enumerate(short_clips)}
    return clipmap


def mask_pads(x: Tensor, pe: Tensor) -> Tensor:
    """Mask positional encoding to zero in pad tokens."""
    # turn all padding elements to zero, so that these tokens are completely
    # disregarded in the attention mechanism
    mask = (x != 0).all(dim=2).int().unsqueeze(2)
    full_mask = mask.expand(-1, -1, x.shape[2])
    return torch.mul(pe, full_mask)


def is_thing(clipart: str) -> bool:
    """Return True if clipart is a thing (i.e. not a person)."""
    return 'boy' not in clipart and 'girl' not in clipart


def define_monitored_metric(params: Namespace) -> Tuple[str, str]:
    """Return name of metric and mode (min/max) monitored for checkpointing."""
    if params.predict_icrs_turn:
        return 'val_icr_label_BinaryAveragePrecision', 'max'
    if params.predict_icrs_clipart:
        return 'val_icr_clip_label_BinaryAveragePrecision', 'max'
    if not params.dont_make_actions:
        return 'val_action_BinaryAveragePrecision', 'max'
    return 'val_loss', 'min'


def percent(numerator: float, denominator: float) -> float:
    """Return the percentage."""
    return 100 * numerator / denominator


def filter_checkpoint(state_dict: OrderedDict) -> Dict[str, Tensor]:
    """Remove the 'model' prefix from Lightning's model state dictionary."""
    return {k.replace('model.', ''): v for k, v in state_dict.items()
            if k.startswith('model.')}
