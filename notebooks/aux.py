#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary functions used in the evaluation scripts.
"""
import os
from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score

THRESHOLD = 0.5


def parse_metadata(line: str) -> Tuple[str, int]:
    """Extract and return the tag and the best epoch of an experiment."""
    _, name, best_epoch = line.split(': ')
    # the addition of the last line break was missing in the experiment script
    name = name.replace('Best epoch', '')
    return name, best_epoch


def define_model_type(config: dict) -> str:
    """Return the type of model based on the configuration."""
    if config['predict_icrs_turn'] and config['dont_make_actions']:
        return 'turn_overhearer'
    if config['predict_icrs_clipart'] and config['dont_make_actions']:
        return 'clip_overhearer'
    if config['predict_icrs_turn'] and not config['dont_make_actions']:
        return 'turn_action_taker'
    if config['predict_icrs_clipart'] and not config['dont_make_actions']:
        return 'clip_action_taker'
    assert (not config['predict_icrs_turn']
            and not config['predict_icrs_clipart'])
    return 'action_taker'


def get_test_epoch(folder: Path) -> int:
    """Extract the last epoch, used to enumerate the test run."""
    file = [x for x in os.listdir(folder) if 'predictions' in x and 'test' in x][0]
    return int(file.split('_')[1])


def classification_margin(preds):
    """Compute the absolute value of the margin between probs of negative and positive class."""
    # adapted from https://modal-python.readthedocs.io/en/latest/content/query_strategies/uncertainty_sampling.html
    return (preds - (1-preds)).abs()


def certainty_dist(preds):
    return (preds - 0.5).abs() # / 0.5 if normalised, becomes classification margin


def binary_cross_entropy(preds, targets):
    """Compute binary cross entropy for each pair in a sequence."""
    predicted = torch.tensor(preds.to_numpy())
    targets = torch.tensor(targets.to_numpy()).double()
    return torch.nn.functional.binary_cross_entropy(predicted, targets, reduction='none')


def stat(a, b):
    """Return absolute value of the difference between two numbers."""
    return np.abs(np.mean(a) - np.mean(b))


def define_experiment_name(config):
    """Define how to name an experiment for the main table."""
    name = ''

    # first, define the type of model
    if config['predict_icrs_turn']:
        name += 'when#'
    elif config['predict_icrs_clipart']:
        name += 'what#'
    else:
        if config['only_icr_turns']:
            name += 'no-filt#'
        else:
            name += 'no-full#'

    if config['checkpoint']:
        name += 'pretrained-'

    if config['dont_make_actions']:
        name += 'Overhearer#'
    else:
        if config['predict_icrs_turn'] or config['predict_icrs_clipart']:
            name += 'iCR-'
        if config['use_scene_before'] and config['use_scene_after']:
            name += 'Action-Detecter#'
        else:
            name += 'Action-Taker#'

    # then define the inputs
    # the gallery is always used as input
    name += 'G'
    if not config['no_instruction']:
        name += ', D'
    if config['use_scene_before']:
        name += ', $S_b$'
    if config['use_scene_after']:
        name += ', $S_a$'
    if config['actions_for_icr'] == 'gold':
        name += ', A'
    elif config['actions_for_icr'] == 'logits':
        name += ', $L_A$'

    return name


def read_predictions(folder: Path, epoch: int, split: str, label_name: str):
    path = folder / f'predictions_{epoch}_{split}_{label_name}.csv'
    return pd.read_csv(path)


def read_gold(folder: Path, split: str, name: str):
    path = folder / f'{split}_gold-labels.csv'
    gold = pd.read_csv(path)
    return gold[gold.name == name].copy()


def compute_main_metrics(labels, scores):
    """Return AP, binary F1 and weighted-average F1 scores."""
    ap = average_precision_score(labels, scores)
    bf1 = f1_score(labels, (scores > THRESHOLD).astype(int), average='binary')
    mf1 = f1_score(labels, (scores > THRESHOLD).astype(int), average='macro')
    return ap, bf1, mf1


def merge_gold_preds(df, gold_df, icr_clip_labels):
    """Merge predictions, gold standard and icr clipart labels."""
    index = ['identifier', 'position']
    gold_df['position'] = pd.to_numeric(gold_df['position'])
    icr_clip_labels['position'] = pd.to_numeric(icr_clip_labels['position'])
    return pd.concat([gold_df.set_index(index), df.set_index(index), icr_clip_labels.set_index(index)], axis=1)


def read_action_predictions(folder, epoch, split):
    """Return a large table with predictions and metadata for all actions."""

    icr_clip_labels = (read_gold(folder, split,  'icr_clip_label')
                       .rename(columns={'label': 'icr_clip_label'})
                       .drop(['game_id', 'turn', 'name', 'clipart'], axis=1))
    
    icr_turn_labels = (read_gold(folder, split,  'iCR')
                       .rename(columns={'label': 'icr_turn_label'})
                       .drop(['position', 'clipart'], axis=1))


    acted_df = read_predictions(folder, epoch, split, 'action')
    gold_acted = read_gold(folder, split,  'action')
    acted = merge_gold_preds(acted_df, gold_acted, icr_clip_labels)
    
    size_df = read_predictions(folder, epoch, split, 'action_size')
    gold_size = read_gold(folder, split,  'action_size')
    size = merge_gold_preds(size_df, gold_size, icr_clip_labels)

    presence_df = read_predictions(folder, epoch, split, 'action_presence')
    gold_presence = read_gold(folder, split,  'action_presence')
    presence = merge_gold_preds(presence_df, gold_presence, icr_clip_labels)
    
    flip_df = read_predictions(folder, epoch, split, 'action_flip')
    gold_flip = read_gold(folder, split,  'action_flip')
    flip = merge_gold_preds(flip_df, gold_flip, icr_clip_labels)

    move_df = read_predictions(folder, epoch, split, 'action_move')
    gold_move = read_gold(folder, split,  'action_move')
    move = merge_gold_preds(move_df, gold_move, icr_clip_labels)

    return pd.concat([acted, size, presence, flip, move]), icr_turn_labels


def define_sorter(keyword):

    if keyword == 'when':
        lower_portion = [
            'no-full#Action-Taker#G',
            'no-full#Action-Taker#G, D',
            'no-full#Action-Taker#G, D, $S_b$',

            f'{keyword}#iCR-Action-Taker#G, D',
            f'{keyword}#iCR-Action-Taker#G, D, $L_A$',
            f'{keyword}#iCR-Action-Taker#G, D, $S_b$',
            f'{keyword}#iCR-Action-Taker#G, D, $S_b$, $L_A$',

            f'{keyword}#iCR-Action-Detecter#G, D, $S_b$, $S_a$',
            f'{keyword}#iCR-Action-Detecter#G, D, $S_b$, $S_a$, $L_A$',
        ]
    elif keyword == 'what':
        lower_portion = [
            'no-filt#Action-Taker#G',
            'no-filt#Action-Taker#G, D',
            'no-filt#Action-Taker#G, D, $S_b$',

            f'{keyword}#pretrained-iCR-Action-Taker#G, D',
            f'{keyword}#pretrained-iCR-Action-Taker#G, D, $L_A$',
            f'{keyword}#pretrained-iCR-Action-Taker#G, D, $S_b$',
            f'{keyword}#pretrained-iCR-Action-Taker#G, D, $S_b$, $L_A$',

            f'{keyword}#pretrained-iCR-Action-Detecter#G, D, $S_b$, $S_a$',
            f'{keyword}#pretrained-iCR-Action-Detecter#G, D, $S_b$, $S_a$, $L_A$',
        ]
    return [
        f'{keyword}#Overhearer#G',
        f'{keyword}#Overhearer#G, D',
        f'{keyword}#Overhearer#G, D, $S_b$',
        f'{keyword}#Overhearer#G, D, $S_b$, $S_a$',
        f'{keyword}#Overhearer#G, D, A',
        f'{keyword}#Overhearer#G, D, $S_b$, A',
        f'{keyword}#Overhearer#G, D, $S_b$, $S_a$, A'] + lower_portion
