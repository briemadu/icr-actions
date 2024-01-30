#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of several constants, lists and dictionaries use in the experiments.
"""

import json


BEFORE_PREFIX = 'before'

AFTER_PREFIX = 'after'

REDUCTION = 'sum'

NA_VALUE = 0

COMET_LOG_PATH = 'comet-logs/'

SPLITS = ('train', 'val', 'test')

ICR_MAP = {'not_icr': 0, 'icr': 1}

# n/a has to be the same as NA_VALUE
FACES = {'n/a': 0, 'angry': 1, 'wide_smile': 2, 'smile': 3,
         'sad': 4, 'scared': 5}

POSES = {'n/a': 0, 'arms_right': 1, 'arms_up': 2, 'kicking': 3, 'running': 4,
         'leg_crossed': 5, 'sit': 6, 'wave': 7}

RESCALING = {0: 1.0, 1: 0.7, 2: 0.49}

EMPTY_SCENES = {'train': 0, 'val': 8, 'test': 9}

with open('../data/clipsizes.json', 'r') as file:
    CLIPSIZES = json.load(file)

N_POSE_CLASSES = len(POSES)

N_FACE_CLASSES = len(FACES)

N_FLIP_CLASSES = 3

N_SIZE_CLASSES = 4

N_PRESENCE_CLASSES = 2

LEN_POSITION = 5

N_CLIPS = 59

RGB_DIM = 255

ACTION_LABELS = ['action', 'action_presence', 'action_move',
                 'action_size', 'action_flip']

N_ACTION_TYPES = len(ACTION_LABELS)

DATA_CONFIG = ['annotation_path', 'codraw_path', 'token_embeddings_path',
               'scenes_path', 'context_size', 'dont_merge_persons',
               'dont_separate_actions', 'langmodel', 'only_icr_dialogues',
               'only_icr_turns', 'reduce_turns_without_actions',
               'score_threshold']

MODEL_CONFIG = ['actions_for_icr', 'd_model', 'dont_make_actions',
                'dont_preprocess_scenes', 'dropout', 'full_trf_encoder',
                'hidden_dim', 'hidden_dim_trf', 'nheads', 'nlayers',
                'no_instruction', 'predict_icrs_turn', 'predict_icrs_clipart',
                'random_baseline', 'unfreeze_resnet', 'use_scene_after',
                'use_scene_before']

EXPERIMENT_CONFIG = ['batch_size', 'checkpoint', 'lr', 'pos_weight',
                     'outputs_path', 'scheduler_step', 'use_weighted_loss',
                     'use_scheduler', 'weight_decay']

TRAINER_CONFIG = ['batch_size', 'clip', 'device', 'gpu', 'n_epochs',
                  'n_grad_accumulate','n_reload_data', 'random_seed']

COMET_CONFIG = ['comet_key', 'comet_project', 'comet_tag',
                'comet_workspace', 'ignore_comet']

LABELS = ['action_presence', 'action_move', 'action_flip', 'action_size',
          'action', 'icr_clip_label', 'icr_label']

OUT_HEADER = ['identifier', 'game_id', 'turn', 'position', 'name', 'clipart',
              'label']
