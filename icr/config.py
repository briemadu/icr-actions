#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reads all arguments from the command line and split them into configuration
objects for each part of the experiment (data, model, trained and logger).
"""

from argparse import ArgumentParser, Namespace


def args() -> Namespace:
    """Reads CLI arguments and returns them grouped into config objects."""
    parser = ArgumentParser(
        description='Experiments with iCR models in the CoDraw task.')

    # _________________________________ PATHS _________________________________
    parser.add_argument('-annotation_path',
                        default='../data/codraw-icr-v2.tsv',
                        type=str, help='Path to iCR annotation tsv file.')
    parser.add_argument('-codraw_path',
                        default='../data/data/CoDraw-master/dataset/CoDraw_1_0.json',
                        type=str, help='Path to CoDraw data JSON file.')
    parser.add_argument('-outputs_path',
                        default='./outputs/',
                        type=str, help='Path to dir where to log checkpoints.')
    parser.add_argument('-token_embeddings_path',
                        default='../data/text_embeddings/',
                        type=str, help='Path to dir of token embeddings.')
    parser.add_argument('-scenes_path',
                        default='../data/data/preprocessed/images/raw/',
                        type=str, help='Path to dir of scene embeddings.')

    # _________________________________ COMET _________________________________
    parser.add_argument('-comet_key', default='',
                        type=str, help='Comet.ml personal key.')
    parser.add_argument('-comet_project', default="cr-codraw-eacl24-manuscript",
                        type=str, help='Comet.ml project name.')
    parser.add_argument('-comet_workspace', default='',
                        type=str, help='Comet.ml workspace name.')
    parser.add_argument('-ignore_comet', action='store_true',
                        help='Do not log details to Comet_ml.')
    parser.add_argument('-comet_tag', default='',
                        type=str, help='A tag to identify experiment.')

    # _______________________________ DATA ____________________________________
    parser.add_argument('-context_size', default=3, type=int,
                        help='How many turns to append before instruction.')
    parser.add_argument('-dont_merge_persons', action='store_true',
                        help='Do not merge boys and girls as one clipart,\
                              make pose and face be attributes.')
    parser.add_argument('-langmodel', default='bert-base-uncased',
                        type=str, choices=['bert-base-uncased', 'roberta-base',
                                           'distilbert-base-uncased'],
                        help='Which utterance embeddings to use.')
    parser.add_argument('-only_icr_dialogues', action='store_true',
                        help='Use only dialogues containing at least one iCR.')
    parser.add_argument('-only_icr_turns', action='store_true',
                        help='Use only turns containing an iCR.')
    parser.add_argument('-score_threshold', default=0, type=float,
                        help='Use only games with final scores > threshold.')
    parser.add_argument('-reduce_turns_without_actions', action='store_true',
                        help='Remove almost all turns without any actions.')
    parser.add_argument('-dont_separate_actions', action='store_true',
                        help='Make additions/deletions not disjoint from \
                              other actions.')

    # ______________________________ EXPERIMENT _______________________________
    parser.add_argument('-batch_size', default=32, type=int,
                        help='Batch size.')
    parser.add_argument('-checkpoint', default='', type=str,
                        help='Path to model checkpoint to load.')
    parser.add_argument('-lr', default=0.0001, type=float,
                        help='Learning rate.')
    parser.add_argument('-pos_weight', default=2, type=float,
                        help='Weight for positive class in loss function.')
    parser.add_argument('-scheduler_step', default=2, type=int,
                        help='Update LR after n epochs.')
    parser.add_argument('-use_weighted_loss', action='store_true',
                        help='Use learnable weights in the sum of losses.')
    parser.add_argument('-use_scheduler', action='store_true',
                        help='Use LR step scheduler.')
    parser.add_argument('-weight_decay', default=0.0, type=float,
                        help='Weight decay for L2 regularisation.')

    # ________________________________ MODEL __________________________________
    parser.add_argument('-actions_for_icr', default='none',
                        type=str, choices=['none', 'gold', 'logits'],
                        help='Whether to use last actions as input to iCR \
                            prediction; if yes, as logits or real.')
    parser.add_argument('-d_model', default=256, type=int,
                        help='Size of clipart representation with embedding.')
    parser.add_argument('-dropout', default=0.1, type=float,
                        help='Droupout.')
    parser.add_argument('-full_trf_encoder', action='store_true',
                        help='Use a full Transformer (instead of decoder).')
    parser.add_argument('-hidden_dim', default=256, type=int,
                        help='Classifiers\'s hidden layer dimension.')
    parser.add_argument('-hidden_dim_trf', default=2048, type=int,
                        help='Classifiers\'s hidden layer dimension.')
    parser.add_argument('-nheads', default=16, type=int,
                        help='Heads in the Transformer for the scene state.')
    parser.add_argument('-nlayers', default=3, type=int,
                        help='Heads in the Transformer for the scene state.')
    parser.add_argument('-use_scene_before', action='store_true',
                        help='Use scene before actions in the input.')
    parser.add_argument('-use_scene_after', action='store_true',
                        help='Use scene after actions in the input.')
    parser.add_argument('-no_instruction', action='store_true',
                        help='Do not use the instruction in the input.')
    parser.add_argument('-dont_make_actions', action='store_true',
                        help='Do not learn to perform actions.')
    parser.add_argument('-predict_icrs_turn', action='store_true',
                        help='Learn to predict whether an iCR is to be made.')
    parser.add_argument('-predict_icrs_clipart', action='store_true',
                        help='Learn to predict whether each clipart is to be \
                             in the iCR.')
    parser.add_argument('-random_baseline', action='store_true',
                        help='Random baseline only with scene state.')
    parser.add_argument('-dont_preprocess_scenes', action='store_true',
                        help='Do notreprocess raw scenes before ResNet. \
                              Instead, only normalise it.')
    parser.add_argument('-unfreeze_resnet', action='store_true',
                        help='Fine-tune weights of pretrained ResNet.')

    # _____________________________ TRAINING __________________________________
    parser.add_argument('-n_grad_accumulate', default=1, type=int,
                        help='Steps for batch gradient accumulation.')
    parser.add_argument('-clip', default=1, type=float,
                        help='Clipping size, use 0 for no clipping.')
    parser.add_argument('-device', default='gpu', type=str,
                        choices=['cpu', 'gpu'], help='Which device to use.')
    parser.add_argument('-gpu', default=1, type=int,
                        choices=[0, 1], help='Which gpu to use.')
    parser.add_argument('-n_epochs', default=30, type=int,
                        help='Number of epochs to train the model.')
    parser.add_argument('-n_reload_data', default=1, type=int,
                        help='Reload data every n epochs.')
    parser.add_argument('-random_seed', default=1234, type=int,
                        help='Random seed set for reproducibility.')

    params = parser.parse_args()
    return params
