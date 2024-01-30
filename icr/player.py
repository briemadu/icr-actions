#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of the neural network-based components used in the model for the
iCR experiments.
"""

from typing import Dict, List, Tuple

import torch
from positional_encodings.torch_encodings import PositionalEncoding1D
from torch import nn, Tensor

from icr import constants
from icr.aux import mask_pads
from icr.components import (
    ActionsMaker, CrossEncoder, iCRClipDecoder, iCRTurnDecoder,
    SceneEncoder, SelfCrossEncoder, StateEmbedding)


class CodrawPlayer(nn.Module):
    """Implements a CoDraw player; many variations are possible."""
    def __init__(self,
                 actions_for_icr: str,
                 d_model: int,
                 dont_make_actions: bool,
                 dont_preprocess_scenes: bool,
                 dropout: float,
                 full_trf_encoder: bool,
                 nheads: int,
                 nlayers: int,
                 hidden_dim: int,
                 hidden_dim_trf: int,
                 no_instruction: bool,
                 predict_icrs_turn: bool,
                 predict_icrs_clipart: bool,
                 random_baseline: bool,
                 use_scene_before: bool,
                 use_scene_after: bool,
                 unfreeze_resnet: bool):
        super().__init__()

        self.actions_for_icr = actions_for_icr
        self.make_actions = not dont_make_actions
        self.predict_icrs_clipart = predict_icrs_clipart
        self.predict_icrs_turn = predict_icrs_turn
        self.random_baseline = random_baseline
        self.use_instruction = not no_instruction
        self.use_scene_before = use_scene_before
        self.use_scene_after = use_scene_after

        # gallery is always part of the input
        self.gallery_embedder = StateEmbedding(
            n_cliparts=constants.N_CLIPS,
            n_faces=constants.N_FACE_CLASSES,
            n_flips=constants.N_FLIP_CLASSES,
            n_poses=constants.N_POSE_CLASSES,
            n_positions=constants.LEN_POSITION,
            n_presences=constants.N_PRESENCE_CLASSES,
            n_sizes=constants.N_SIZE_CLASSES,
            total_dim=d_model,
            prefix=constants.BEFORE_PREFIX)

        if not self.random_baseline:
            # positional encoding for the dialogue
            self.pe = PositionalEncoding1D(d_model)
            # which input components to use besides gallery
            if self.use_instruction:
                self.text_compresser = nn.Linear(768, d_model)
            if self.use_scene_before or self.use_scene_after:
                self.scene_encoder = SceneEncoder(
                    d_model=d_model,
                    dropout=dropout,
                    unfreeze=unfreeze_resnet,
                    dont_preprocess_scenes=dont_preprocess_scenes)
            # contextual encoder of the cliparts
            encoder = SelfCrossEncoder if full_trf_encoder else CrossEncoder
            self.encoder = encoder(
                d_model=d_model, hidden_dim=hidden_dim_trf,
                nheads=nheads, nlayers=nlayers, dropout=dropout)

        # what to predict: actions and/or iCRs on turn or clipart level
        if self.make_actions:
            self.actions_decoder = ActionsMaker(
                d_model=d_model, hidden_dim=hidden_dim, dropout=dropout)

        icr_dim = self._define_state_dim(d_model)
        if predict_icrs_clipart:
            self.icr_clip_decoder = iCRClipDecoder(
                d_model=icr_dim, hidden_dim=hidden_dim, dropout=dropout)
        elif predict_icrs_turn:
            self.icr_turn_decoder = iCRTurnDecoder(
                d_model=icr_dim, hidden_dim=hidden_dim, dropout=dropout)

        self.inputs, self.labels = self._define_inputs_and_golds()

    def _define_state_dim(self, d_model: int) -> int:
        """Return dimension for iCR prediction, larger if actions are used."""
        if self.actions_for_icr == 'none':
            return d_model
        return d_model + constants.N_ACTION_TYPES

    def _define_inputs_and_golds(self) -> Tuple[List, List]:
        """Create list of input and label names used in the model."""
        inputs = self.gallery_embedder.inputs[:]
        if self.use_instruction:
            inputs += ['dialogue']
        if self.use_scene_before:
            inputs += ['scene_before']
        if self.use_scene_after:
            inputs += ['scene_after']

        labels = []
        if self.make_actions:
            labels += self.actions_decoder.labels
        if self.predict_icrs_turn:
            labels += self.icr_turn_decoder.labels
        elif self.predict_icrs_clipart:
            labels += self.icr_clip_decoder.labels

        return inputs, labels

    def filter_batch(self, batch: Dict[str, Tensor]) -> Tuple[Dict, Dict]:
        """Extract from the batch dict only what the model uses."""
        label_list = self.labels + constants.ACTION_LABELS
        inputs = {k: v for k, v in batch.items() if k in self.inputs}
        labels = {k: v for k, v in batch.items() if k in label_list}
        return inputs, labels

    def _define_state(self, game_state: Tensor, logits: Dict[str, Tensor],
                      labels: Dict[str, Tensor]) -> Tensor:
        """Add actions or logits to input for iCR predictor, if applicable."""
        if self.actions_for_icr == 'none':
            return game_state

        actions = constants.ACTION_LABELS
        if self.actions_for_icr == 'logits':
            inputs = [logits[f'pred-{name}'].unsqueeze(2) for name in actions]
        else:
            inputs = [labels[name].unsqueeze(2) for name in actions]
        return torch.cat([game_state] + inputs, dim=2)

    def forward(self, inputs: Dict[str, Tensor],
                labels: Dict[str, Tensor]) -> Dict[str, Tensor]:

        # state of the gallery and symbolic scene before the actions
        gallery = self.gallery_embedder(inputs)

        # build the other sources of input, if applicable (scenes, dialogue)
        if self.random_baseline:
            game_state = gallery
        else:
            memory = []
            if self.use_scene_before:
                scene_before = self.scene_encoder(inputs['scene_before'])
                memory.append(scene_before)
            if self.use_scene_after:
                scene_after = self.scene_encoder(inputs['scene_after'])
                memory.append(scene_after)
            if memory:
                # add position encodings to elements in the image features
                # although it has its own learnt position encodings, we also
                # want to capture its position in the memory
                scenes = torch.cat(memory, dim=1)
                positions = self.pe(scenes)
                memory = [scenes + positions]

            if self.use_instruction:
                compressed = self.text_compresser(inputs['dialogue'])
                positions = mask_pads(compressed, self.pe(compressed))
                memory.append(compressed + positions)

            # join what's in the memory, possibly scenes and instruction
            memory = torch.cat(memory, dim=1)
            game_state = self.encoder(gallery, memory)

        action_outputs = {}
        if self.make_actions:
            action_outputs = self.actions_decoder(game_state)

        icr_outputs = {}
        game_state = self._define_state(game_state, action_outputs, labels)
        if self.predict_icrs_clipart:
            icr_outputs = self.icr_clip_decoder(game_state)
        elif self.predict_icrs_turn:
            icr_outputs = self.icr_turn_decoder(game_state)

        return {**action_outputs, **icr_outputs}
