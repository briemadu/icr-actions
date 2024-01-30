#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of the neural network-based components used in the model for the
iCR experiments.
"""

from typing import Dict, List

import torch
from positional_encodings.torch_encodings import PositionalEncoding1D
from torch import nn, Tensor
from torchvision.models import resnet50, ResNet50_Weights

from icr import constants
from icr.structs.dataconf import SIZE_GALLERY


class LearnableLossWeights(nn.Module):
    """Implements weighted loss for multitask learning."""
    def __init__(self, weight_names: List[str]):
        super().__init__()
        weights = {f'pred-{name}': nn.Parameter(torch.tensor(1.),
                                                requires_grad=True)
                   for name in weight_names}
        self.weights = nn.ParameterDict(weights)

    def forward(self, name: str, loss: Tensor) -> Tensor:
        return self.weights[name] * loss


class CrossEncoder(nn.Module):
    """Implements an encoder with a memory."""
    def __init__(self, d_model: int, hidden_dim: int, nheads: int,
                 nlayers: int, dropout: float):
        super().__init__()
        # "misuse" a decoder (without a mask) for cross-attention
        trf_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nheads, batch_first=True,
            dim_feedforward=hidden_dim, dropout=dropout)
        self.encoder = nn.TransformerDecoder(trf_layer, num_layers=nlayers)

    def forward(self, target: Tensor, memory: Tensor) -> Tensor:
        return self.encoder(target, memory)


class SelfCrossEncoder(nn.Module):
    """Implements a Transformer as an encoder with a memory."""
    def __init__(self, d_model: int, hidden_dim: int, nheads: int,
                 nlayers: int, dropout: float):
        super().__init__()
        self.encoder = nn.Transformer(
            d_model=d_model, nhead=nheads, batch_first=True,
            dim_feedforward=hidden_dim, num_encoder_layers=nlayers,
            num_decoder_layers=nlayers, dropout=dropout)

    def forward(self, target: Tensor, source: Tensor) -> Tensor:
        return self.encoder(source, target)


class LearnablePosition(nn.Module):
    """Learns positional encoding for the scene feature embeddings.

    Originally from the DETR model https://arxiv.org/abs/2005.12872
    """
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.row_embed = nn.Parameter(torch.rand(50, d_model // 2))
        self.col_embed = nn.Parameter(torch.rand(50, d_model // 2))
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_scene: Tensor, height: int, width: int) -> Tensor:
        """Computes and adds positional embeddings to the input scene."""
        pos = torch.cat([
            self.col_embed[:width].unsqueeze(0).repeat(height, 1, 1),
            self.row_embed[:height].unsqueeze(1).repeat(1, width, 1)],
            dim=-1).flatten(0, 1).unsqueeze(1)
        return self.dropout(enc_scene + pos.permute(1, 0, 2))


class SceneEncoder(nn.Module):
    """Implements ResNet backbone for encoding scene, with CNN layer on top.

    Originally from the DETR model https://arxiv.org/abs/2005.12872
    Based on https://pytorch.org/vision/0.14/models.html#object-detection-instance-segmentation-and-person-keypoint-detection
    """
    def __init__(self, d_model: int, dropout: float, unfreeze: bool,
                 dont_preprocess_scenes: bool):
        super().__init__()
        self.preprocess_scenes = not dont_preprocess_scenes
        weights = ResNet50_Weights.DEFAULT
        if self.preprocess_scenes:
            self.img_preprocess = weights.transforms()
        model = list(resnet50(weights=weights).children())[:-2]
        self.backbone = nn.Sequential(*model)
        self.conv = nn.Conv2d(2048, d_model, 1)
        self.positions = LearnablePosition(d_model, dropout)
        if not unfreeze:
            self.freeze_params()

    def freeze_params(self) -> None:
        """Prevent fine-tuning of pretrained CV model."""
        for parameter in list(self.backbone.parameters()):
            parameter.requires_grad = False
        self.backbone.eval()

    def _preprocess(self, scene: Tensor) -> Tensor:
        if self.preprocess_scenes:
            return self.img_preprocess(scene)
        return scene.float() / constants.RGB_DIM

    def forward(self, scene: Tensor) -> Tensor:
        preproc_scene = self._preprocess(scene)
        features = self.conv(self.backbone(preproc_scene))
        height, width = features.shape[-2:]
        flattened = features.flatten(2).permute(0, 2, 1)
        return self.positions(flattened, height, width)


class ActionsMaker(nn.Module):
    """Decodes all actions on cliparts."""
    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()

        # one decoder for each action, each output a probability (as logit)
        self.decoder_presence = ProbDecoder(d_model, hidden_dim, 1, dropout)
        self.decoder_move = ProbDecoder(d_model, hidden_dim, 1, dropout)
        self.decoder_size = ProbDecoder(d_model, hidden_dim, 1, dropout)
        self.decoder_flip = ProbDecoder(d_model, hidden_dim, 1, dropout)
        # an extra generic "acted upon" class
        self.decoder_action = ProbDecoder(d_model, hidden_dim, 1, dropout)

        self.labels = constants.ACTION_LABELS

    def forward(self, game_state: Tensor) -> Dict[str, Tensor]:
        out_action = self.decoder_action(game_state)
        out_presence = self.decoder_presence(game_state)
        out_move = self.decoder_move(game_state)
        out_size = self.decoder_size(game_state)
        out_flip = self.decoder_flip(game_state)

        return {'pred-action': out_action.squeeze(2),
                'pred-action_presence': out_presence.squeeze(2),
                'pred-action_move': out_move.squeeze(2),
                'pred-action_size': out_size.squeeze(2),
                'pred-action_flip': out_flip.squeeze(2)}


class ProbDecoder(nn.Module):
    """A classifier for the probability of one event (as logit)."""
    def __init__(self, d_model: int, hidden_dim: int, output_dim: int,
                 dropout: float):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.decoder(inputs)


class iCRClipDecoder(nn.Module):
    """Predicts the probability of a clipart being subject to an iCR."""
    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.decoder_icr = ProbDecoder(d_model, hidden_dim, 1, dropout)
        self.labels = ['icr_clip_label']

    def forward(self, game_state: Tensor) -> Dict[str, Tensor]:
        out_icr = self.decoder_icr(game_state)
        return {'pred-icr_clip_label': out_icr.squeeze(2)}


class iCRTurnDecoder(nn.Module):
    """Predicts the probability of an iCR being made at a turn."""
    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.decoder_icr = ProbDecoder(d_model, hidden_dim, 1, dropout)
        self.labels = ['icr_label']

    def forward(self, game_state: Tensor) -> Tensor:
        out_icr = self.decoder_icr(game_state.mean(1))
        # alternative: flatten, but in a comparison the result was equivalent,
        # and this uses less parameters. If wish to flatten, decoder_icr
        # should be set to get SIZE_GALLERY * d_model as input dim
        # out_icr = self.decoder_icr(game_state.flatten(1))
        return {'pred-icr_label': out_icr.squeeze(1)}


class StateEmbedding(nn.Module):
    """Embeds the state of the gallery+scene."""
    def __init__(self, n_cliparts: int, n_positions: int, n_flips: int,
                 n_presences: int, n_sizes: int, n_faces: int, n_poses: int,
                 total_dim: int, prefix: str):
        super().__init__()

        self.prefix = prefix

        self.clip_embedder = nn.Embedding(n_cliparts, total_dim-100)
        self.position_embedder = nn.Linear(n_positions, 30)
        self.flip_embedder = nn.Embedding(n_flips, 10)
        self.presence_embedder = nn.Embedding(n_presences, 10)
        self.size_embedder = nn.Embedding(n_sizes, 10)
        self.face_embedder = nn.Embedding(n_faces, 20)
        self.pose_embedder = nn.Embedding(n_poses, 20)

        self.inputs = [
            'clip_id', f'{prefix}_face', f'{prefix}_pose',
            f'{prefix}_presence', f'{prefix}_size', f'{prefix}_flip',
            f'{prefix}_x_center', f'{prefix}_y_center',
            f'{prefix}_area', f'{prefix}_width', f'{prefix}_height']

    def _build_position_features(self, inputs: Tensor) -> Tensor:
        """Concatenate all features."""
        return torch.cat([
            inputs[f'{self.prefix}_x_center'].unsqueeze(2),
            inputs[f'{self.prefix}_y_center'].unsqueeze(2),
            inputs[f'{self.prefix}_area'].unsqueeze(2),
            inputs[f'{self.prefix}_width'].unsqueeze(2),
            inputs[f'{self.prefix}_height'].unsqueeze(2),
            ], dim=2).float()

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:

        clips = self.clip_embedder(inputs['clip_id'])
        face = self.face_embedder(inputs[f'{self.prefix}_face'])
        pose = self.pose_embedder(inputs[f'{self.prefix}_pose'])
        presence = self.presence_embedder(inputs[f'{self.prefix}_presence'])
        size = self.size_embedder(inputs[f'{self.prefix}_size'])
        flip = self.flip_embedder(inputs[f'{self.prefix}_flip'])

        position_features = self._build_position_features(inputs)
        position = self.position_embedder(position_features)

        gallery_state = torch.cat([clips, pose, face, presence, position,
                                   size, flip], dim=2)
        return gallery_state
