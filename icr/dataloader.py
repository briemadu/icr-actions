#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loads the CoDraw data according to the hyperparameters in the settings.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import csv
import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from icr.aux import (get_attributes, get_mentioned_cliparts, get_pose_face,
                     is_thing, parse_id, percent)
from icr.constants import (AFTER_PREFIX, BEFORE_PREFIX, CLIPSIZES,
                           EMPTY_SCENES, ICR_MAP, LABELS, NA_VALUE, OUT_HEADER,
                           RESCALING)
from icr.structs.dataconf import (HEIGHT, WIDTH, BALLS, CLOUDS, GLASSES, HATS,
                                  TREES, SIZE_GALLERY)
from icr.structs.game import Game


class CodrawData(Dataset):
    """Build the CoDraw datapoints for one split."""
    def __init__(self, split: str, clipmap: Dict[str, int],
                 annotation_path: str, codraw_path: str,
                 token_embeddings_path: str, scenes_path: str,
                 context_size: int, dont_merge_persons: bool,
                 dont_separate_actions: bool, langmodel: str,
                 only_icr_dialogues: bool, only_icr_turns: bool,
                 reduce_turns_without_actions: bool, score_threshold: float):

        self.clipmap = clipmap
        self.context_size = context_size
        self.dont_merge_persons = dont_merge_persons
        self.dont_separate_actions = dont_separate_actions
        self.label_id = ICR_MAP
        self.only_icr_dialogues = only_icr_dialogues
        self.only_icr_turns = only_icr_turns
        self.split = split
        self.reduce_turns_without_actions = reduce_turns_without_actions
        self.score_threshold = score_threshold

        codraw = self._load_codraw(codraw_path)
        self.icrs = self._load_icrs(annotation_path, codraw)
        self.games, self.datapoints, = self._construct(codraw)
        self.scenes = self._load_raw_scenes(scenes_path)
        self.instructions = self._load_texts(langmodel, token_embeddings_path)

        self.stats = self.compute_stats()

    def __len__(self) -> int:
        return len(self.datapoints)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        game_id, turn = self.datapoints[idx]
        cliplist = self.games[game_id].scenes.gallery

        instruction = self.get_instruction(game_id, turn)
        context = self.get_context(game_id, turn)
        scene_before, scene_after = self.get_scenes(game_id, turn)
        state_before, state_after = self.build_states(game_id, turn)
        actions = self.build_actions(state_before, state_after)
        icr_label = self.get_icr_turn_label(game_id, turn)
        icr_clip_label = self.get_icr_clipart_label(game_id, turn, cliplist)

        # append context to the last instruction
        dialogue = instruction
        if self.context_size > 0:
            dialogue = torch.cat([context, instruction], dim=0)

        data = {'dialogue': dialogue, 'game_id': game_id, 'identifier': idx,
                'icr_label': icr_label, 'icr_clip_label': icr_clip_label,
                'scene_after': scene_after, 'scene_before': scene_before,
                'turn': turn}

        return {**data, **state_before, **state_after, **actions}

    def _construct(self, codraw: Dict) -> Tuple[Dict, Dict]:
        """Build all datapoints according to specification."""
        print(f'\nconstruct {self.split} datapoints...')
        games: Dict[int, Game] = {}
        datapoints: Dict[int, Tuple[int, int]] = {}

        for game_id, data in tqdm(codraw.items(), desc=f'build {self.split}'):
            icr_turns = list(self.icrs[game_id].keys())
            if self.only_icr_dialogues and not icr_turns:
                # ignore dialogues without any iCR
                continue
            game = Game(game_id, data, icr_turns, quick_load=False)
            if game.final_score < self.score_threshold:
                # ignore games not played so well
                continue
            games[game_id] = game
            for turn in range(game.n_turns):
                if self.only_icr_turns and turn not in icr_turns:
                    # ignore turns without iCRs
                    continue
                if self._filter_actions(game, turn, icr_turns):
                    # decrease the numer of turns without actions
                    continue
                idx = len(datapoints)
                datapoints[idx] = (game_id, turn)
        return games, datapoints

    def _filter_actions(self, game: Game, turn: int,
                        icr_turns: List[int]) -> bool:
        """Reduce the proportion of turns without actions in training data."""
        if (self.reduce_turns_without_actions
                and game.actions.n_actions_per_turn()[turn] == 0
                and self.split == 'train'
                and turn not in icr_turns
                and random.random() > 0.1):
            return True
        return False

    def _load_icrs(self, path: str, codraw: Dict) -> Dict[int, Dict]:
        """Load annotation and return dictionary of iCR turns and cliparts."""
        annot = pd.read_csv(Path(path), sep='\t')
        icrs = {name: {} for name in codraw}
        for _, row in annot.iterrows():
            if self.split not in row.game_name:
                continue
            game_id = parse_id(row.game_name)
            mentioned = get_mentioned_cliparts(row)
            icrs[game_id][row.turn] = mentioned
        return icrs

    def _load_codraw(self, path: str) -> Dict[int, Dict]:
        """Read CoDraw JSON file and return dictionary with the split games."""
        with open(Path(path), 'r') as f:
            codraw = json.load(f)['data']
        # keep only the games in the current split; use only index as reference
        games = {parse_id(game_id): data for game_id, data in codraw.items()
                 if self.split in game_id}
        return games

    def _load_raw_scenes(self, scenes_path: str) -> Dict[int, np.array]:
        """Read and store scene files."""
        print(f'load {self.split} scenes...')

        fname = Path(scenes_path) / f'codraw_images_{self.split}.hdf5'
        with h5py.File(fname, 'r') as file:
            img_rgb = {int(key): scenes[:] for key, scenes in file.items()}
        return img_rgb

    def _load_texts(self, langmodel: str,
                    token_embeddings_path: str) -> Dict[int, np.array]:
        """Read and store the text embeddings for the instructions."""
        print(f'load {self.split} text embeddings...')

        file = f'{langmodel}_drawer-teller_{self.split}.hdf5'
        fname = Path(token_embeddings_path) / langmodel / file
        with h5py.File(fname, 'r') as file:
            embs = {int(key): value[:] for key, value in file.items()}
        return embs

    def get_instruction(self, game_id: int, turn: int) -> Tensor:
        """Return the instruction embedding at current turn."""
        return torch.tensor(self.instructions[game_id][turn])

    def get_context(self, game_id: int, turn: int) -> Optional[Tensor]:
        """Return the context embedding."""
        if self.context_size == 0:
            # no context needed
            return None
        if turn == 0:
            # build one initial tensor with zeros, will be padded below
            context = torch.zeros(self.instructions[game_id][0].shape)
        else:
            begin = max(0, turn - self.context_size)
            # do not include the turn itself, as it's the instruction
            context = torch.tensor(self.instructions[game_id][begin:turn])
        if len(context.shape) == 2:
            context = context.unsqueeze(0)

        n_turns, seq_len, emb_dim = context.shape
        pad_dim = self.context_size - n_turns
        if pad_dim > 0:
            # pad the context with zeros to the left, so that all context
            # tensors have the same number of dimensions
            pad = torch.zeros(pad_dim, seq_len, emb_dim)
            context = torch.cat([pad, context], dim=0)
        return torch.stack([token for turn in context for token in turn])

    def get_scenes(self, game_id: int, turn: int) -> Tuple[Tensor, Tensor]:
        """Return the scene before and after the instruction."""

        # return previous scene, unless it's the first turn, in which case we
        # always get an empty scene as the initial scene
        if turn > 0:
            scene_before = self.scenes[game_id][turn-1]
        else:
            aux_id = EMPTY_SCENES[self.split]
            scene_before = self.scenes[aux_id][0]
        scene_after = self.scenes[game_id][turn]
        return torch.tensor(scene_before), torch.tensor(scene_after)

    def get_icr_turn_label(self, game_id: int, turn: int) -> Tensor:
        """Return label representing whether an iCR occurred at this turn."""
        if turn in self.icrs[game_id]:
            return torch.tensor(self.label_id['icr'])
        return torch.tensor(self.label_id['not_icr'])

    def get_icr_clipart_label(self, game_id: int, turn: int,
                              clips: List[str]) -> Tensor:
        """Return labels for whether each clipart is mentioned in iCR."""
        gallery = [c if is_thing(c) else c.split()[0] for c in clips]
        mentions = []
        if turn in self.icrs[game_id]:
            mentions = self.icrs[game_id][turn][:]
            # NOTE: design decision: we consider that all objects belonging to
            # a meta-group are the subject of iCR at this turn
            # that e.g. all hats in the gallery are subject of an icr about
            # the hat group.
            # Besides, the 'ambiguous' group is being ignored. It occurs
            # around 300 times; for 232 times it's the only class. These cases
            # will not be considered, as we cannot know for sure what clipart
            # it is talking about.
            if 'hat group' in mentions:
                mentions += HATS
            if 'ball group' in mentions:
                mentions += BALLS
            if 'cloud group' in mentions:
                mentions += CLOUDS
            if 'glasses group' in mentions:
                mentions += GLASSES
            if 'tree group' in mentions:
                mentions += TREES
        return torch.tensor([1 if clip in mentions else 0 for clip in gallery])

    def _get_cliparts(self, game_id: int, turn: int) -> List[Any]:
        """Return a list of cliparts in the current scene."""
        # it will be a list of clipart names as strings, if the scene is empty,
        # or a list of clipart objects otherwise
        gallery = self.games[game_id].scenes.gallery
        if turn == -1:
            # build an empty state with the gallery, as the previous state
            # of the current turn 0
            cliparts = gallery
        else:
            cliparts = self.games[game_id].scenes.seq[turn].cliparts
            if not cliparts:
                # sometimes, scene strings are empty in the JSON file, so we
                # use the gallery again, considering the scene to be empty
                cliparts = gallery
            else:
                # ensure that the clipart order is always fixed
                assert [clipart.name for clipart in cliparts] == gallery
        return cliparts

    def build_states(self, game_id: int, turn: int) -> Tuple[Tensor, Tensor]:
        """Return dictionaries with the states before and after the turn."""
        old_cliparts = self._get_cliparts(game_id, turn-1)
        new_cliparts = self._get_cliparts(game_id, turn)

        old_state = [self._build_clipart_state(clip) for clip in old_cliparts]
        new_state = [self._build_clipart_state(clip) for clip in new_cliparts]

        old_tensors = self._create_state_tensors(old_state, BEFORE_PREFIX)
        new_tensors = self._create_state_tensors(new_state, AFTER_PREFIX)

        # ensure that the order is fixed, and keep only one
        assert torch.equal(old_tensors['clip_id'], new_tensors['clip_id'])
        del new_tensors['clip_id']
        return old_tensors, new_tensors

    def _build_clipart_state(self, clipart: Any) -> Dict[str, Any]:
        """Return dictionary with all clipart attributes at its state."""
        name = clipart.name if not isinstance(clipart, str) else clipart
        pose, face = get_pose_face(name)
        if not self.dont_merge_persons and ('boy' in name or 'girl' in name):
            name = name.split()[0]
        clip_id = self.clipmap[name]

        # NOTE: it would be possible to split these two cases,
        # or implement option to use the default values for cliparts that are
        # in the gallery and thus one category less
        # See documentation on reasons for this design decision.
        if isinstance(clipart, str) or not clipart.exists:
            # either scene was empty, so clipart is a str name from the gallery
            # or the clipart is not present in the scene
            presence = 0
            size, flip, area = 3 * [NA_VALUE]
            x_center, x_top, x_bottom, width = 4 * [NA_VALUE]
            y_center, y_top, y_bottom, height = 4 * [NA_VALUE]
        else:
            presence = 1
            # add one because 0 is NA_VALUE used for not present cliparts
            size = clipart.z + 1
            flip = clipart.flip + 1

            orig_width = CLIPSIZES[clipart.png]['width']
            orig_height = CLIPSIZES[clipart.png]['height']
            scale = RESCALING[clipart.z]
            # rescale all according to scene size, to normalise to [0, 1]
            # it's still possible to be above or below [0, 1] because sometimes
            # the center is outside the scene
            x_center = clipart.x / WIDTH
            x_top = (clipart.x + int(orig_width * scale) // 2) / WIDTH
            x_bottom = (clipart.x - int(orig_width * scale) // 2) / WIDTH
            width = orig_width / WIDTH
            y_center = clipart.y / HEIGHT
            y_top = (clipart.y + int(orig_height * scale) // 2) / HEIGHT
            y_bottom = (clipart.y - int(orig_height * scale) // 2) / HEIGHT
            height = orig_height / HEIGHT
            area = (orig_width*scale * orig_height*scale) / (WIDTH * HEIGHT)

        return {'clip_id': clip_id, 'pose': pose, 'face': face,
                'presence': presence, 'size': size, 'flip': flip,
                'x_center': x_center, 'x_top': x_top, 'x_bottom': x_bottom,
                'width': width, 'height': height, 'area': area,
                'y_center': y_center, 'y_top': y_top, 'y_bottom': y_bottom}

    @staticmethod
    def _create_state_tensors(cliplist: list, prefix: str) -> Dict[str, Any]:
        """Return a dictionary of tensors for each clipart attribute."""
        return {
            'clip_id': get_attributes(cliplist, 'clip_id'),
            f'{prefix}_presence': get_attributes(cliplist, 'presence'),
            f'{prefix}_x_center': get_attributes(cliplist, 'x_center'),
            f'{prefix}_x_top': get_attributes(cliplist, 'x_top'),
            f'{prefix}_x_bottom': get_attributes(cliplist, 'x_bottom'),
            f'{prefix}_width': get_attributes(cliplist, 'width'),
            f'{prefix}_y_center': get_attributes(cliplist, 'y_center'),
            f'{prefix}_y_top': get_attributes(cliplist, 'y_top'),
            f'{prefix}_y_bottom': get_attributes(cliplist, 'y_bottom'),
            f'{prefix}_height': get_attributes(cliplist, 'height'),
            f'{prefix}_area': get_attributes(cliplist, 'area'),
            f'{prefix}_size': get_attributes(cliplist, 'size'),
            f'{prefix}_flip': get_attributes(cliplist, 'flip'),
            f'{prefix}_pose': get_attributes(cliplist, 'pose'),
            f'{prefix}_face': get_attributes(cliplist, 'face'),
        }

    def build_actions(self, state_before: Dict, state_after: Dict) -> Dict:
        """Return dictionary with types of actions."""
        presence_before = state_before[f'{BEFORE_PREFIX}_presence']
        presence_after = state_after[f'{AFTER_PREFIX}_presence']
        presence_action = self._get_actions(presence_before, presence_after)
        mask = (presence_action == 0).int()
        if self.dont_separate_actions:
            # let added/removed cliparts also have position, flip, size actions
            mask = None

        flip_before = state_before[f'{BEFORE_PREFIX}_flip']
        flip_after = state_after[f'{AFTER_PREFIX}_flip']
        flip_action = self._get_actions(flip_before, flip_after, mask=mask)

        size_before = state_before[f'{BEFORE_PREFIX}_size']
        size_after = state_after[f'{AFTER_PREFIX}_size']
        size_action = self._get_actions(size_before, size_after, mask=mask)

        x_before = state_before[f'{BEFORE_PREFIX}_x_center'].unsqueeze(0)
        x_after = state_after[f'{AFTER_PREFIX}_x_center'].unsqueeze(0)
        y_before = state_before[f'{BEFORE_PREFIX}_y_center'].unsqueeze(0)
        y_after = state_after[f'{AFTER_PREFIX}_y_center'].unsqueeze(0)

        position_before = torch.cat([x_before, y_before], dim=0)
        position_after = torch.cat([x_after, y_after], dim=0)
        move_action = self._get_actions(position_before, position_after,
                                        mask=mask)

        any_action = (torch.cat([presence_action.unsqueeze(0),
                                 flip_action.unsqueeze(0),
                                 size_action.unsqueeze(0),
                                 move_action.unsqueeze(0)],
                                dim=0).sum(dim=0) > 0).int()

        return {'action': any_action,
                'action_presence': presence_action,
                'action_flip': flip_action,
                'action_size': size_action,
                'action_move': move_action}

    @staticmethod
    def _get_actions(before: Tensor, after: Tensor,
                     mask: Tensor = None) -> Tensor:
        """Return an array which is 1 if an action occurred."""
        if len(before.shape) == 1:
            actions = (before != after).int()
        else:
            actions = ((before != after).int().sum(dim=0) > 0).int()
        if mask is not None:
            # additions/deletions should not have other actions
            actions = torch.mul(actions, mask)
        return actions

    def compute_stats(self) -> Dict[str, float]:
        """Compute proportions of the labels in the dataset split."""
        # proportion of icrs, position actions, size actions, move actions,
        # flip actions, all actions, icrs on cliparts
        count_icr = 0.
        count_icr_clip = 0.
        count_presence_action = 0.
        count_flip_action = 0.
        count_size_action = 0.
        count_move_action = 0.
        count_action = 0.
        n_datapoints = 0.
        n_cliparts = 0.

        loader = DataLoader(self, batch_size=32)
        for batch in tqdm(loader, desc='compute stats'):
            n_datapoints += batch['game_id'].shape[0]
            n_cliparts += SIZE_GALLERY * batch['game_id'].shape[0]
            count_presence_action += batch['action_presence'].sum().item()
            count_flip_action += batch['action_flip'].sum().item()
            count_size_action += batch['action_size'].sum().item()
            count_move_action += batch['action_move'].sum().item()
            count_action += batch['action'].sum().item()
            count_icr += batch['icr_label'].sum().item()
            count_icr_clip += batch['icr_clip_label'].sum().item()

        return {
            f'{self.split}_icr_count': count_icr,
            f'{self.split}_icr_%': percent(count_icr, n_datapoints),
            f'{self.split}_icrclip_count': count_icr_clip,
            f'{self.split}_icrclip_%': percent(count_icr_clip, n_cliparts),
            f'{self.split}_presence_count': count_presence_action,
            f'{self.split}_presence_%': percent(count_presence_action,
                                                n_cliparts),
            f'{self.split}_flip_count': count_flip_action,
            f'{self.split}_flip_%': percent(count_flip_action, n_cliparts),
            f'{self.split}_size_count': count_size_action,
            f'{self.split}_size_%': percent(count_size_action, n_cliparts),
            f'{self.split}_move_count': count_move_action,
            f'{self.split}_move_%': percent(count_move_action, n_cliparts),
            f'{self.split}_action_count': count_action,
            f'{self.split}_action_%': percent(count_action, n_cliparts)}

    def save_structure(self, path: str) -> None:
        """Save JSON files with the created data and labels."""
        with open(Path(path) / f'{self.split}_datapoints.json', 'w') as file:
            json.dump(self.datapoints, file)

        file = Path(path) / f'{self.split}_gold-labels.csv'
        with open(file, 'w') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(OUT_HEADER)
            for idx, (game_id, turn) in tqdm(self.datapoints.items(),
                                             desc='save datapoints'):
                datapoint = self[idx]
                icr_label = datapoint['icr_label'].item()
                row = [idx, game_id, turn, '-', 'iCR', '-', icr_label]
                writer.writerow(row)
                for name in LABELS[:-1]:
                    clip_ids = datapoint['clip_id']
                    labels = datapoint[name]
                    for i, (clipid, label) in enumerate(zip(clip_ids, labels)):
                        row = [idx, game_id, turn, i, name, clipid.item(),
                               label.item()]
                        writer.writerow(row)
