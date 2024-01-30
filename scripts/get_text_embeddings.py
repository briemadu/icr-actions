#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script retrieves text embeddings from pre-trained models for all CoDraw
turns. It is possible to pass the name of the model as argument and also one
of these options regarding which embeddings to return:

    - teller: only the teller's (instruction giver) utterances
    - drawer: only the drawer's (instruction follower) utterances
    - drawer-teller: the last drawer's utterance followed by the current
                     teller's utterance

We follow the documentation from:

    - BERT: https://huggingface.co/bert-base-
    - RoBERTa: https://huggingface.co/roberta-base
    - DistilBERT: https://huggingface.co/distilbert-base-uncased


Note: when we load BERT via BertModel.from_pretrained("bert-base-uncased"),
the following warning appears:

    > Some weights of the model checkpoint at bert-base-uncased were not used
    when initializing BertModel: ['cls.predictions.transform.LayerNorm.bias',
    'cls.predictions.transform.LayerNorm.weight',
    'cls.predictions.transform.dense.weight', 'cls.predictions.bias',
    'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight',
    'cls.seq_relationship.weight', 'cls.seq_relationship.bias']
    - This IS expected if you are initializing BertModel from the checkpoint
    of a model trained on another task or with another architecture
    (e.g. initializing a BertForSequenceClassification model from a
    BertForPreTraining model).
    - This IS NOT expected if you are initializing BertModel from the
    checkpoint of a model that you expect to be exactly identical
    (initializing a BertForSequenceClassification model from a
    BertForSequenceClassification model).

This is documented e.g. https://github.com/huggingface/transformers/issues/5421

We understand that these are layers that were in the checkpoint but are not
needed in BERT; it does not seem to be the case where some layers get
initialised randomly because they were not in the checkpoint. In other words,
these layers are not missing, they are ignored.

Sanity checks:
model = BertForPreTraining.from_pretrained("bert-base-uncased")
Loading like this throws no warning. But it has an extra layer and does not
return the poooled output (although the pooled layer is there,
before the cls layer).

model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
This, on the other hand, throws the same warning, but also another warning
saying that there are new layers that were randomly initialised.

--------

BERT:
    - Maximum length for teller: 46 tokens
    - Maximum length for drawer: 44 tokens
    - Maximum length for merged: 75 tokens
    - Embedding dimensions: 768

"""

import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import (BertTokenizer, BertModel,
                          RobertaTokenizer, RobertaModel,
                          DistilBertTokenizer, DistilBertModel)

SPLITS = ('train', 'val', 'test')
PREFIX_TELLER = '<TELLER>'
PREFIX_DRAWER = '<DRAWER>'
EMPTY_DRAWER = f'{PREFIX_DRAWER} --'

Model = transformers.models


def get_model_and_tokenizer(model_name: str,
                            device: str) -> Tuple[Model, Model, int]:
    """Load and return the pretrained model and tokenizer architectures."""

    if 'distilbert' in model_name:
        model = DistilBertModel.from_pretrained(model_name).to(device)
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        dims = 768
    elif 'roberta' in model_name:
        model = RobertaModel.from_pretrained(model_name).to(device)
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        dims = 768
    elif 'bert' in model_name:
        model = BertModel.from_pretrained(model_name).to(device)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        dims = 768
    else:
        raise Exception('Unknown model name; to use it, add the imports\
                         and adapt this function accordingly.')
    model.eval()
    return model, tokenizer, dims


def round_to_next_ten(number: int) -> int:
    """Round a number up to next group of ten."""
    return int((number / 10 + 1)) * 10


def build_message(turn: dict, player: str, last_msg_drawer: str = None) -> str:
    """Construct message string depending on player type."""
    if player == 'teller':
        message = f'{PREFIX_TELLER} {turn["msg_t"]}'
    elif player == 'drawer':
        message = f'{PREFIX_DRAWER} {turn["msg_d"]}'
    elif player == 'drawer-teller':
        assert last_msg_drawer is not None
        message = f'{last_msg_drawer} {PREFIX_TELLER} {turn["msg_t"]}'
    else:
        raise Exception('Invalid player type!')
    return message


def compute_max_tokens(data: dict, params: Namespace, tokenizer: Model) -> int:
    """Compute the maximum length variables.

    The maximum number of tokens is computed after adding prefixes and after
    tokenization. Merged means the last utterance by the drawer followed by 
    the current utterance by the teller. Return numbers rounded up to the next
    group of ten.
    """
    max_len = 0
    for game in tqdm(data.values(), desc='computing max lengths'):

        # first teller message has nothing before it, so we add an empty string
        last_msg_drawer = EMPTY_DRAWER

        for turn in game['dialog']:
            msg = build_message(turn, params.player, last_msg_drawer)
            tokens = tokenizer(msg, return_tensors='pt')
            if tokens['input_ids'].shape[1] > max_len:
                max_len = tokens['input_ids'].shape[1]

            last_msg_drawer = build_message(turn, 'drawer')

    print(f'\nMaximum length for {params.player}: {max_len} tokens.')
    print(f'Result is rounded to {round_to_next_ten(max_len)}.\n')

    return round_to_next_ten(max_len)


def save_embeddings(data: dict, model: Model, tokenizer: Model, dest_dir: Path,
                    split: str, max_tokens: int, params: Namespace,
                    device: str, dims: int) -> None:
    """Retrieve and save the embeddings for all games."""
    file = Path(f'{dest_dir}/{params.model}_{params.player}_{split}.hdf5')
    with torch.no_grad():
        with h5py.File(file, 'w') as outfile:
            for name, dialogue in tqdm(data.items(), desc=split):
                if split not in name:
                    continue
                game_id = str(int(name.split('_')[1]))
                seq = []
                last_msg_drawer = EMPTY_DRAWER
                for turn in dialogue['dialog']:
                    # create utterance and get the embeddings
                    msg = build_message(turn, params.player, last_msg_drawer)
                    tokens = tokenizer(msg, return_tensors='pt').to(device)
                    assert tokens['input_ids'].shape[1] <= max_tokens
                    output = model(**tokens)[0].cpu()
                    # pad the sequence length to the max_tokens size with 0s
                    dim = output.shape[1]
                    pad = torch.zeros([1, max_tokens - dim, output.shape[2]])
                    padded = torch.cat([output, pad], dim=1)
                    seq.append(padded[0])
                    # save drawer's message for the next turn, if needed
                    last_msg_drawer = build_message(turn, 'drawer')
                seq = np.array(seq)
                assert seq.shape == (len(dialogue['dialog']), max_tokens, dims)
                outfile.create_dataset(game_id, data=seq)


if __name__ == "__main__":

    parser = ArgumentParser(description='Collect pre-trained text embeddings.')
    parser.add_argument('-model',
                        default='bert-base-uncased',
                        type=str,
                        choices=['bert-base-uncased', 'roberta-base',
                                 'distilbert-base-uncased'],
                        help='Which pre-trained model to use')
    parser.add_argument('-player',
                        default='drawer-teller',
                        type=str,
                        choices=['drawer-teller', 'teller', 'drawer'],
                        help='Which player whose utterances are retrieved.')
    parser.add_argument('-codraw_path',
                        default='../data/data/CoDraw-master/dataset/CoDraw_1_0.json',
                        type=str,
                        help='Path to the CoDraw JSON file.')
    parser.add_argument('-output_dir',
                        default='../data/text_embeddings/',
                        type=str,
                        help='Path to save the embeddings.')
    params = parser.parse_args()

    # Create output directory if it does not exist yet
    dest_dir = Path(params.output_dir) / params.model
    try:
        os.mkdir(dest_dir)
    except FileExistsError:
        print('The output directory already exists, files may be overwritten!')

    # Read the data with the text utterances by both players
    with open(Path(params.codraw_path), 'r') as file:
        codraw = json.load(file)

    # Set up
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, tokenizer, dims = get_model_and_tokenizer(params.model, device)
    max_tokens = compute_max_tokens(codraw['data'], params, tokenizer)

    for split in SPLITS:
        save_embeddings(codraw['data'], model, tokenizer, dest_dir, split,
                        max_tokens, params, device, dims)
