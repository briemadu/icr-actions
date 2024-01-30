#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of the experiment with a PytorchLightining model.
"""

from argparse import Namespace
from typing import Dict, List

import pytorch_lightning as pl
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, Dataset

from icr.aux import filter_checkpoint
from icr.constants import REDUCTION, SPLITS
from icr.evaluator import Metrics, Outputs
from icr.components import LearnableLossWeights
from icr.player import CodrawPlayer


StrTDict = Dict[str, Tensor]


class iCRExperiment(pl.LightningModule):
    """Lightining experiment."""
    def __init__(self,
                 datasets: Dict[str, Dataset],
                 model_config: Namespace,
                 batch_size: int,
                 checkpoint: str,
                 lr: float,
                 outputs_path: str,
                 pos_weight: float,
                 scheduler_step: int,
                 use_weighted_loss: str,
                 use_scheduler: bool,
                 weight_decay: float):
        super().__init__()

        self.datasets = datasets
        self.batch_size = batch_size
        self.lr = lr
        self.scheduler_step = scheduler_step
        self.use_weighted_loss = use_weighted_loss
        self.use_scheduler = use_scheduler
        self.weight_decay = weight_decay

        self.model = CodrawPlayer(**vars(model_config))
        if checkpoint:
            pretrained_state = torch.load(checkpoint)['state_dict']
            weights = filter_checkpoint(pretrained_state)
            # using strict to allow loading only the parameters that match
            self.model.load_state_dict(weights, strict=False)
            print('\nLoaded model checkpoint!')

        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight),
                                         reduction=REDUCTION)
        if self.use_weighted_loss:
            self.loss_weights = LearnableLossWeights(self.model.labels)

        self.evaluator = self._define_metrics()
        self.outputs = Outputs(outputs_path, self.model.labels)

    def _define_metrics(self):
        metrics = {f'{split}-metrics': Metrics(self.model.labels, split)
                   for split in SPLITS}
        return nn.ModuleDict(metrics)

    def _compute_loss(self, outputs: StrTDict, gold: StrTDict,
                      split: str) -> Tensor:
        loss_sum = 0
        for name, preds in outputs.items():
            gold_labels = gold[name.replace('pred-', '')].float()
            loss = self.loss(preds, gold_labels)
            loss_name = f'{split}_{name}_loss'
            self.log(loss_name, loss, on_step=False, on_epoch=True)
            if self.use_weighted_loss:
                loss = self.loss_weights(name, loss)
            loss_sum += loss
        self.log(f'{split}_loss', loss_sum, on_step=False, on_epoch=True)
        return loss_sum

    def configure_optimizers(self) -> List:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay)
        if self.use_scheduler:
            lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=self.scheduler_step, gamma=0.9)
            return [optimizer], [lr_scheduler]
        return [optimizer]

    def forward(self, inputs: StrTDict, labels: StrTDict) -> StrTDict:
        # the labels are necessary for teacher forcing the actions
        return self.model(inputs, labels)

    def _step(self, batch: StrTDict, split: str) -> Tensor:
        inputs, labels = self.model.filter_batch(batch)

        outputs = self(inputs, labels)
        loss = self._compute_loss(outputs, labels, split)
        self.evaluator[f'{split}-metrics'].update(outputs, labels)

        if split != 'train' and not self.trainer.sanity_checking:
            probs = {name: torch.sigmoid(logits)
                     for name, logits in outputs.items()}
            self.outputs.update(probs, batch['identifier'], split)
        return loss

    def training_step(self, batch: StrTDict, batch_idx) -> Tensor:
        loss = self._step(batch, 'train')
        return loss

    def validation_step(self, batch: StrTDict, batch_idx) -> None:
        _ = self._step(batch, 'val')

    def test_step(self, batch: StrTDict, batch_idx) -> None:
        _ = self._step(batch, 'test')

    def _end_epoch(self, split: str) -> None:
        # compute, log and reset metrics
        metrics, confmatrices = self.evaluator[f'{split}-metrics'].compute()
        self.log_dict(metrics)
        for name, conf_matrix in confmatrices.items():
            self.logger.experiment.log_confusion_matrix(
                matrix=conf_matrix,
                epoch=self.current_epoch,
                file_name=name)

        self.evaluator[f'{split}-metrics'].reset()
        if split != 'train':
            # save and reset outputs
            self.outputs.save(split, self.current_epoch, self.logger.version)
            self.outputs.reset(split)

    def on_train_epoch_end(self) -> None:
        self._end_epoch('train')

    def on_validation_epoch_end(self) -> None:
        self._end_epoch('val')

    def on_test_epoch_end(self) -> None:
        self._end_epoch('test')

    def on_fit_end(self) -> None:
        path = self.trainer.checkpoint_callback.best_model_path
        self.logger.experiment.log_other('best_checkpoint', path)
        # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
        n_trainable = sum(p.numel() for p in self.model.parameters()
                          if p.requires_grad)
        n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.experiment.log_other('n_trainable_params', n_trainable)
        self.logger.experiment.log_other('n_params', n_params)

    def train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.datasets['val'], batch_size=self.batch_size,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size=self.batch_size,
                          shuffle=False)
