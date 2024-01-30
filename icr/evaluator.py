#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes to aid evaluation: saving outputs and computing multiple metrics.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import csv
import torch
from torch import nn, Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy, BinaryAUROC, BinaryAveragePrecision, BinaryConfusionMatrix,
    BinaryCohenKappa, BinaryF1Score, BinaryMatthewsCorrCoef, BinaryPrecision,
    BinaryRecall)

StrTDict = Dict[str, Tensor]


def create_binary_metrics(split: str, name: str) -> MetricCollection:
    """Return a collection of binary torchmetrics named by split and type."""
    return MetricCollection([
            BinaryAccuracy(),
            BinaryPrecision(),
            BinaryRecall(),
            BinaryAveragePrecision(),
            BinaryF1Score(),
            BinaryCohenKappa(),
            BinaryAUROC(),
            BinaryMatthewsCorrCoef(),
            BinaryConfusionMatrix()
        ], prefix=f'{split}_{name}_')


class Outputs:
    """Collects and saves the outputs of an experiment."""
    def __init__(self, outputs_path: str, names: List[str]):
        self.outputs_path = outputs_path
        self.names = names
        self.outputs = {split: {name: {} for name in names}
                        for split in ['val', 'test']}

    def update(self, outputs: StrTDict, refs: Tensor, split: str) -> None:
        """Add a batch of predictions to memory, referenced by identifiers."""
        for pred_name, predictions in outputs.items():
            name = pred_name.replace('pred-', '')
            for ref, preds in zip(refs, predictions):
                ref_id = ref.item()
                if not preds.shape:
                    self.outputs[split][name][ref_id] = {'-': preds.item()}
                else:
                    self.outputs[split][name][ref_id] = {}
                    for posit, pred in enumerate(preds):
                        self.outputs[split][name][ref_id][posit] = pred.item()

    def save(self, split: str, epoch: int, version: str) -> None:
        """Save all predictions to .csv files by type."""
        for name in self.names:
            directory = Path(f'{self.outputs_path}/{version}/')
            file = directory / f'predictions_{epoch}_{split}_{name}.csv'
            self._write_file(file, split, name)

    def _write_file(self, path: Path, split: str, name: str) -> None:
        """Write output file with one category of predictions."""
        with open(path, 'w') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(['identifier', 'position', 'prediction'])
            for ref, predictions in self.outputs[split][name].items():
                for position, pred in predictions.items():
                    writer.writerow([ref, position, pred])

    def reset(self, split: str) -> None:
        """Re-initialise memory for one split."""
        self.outputs[split] = {name: {} for name in self.names}


class Metrics(torch.nn.Module):
    """Handles all the evaluation metrics of an experiment."""
    def __init__(self, names: List[str], split: str):
        super().__init__()
        self.metrics = nn.ModuleDict({
            name: create_binary_metrics(split, name) for name in names})

    def forward(self) -> None:
        """This method is not necessary."""
        # This class has to inherit from torch.nn.Module just so
        # PyTorch Lightning recognises the metrics correctly.
        return None

    def update(self, outputs: StrTDict, gold: StrTDict) -> None:
        """Update all metrics with a new batch of predictions."""
        for pred_name, predictions in outputs.items():
            name = pred_name.replace('pred-', '')
            self.metrics[name].update(predictions, gold[name])

    def compute(self) -> Tuple[StrTDict, StrTDict]:
        """Return dicts with metrics and confusion matrices for all types."""
        results = {}
        for metric in self.metrics.values():
            results = {**results, **metric.compute()}

        metrics = {k: v for k, v in results.items() if 'Confusion' not in k}
        confmatrices = {k: v.cpu().numpy() for k, v in results.items()
                        if 'Confusion' in k}
        return metrics, confmatrices

    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()
