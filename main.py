#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script that runs the complete training, validation and test.
"""

import warnings

import comet_ml
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint, LearningRateMonitor, EarlyStopping)

from icr import aux
from icr.constants import COMET_LOG_PATH, SPLITS
from icr.config import args
from icr.dataloader import CodrawData
from icr.plexperiment import iCRExperiment


# ignore Lightning warning about increasing the number of workers
warnings.filterwarnings("ignore", ".*does not have many workers.*")
# ignore warning for Matthews metric, it is undefined in the first epochs
warnings.filterwarnings("ignore", ".*Converting it to torch.float32.*")
# warning that some layers cannot be forced to be deterministic
warnings.filterwarnings("ignore", ".*but this operation is not deterministic because.*")

print('\n---------- Running iCR experiment ----------\n')

params = args()
aux.check_params_consistency(params)
data_cfg, comet_cfg, trainer_cfg, model_cfg, exp_cfg = aux.split_config(params)

pl.seed_everything(trainer_cfg.random_seed)
torch.use_deterministic_algorithms(True, warn_only=True)

clipmap = aux.define_cliparts(data_cfg.dont_merge_persons)
datasets = {split: CodrawData(split, clipmap, **vars(data_cfg))
            for split in SPLITS}

monitored, mode = aux.define_monitored_metric(params)
checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                      monitor=monitored,
                                      mode=mode,
                                      filename=monitored+'_{epoch}')
lr_monitor = LearningRateMonitor(logging_interval='epoch', log_momentum=True)
early_stopper = EarlyStopping(monitored, patience=8, mode=mode, min_delta=0.001)

model = iCRExperiment(datasets, model_cfg, **vars(exp_cfg))

print('\n')
logger = CometLogger(
    api_key=comet_cfg.comet_key,
    workspace=comet_cfg.comet_workspace,
    save_dir=COMET_LOG_PATH,
    project_name=comet_cfg.comet_project,
    disabled=comet_cfg.ignore_comet,
    auto_metric_logging=False)

log_dir = aux.log_all(logger, params, datasets, clipmap, monitored)

trainer = pl.Trainer(
    accelerator=trainer_cfg.device,
    devices=[trainer_cfg.gpu],
    max_epochs=trainer_cfg.n_epochs,
    num_sanity_val_steps=2,
    gradient_clip_val=trainer_cfg.clip if trainer_cfg.clip != 0 else None,
    accumulate_grad_batches=trainer_cfg.n_grad_accumulate,
    logger=logger,
    reload_dataloaders_every_n_epochs=trainer_cfg.n_reload_data,
    callbacks=[lr_monitor, checkpoint_callback, early_stopper])

trainer.fit(model=model)
trainer.test(model=model, ckpt_path=checkpoint_callback.best_model_path)

aux.log_final_state(logger, log_dir, checkpoint_callback.best_model_path)
