#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
An adapted version of main.py used for hyperparameter search.
"""

import warnings

import comet_ml
from comet_ml import Optimizer

import pytorch_lightning as pl
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
warnings.filterwarnings("ignore", ".*Converting it to torch.float32.*")

print('\n---------- Running iCR experiment ----------\n')

params = args()

### Optimizer adaptation

optim_config = {
    "algorithm": "bayes",
    "parameters": {
        #"langmodel": {"type": "categorical", "values": ['bert-base-uncased', 'roberta-base', 'distilbert-base-uncased']},
        "batch_size": {"type": "discrete", "values": [16, 32, 64, 128, 256]},
        "lr": {"type": "discrete", "values": [0.1, 0.01, 0.001, 0.0001, 0.003, 0.0003, 0.00001, 0.0005]},
        "pos_weight": {"type": "float", "scaling_type": "uniform", "min": 0.8, "max": 3},
        "scheduler_step": {"type": "integer", "min": 1, "max": 10},
        "weight_decay": {"type": "discrete", "values": [1, 0.1, 0.01, 0.001, 0.0001]},
        "random_seed": {"type": "integer", "min": 1, "max": 54321},
        "d_model": {"type": "discrete", "values": [128, 256, 512]},
        "hidden_dim_trf": {"type": "discrete", "values": [256, 512, 1024]},
        "hidden_dim": {"type": "discrete", "values": [32, 64, 128, 256, 512, 1024]},
        "dropout": {"type": "discrete", "values": [0.1, 0.2, 0.3]},
        "clip": {"type": "discrete", "values": [0, 0.25, 0.5, 1, 2.5, 5]},
        "n_grad_accumulate": {"type": "discrete", "values": [1, 2, 5, 10, 25]},
        "nheads": {"type": "discrete", "values": [1, 2, 4, 8, 32]},
        "nlayers": {"type": "integer", "min": 1, "max": 6},
        "n_reload_data": {"type": "integer", "min": 1, "max": 10},
        "context_size": {"type": "integer", "min": 1, "max": 5},

        "use_weighted_loss": {"type": "discrete", "values": [0, 1]},
        "use_scheduler": {"type": "discrete", "values": [0, 1]},
        #"dont_preprocess_scenes": {"type": "discrete", "values": [0, 1]},
        #"full_trf_encoder": {"type": "discrete", "values": [0, 1]},
    },
    "spec": {
        "metric": "val_icr_label_BinaryAveragePrecision",
        "objective": "maximize",
    },
}
opt = Optimizer(optim_config)

for experiment in opt.get_experiments(
        api_key=params.comet_key,
        workspace=params.comet_workspace,
        project_name=params.comet_project,
        auto_output_logging="simple",
        log_git_metadata=False,
        log_git_patch=False):
    
    # fix the type of model for search
    # we do search in the basic model that takes symbolic scene, instruction
    # and predict iCR; later we investigate the effect of other variations
    # e.g. use scenes, use action teacher forcing etc.
    params.actions_for_icr = 'none'
    params.predict_icrs_turn = True
    params.dont_make_actions = True

    # set parameters for search
    #params.langmodel = experiment.get_parameter('langmodel')
    params.batch_size = experiment.get_parameter('batch_size')
    params.lr = experiment.get_parameter('lr')
    params.pos_weight = experiment.get_parameter('pos_weight')
    params.scheduler_step = experiment.get_parameter('scheduler_step')
    params.weight_decay = experiment.get_parameter('weight_decay')
    params.random_seed = experiment.get_parameter('random_seed')
    params.d_model = experiment.get_parameter('d_model')
    params.hidden_dim_trf = experiment.get_parameter('hidden_dim_trf')
    params.hidden_dim = experiment.get_parameter('hidden_dim')
    params.dropout = experiment.get_parameter('dropout')
    params.clip = experiment.get_parameter('clip')
    params.n_grad_accumulate = experiment.get_parameter('n_grad_accumulate')
    params.nheads = experiment.get_parameter('nheads')
    params.nlayers = experiment.get_parameter('nlayers')
    params.n_reload_data = experiment.get_parameter('n_reload_data')
    params.context_size = experiment.get_parameter('context_size')

    params.use_weighted_loss = bool(experiment.get_parameter('use_weighted_loss'))
    params.use_scheduler = bool(experiment.get_parameter('use_scheduler'))
    #params.dont_preprocess_scenes = bool(experiment.get_parameter('dont_preprocess_scenes'))
    #params.full_trf_encoder = bool(experiment.get_parameter('full_trf_encoder'))

### end of adaptation

    aux.check_params_consistency(params)
    data_cfg, comet_cfg, trainer_cfg, model_cfg, exp_cfg = aux.split_config(params)

    pl.seed_everything(trainer_cfg.random_seed)
    #torch.use_deterministic_algorithms(True, warn_only=True)

    clipmap = aux.define_cliparts(data_cfg.dont_merge_persons)
    datasets = {split: CodrawData(split, clipmap, **vars(data_cfg))
                for split in SPLITS}

    monitored, mode = aux.define_monitored_metric(params)
    checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                          monitor=monitored,
                                          mode=mode,
                                          filename=monitored+'_{epoch}')
    lr_monitor = LearningRateMonitor(logging_interval='epoch', log_momentum=True)
    early_stopper = EarlyStopping(monitored, patience=5, mode=mode, min_delta=0.01)

    model = iCRExperiment(datasets, model_cfg, **vars(exp_cfg))

    print('\n')
    logger = CometLogger(
        api_key=comet_cfg.comet_key,
        workspace=comet_cfg.comet_workspace,
        save_dir=COMET_LOG_PATH,
        project_name=comet_cfg.comet_project,
        disabled=comet_cfg.ignore_comet,
        auto_metric_logging=False,
        ### add optim experiment
        experiment_key=experiment.get_key())

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

    best_model = checkpoint_callback.best_model_path
    aux.log_best_epoch(logger, log_dir, best_model)

    #trainer.test(model=model, ckpt_path=checkpoint_callback.best_model)
