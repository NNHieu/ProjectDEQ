"""train.py
   Train, test, and save models
   Developed as part of Easy-To-Hard project
   April 2021
"""
import os
from typing import List, Optional

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from icecream import ic
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
from .utils import utils

from src.models import LitModel
# hydra.output_subdir = 'null'

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, logger=logger, callbacks=callbacks, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Get metric score for hyperparameter optimization
    score = trainer.callback_metrics.get(config.get("optimized_metric"))

    # Test the model
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    return score

def lossland(config: DictConfig):
    # from pathlib import Path
    import torch
    import logging

    from viztool.utils import name_surface_file, create_surfile
    from viztool.landscape import Surface, Dir2D, scheduler
    from viztool.sampler import Sampler
    from deq.standard.lib.solvers import anderson, forward_iteration

    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")

    ###############################################################################
    # Lossland code
    ###############################################################################
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    rank = 0

    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup()
    train_dl = datamodule.train_dataloader()

    log.info(f"Loading weights from checkpoint <{config.checkpoint}>")
    config.checkpoint = to_absolute_path(config.get("checkpoint"))
    model = LitModel.load_from_checkpoint(config.checkpoint)
    model.model.core.save_result=True
    model.model.core.f_solver = eval(config.model.arch.f_solver)
    model.model.core.f_thres = config.model.arch.f_thres
    model.model.core.f_eps = config.model.arch.f_eps

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, weights_summary=None, logger=None, callbacks=None, _convert_="partial", progress_bar_refresh_rate=0
    )

    def evaluate(model, dl):
        # trainer.test(model, dl)
        score = trainer.test(model, dl, verbose=False)[0]
        return score['test/loss'],

    # Create surface file if not exist
    # save_dir = Path(os.path.dirname(config.checkpoint), f'{os.path.basename(config.checkpoint)}_viz')
    # save_dir.mkdir(exist_ok=True)
    dir_file = 'dir.h5' #save_dir.joinpath('dir.h5')
    surf_file = name_surface_file(config.rect, config.resolution, 'surf')
    layers = ('loss',)
    ic(os.getcwd())
    try:
        create_surfile(model, layers, dir_file, surf_file, config.rect, config.resolution, log)
    except Exception as e:
        os.remove(dir_file)
        os.remove(surf_file)
        raise e

    # Load surface and prepair sampler
    model = model.to(device)
    surface = Surface.load(surf_file)
    log.info('cosine similarity between x-axis and y-axis: %f' % surface.dirs.similarity())
    sampler = Sampler(model, surface, layers, None, comm=None, rank=rank, logger=log)
    sampler.prepair()

    # Get the job
    job_schedule = scheduler.get_job_indices(*surface.get_unplotted_indices('loss'), rank, 1)

    # Exec
    if rank == 0: surface.open('r+')
    log.info('Computing %d values for rank %d'% (len(job_schedule[0]), rank))
    sampler.run(lambda model: evaluate(model, train_dl), job_schedule)
    surface.close()
