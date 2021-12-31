"""train.py
   Train, test, and save models
   Developed as part of Easy-To-Hard project
   April 2021
"""
import argparse
import logging
import os
import sys
from collections import OrderedDict
from typing import List, Optional

import hydra
import numpy as np
import torch
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
from src.utils import utils
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.tensorboard import SummaryWriter

# import .warmup
# from utils import train, test, OptimizerWithSched, load_model_from_checkpoint, \
#     get_dataloaders, to_json, get_optimizer, to_log_file, now, get_model


# A logger for this file


# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# hydra.output_subdir = 'null'


def get_parser():
    parser = argparse.ArgumentParser(description="Deep Thinking")
    parser.add_argument(
        "--checkpoint", default="check_default", type=str, help="where to save the network"
    )
    parser.add_argument("--clip", default=1.0, help="max gradient magnitude for training")
    parser.add_argument("--data_path", default="../data", type=str, help="path to data files")
    parser.add_argument("--debug", action="store_true", help="debug?")
    parser.add_argument("--depth", default=8, type=int, help="depth of the network")
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs for training")
    parser.add_argument("--eval_data", default=20, type=int, help="what size eval data")
    parser.add_argument(
        "--json_name", default="test_stats", type=str, help="name of the json file"
    )
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--lr_decay", default="step", type=str, help="which kind of lr decay")
    parser.add_argument("--lr_factor", default=0.1, type=float, help="learning rate decay factor")
    parser.add_argument(
        "--lr_schedule", nargs="+", default=[100, 150], type=int, help="how often to decrease lr"
    )
    parser.add_argument("--model", default="recur_resnet", type=str, help="model for training")
    parser.add_argument("--model_path", default=None, type=str, help="where is the model saved?")
    parser.add_argument(
        "--no_shuffle", action="store_false", dest="shuffle", help="shuffle training data?"
    )
    parser.add_argument("--optimizer", default="sgd", type=str, help="optimizer")
    parser.add_argument("--output", default="output_default", type=str, help="output subdirectory")
    parser.add_argument("--save_json", action="store_true", help="save json")
    parser.add_argument("--save_period", default=None, type=int, help="how often to save")
    parser.add_argument("--test_batch_size", default=500, type=int, help="batch size for testing")
    parser.add_argument(
        "--test_iterations",
        default=None,
        type=int,
        help="how many, if testing with a different number iterations",
    )
    parser.add_argument("--test_mode", default="default", type=str, help="testing mode")
    parser.add_argument(
        "--train_batch_size", default=128, type=int, help="batch size for training"
    )
    parser.add_argument("--train_data", default=16, type=int, help="what size train data")
    parser.add_argument(
        "--train_log", default="train_log.txt", type=str, help="name of the log file"
    )
    parser.add_argument("--train_mode", default="xent", type=str, help="training mode")
    parser.add_argument(
        "--train_split", default=0.8, type=float, help="percentile of difficulty to train on"
    )
    parser.add_argument("--val_period", default=20, type=int, help="how often to validate")
    parser.add_argument("--warmup_period", default=5, type=int, help="warmup period")
    parser.add_argument("--width", default=4, type=int, help="width of the network")
    return parser


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


def preprocess_config(cfg: DictConfig):
    cfg.train.mode = cfg.train.mode.lower()
    if cfg.train.save_period is None:
        cfg.train.save_period = cfg.epochs
    if cfg.model_path is not None:
        cfg.model_path = to_absolute_path(cfg.model_path)
    cfg.checkpoint = to_absolute_path(cfg.checkpoint)
    # print(OmegaConf.to_yaml(cfg))
