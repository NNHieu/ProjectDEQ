import logging
from typing import List, Optional

import hydra
import pandas as pd
from easy_to_hard_data import PrefixSumDataset
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
import torch
from src.utils import utils
from torch.utils.data import DataLoader

# A logger for this file


log = utils.get_logger(__name__)


def test_recur(config, model, trainer):
    for _, data_conf in config.data.type.items():
        dataset = PrefixSumDataset(config.data.root, num_bits=data_conf.nbit)
        evalloader = DataLoader(
            dataset,
            num_workers=0,
            batch_size=config.data.batch_size,
            shuffle=False,
            drop_last=False,
        )
        for niters in data_conf.iters:
            log.info(f"Evaluate on seq <{data_conf.nbit}>bit with <{niters}> iters")
            model.model.iters = niters
            trainer.test(model, evalloader)
            score = trainer.callback_metrics.get(config.get("optimized_metric"))


def test_fixedpoint(config, model, trainer):
    model.model.fixedpoint_layer.f_thres = config.model.arch.f_thres
    # model.model.fixedpoint_layer.f_thres = config.model.arch.b_thres
    results = []
    for nbit in config.data.type:
        dataset = PrefixSumDataset(config.data.root, num_bits=nbit)
        evalloader = DataLoader(
            dataset,
            num_workers=0,
            batch_size=config.data.batch_size,
            shuffle=False,
            drop_last=False,
        )
        log.info(f"Evaluate on seq <{nbit}>bit")
        score = trainer.test(model, evalloader)[0]
        score["nbit"] = nbit
        results.append(score)
    return results


def test(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)
    log.info(f"Loading weights from checkpoint <{config.checkpoint}>")
    ckpt = torch.load(to_absolute_path(config.get("checkpoint")))
    # print(ckpt.keys())
    model.load_state_dict(ckpt["state_dict"])

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

    if config.type == "fixedpoint":
        results = test_fixedpoint(config, model, trainer)
        log.info(results)
        results = pd.DataFrame(data=results)
        results = results.set_index("nbit")
        results = results.drop("train/jac_loss", axis=1)
        print(results)
    elif config.type == "recur":
        test_recur(config, model, trainer)

    # Init lightning datamodule
    # df = pd.DataFrame()
    # for filepath in glob.glob(f"{args.filepath}/*.json"):
    #     with open(filepath, 'r') as fp:
    #         data = json.load(fp)

    #     for d in data.values():
    #         if not isinstance(d, int):
    #             if not isinstance(d["test_iter"], int):
    #                 d["test_iter"] = 0

    #     num_entries = data.pop("num entries")
    #     little_df = pd.DataFrame.from_dict(data, orient="index")
    #     df = df.append(little_df)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=None,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )
