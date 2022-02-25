import imp
import sys
from ctypes import ArgumentError
from turtle import backward
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from icecream import ic
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import (
    EVAL_DATALOADERS,
    STEP_OUTPUT,
    TRAIN_DATALOADERS,
)
# from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from src.modules import get_model

from deq.shared.stats import SolverStats

class LitModel(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        arch: Dict,
        # output_size: int = 10,
        optimizer_name: str = "adam",
        lr: float = 0.001,
        lr_decay: str = "step",
        lr_schedule: Tuple[int] = [100, 200, 300],
        lr_factor: float = 0.5,
        weight_decay: float = 0.0005,
        pretrain_steps: int = 100,
        warmup_period: int = 10,
        compute_jac_loss: bool = False,
        spectral_radius_mode: bool = False,
        init_std: float = 1.0,
        perturb_std: float = 0.,
        perturb_weight: float = 1.,
    ):
        super().__init__()
        # manually manage the optimization process
        # self.automatic_optimization=False
        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.hparams.lr_decay = self.hparams.lr_decay.lower()
        self.model = get_model(self.hparams.arch, init_std=self.hparams.init_std)
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.model.core.save_result = True

    def forward(
        self,
        x: torch.Tensor,
        deq_mode: bool = True,
        compute_jac_loss=False,
        spectral_radius_mode=False,
    ):
        return self.model(
            x,
            deq_mode=deq_mode,
            compute_jac_loss=compute_jac_loss,
            spectral_radius_mode=spectral_radius_mode,
        )[0]

    def get_z(self, batch_phix, **kwargs):
        return self.model.core(batch_phix, **kwargs)

    def get_phix(self, batch):
        return self.model.in_trans(batch)

    def get_output(self, batch_z):
        return self.model.out_trans(batch_z)

    def step(self, batch: Any, deq_mode: bool = True):
        x, y = batch
        logits = self.forward(
            x,
            deq_mode=deq_mode,
            compute_jac_loss=self.hparams.compute_jac_loss,
            spectral_radius_mode=self.hparams.spectral_radius_mode,
        )
        # self.log("train/jac_loss", jac_loss, on_step=False, on_epoch=True, prog_bar=True)
        logits = logits.squeeze()
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y, None

    def log_core_stats(self, phase):
        core_stats: SolverStats = self.model.core.stats
        self.log(f"{phase}/f_nstep", core_stats.fwd_iters.val, on_epoch=True)
        self.log(f"{phase}/f_lowest",core_stats.fwd_err.val, on_epoch=True,)
        self.log(f"{phase}/f_time",core_stats.fwd_time.val, on_epoch=True,)
        if phase == 'train':
            self.log(f"{phase}/b_nstep", core_stats.bkwd_iters.val, on_epoch=True)
            self.log(f"{phase}/b_lowest",core_stats.bkwd_err.val, on_epoch=True,)
            self.log(f"{phase}/b_time",core_stats.bkwd_time.val, on_epoch=True,)

    def manual_training_step(self, batch: Any, batch_idx: int):
        opt: torch.optim.Optimizer = self.optimizers()
        opt.zero_grad()
        x, y = batch
        z, _, _ = self.get_z(self.get_phix(x), deq_mode=True, compute_jac_loss=False, spectral_radius_mode=False)
        logits = self.get_output(z).squeeze()
        loss = self.criterion(logits, y)
        loss_value = loss.detach()
        preds = torch.argmax(logits, dim=1)

        self.manual_backward(loss)
        opt.step()

        if self.hparams.perturb_std > 0:
            opt.zero_grad()
            with torch.no_grad():
                perturbed_x = x + torch.normal(torch.zeros_like(x), self.hparams.perturb_std)
                # perturbed_x = x + 1e-1
                dx = perturbed_x - x
                norm_dx = torch.norm(dx, dim=-1)
                non_zeros = norm_dx > 0
                norm_dx = norm_dx[non_zeros]
                perturbed_x = perturbed_x[non_zeros]
            perturbed_phix = self.get_phix(perturbed_x)
            perturbed_z,_, _ = self.get_z(perturbed_phix, deq_mode=True, compute_jac_loss=False, spectral_radius_mode=False, backward=True)
            norm_dz = torch.norm(perturbed_z - z.detach()[non_zeros], dim=-1)
            perturb_loss = torch.sum(norm_dz/norm_dx)/norm_dz.shape[0]
            self.log("train/ploss", perturb_loss, on_step=True, on_epoch=False, prog_bar=True)
            perturb_loss = self.hparams.perturb_weight*perturb_loss
            self.manual_backward(perturb_loss)
            opt.step()

        # log train metrics
        acc = self.train_acc(preds, y)
        self.log("train/loss", loss_value, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss_value, "preds": preds, "targets": y}

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        z, _, _ = self.get_z(self.get_phix(x), deq_mode=True, compute_jac_loss=False, spectral_radius_mode=False)
        logits = self.get_output(z).squeeze()
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        # log train metrics
        acc = self.train_acc(preds, y)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": y}


    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, unused: Optional[int] = 0
    ) -> None:
        self.log_core_stats('train')
        return super().on_train_batch_end(outputs, batch, batch_idx, unused=unused)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, sradius = self.step(batch)
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log_core_stats('val')

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, sradius = self.step(batch)
        acc = self.test_acc(preds, targets)

        self.log_core_stats('test')
        self.log("test/loss", loss, on_epoch=True)
        self.log("test/acc", acc, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_epoch_end(self):
        # reset metrics at the end of every epoch!
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()
        self.model.core.stats.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer_name = self.hparams.optimizer_name.lower()
        lr = self.hparams.lr
        base_params = [p for n, p in self.named_parameters()]
        recur_params = []
        iters = 1

        # if "recur" in model:
        #     base_params = [p for n, p in net.named_parameters() if "recur" not in n]
        #     recur_params = [p for n, p in net.named_parameters() if "recur" in n]
        #     iters = net.iters
        # else:
        #     base_params = [p for n, p in net.named_parameters()]
        #     recur_params = []
        #     iters = 1

        all_params = [{"params": base_params}, {"params": recur_params, "lr": lr / iters}]

        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(all_params, lr=lr, weight_decay=2e-4, momentum=0.9)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(all_params, lr=lr, weight_decay=2e-4)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                all_params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False
            )
        elif optimizer_name == "adadelta":
            optimizer = torch.optim.Adadelta(all_params, lr=lr, rho=0.9, eps=1e-06, weight_decay=0)
        else:
            print(
                f"{ic.format()}: Optimizer choise of {optimizer_name} not yet implmented. Exiting."
            )
            sys.exit()

        # warmup_scheduler = ExponentialWarmup(optimizer, warmup_period=self.hparams.warmup_period)

        if self.hparams.lr_decay.lower() == "step":
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.hparams.lr_schedule,
                gamma=self.hparams.lr_factor,
                last_epoch=-1,
            )
        elif self.hparams.lr_decay.lower() == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.hparams.epochs, eta_min=0, last_epoch=-1, verbose=False
            )
        else:
            print(
                f"{ic.format()}: Learning rate decay style {self.hparams.lr_decay} not yet implemented."
                f"Exiting."
            )
            sys.exit()
        return ([optimizer], [lr_scheduler])
