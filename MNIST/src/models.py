import imp
import sys
from ctypes import ArgumentError
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from icecream import ic
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import (
    EVAL_DATALOADERS,
    STEP_OUTPUT,
    TRAIN_DATALOADERS,
)
from torchmetrics import MaxMetric
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
        compute_jac_loss: bool = False,
        spectral_radius_mode: bool = False,
        init_std: float = 1.0,
        noise_sd: float = 0.,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.hparams.lr_decay = self.hparams.lr_decay.lower()
        self.model = get_model(self.hparams.arch, init_std=self.hparams.init_std)
        self.core = self.model.core
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # self.core.save_result = True

    def _forward(
        self,
        x: torch.Tensor,
        deq_mode: bool = True,
        compute_jac_loss=False,
        spectral_radius_mode=False,
    ):
        result, jac_loss, sradius = self.model(
            x,
            deq_mode=deq_mode,
            compute_jac_loss=compute_jac_loss,
            spectral_radius_mode=spectral_radius_mode,
        )
        out = [result]
        if compute_jac_loss: out.append(compute_jac_loss)
        if spectral_radius_mode: out.append(sradius)
        return out
    
    def forward(
        self,
        x: torch.Tensor,
        deq_mode: bool = True,
        compute_jac_loss=False,
        spectral_radius_mode=False,
    ):
        return self._forward(x, deq_mode, compute_jac_loss, spectral_radius_mode)[0]

    def step(self, batch: Any, deq_mode: bool = True):
        x, y = batch
        if self.hparams.noise_sd > 0:
            x = x + torch.randn_like(x, requires_grad=False).to(x.device) * self.hparams.noise_sd
        
        logits = self._forward(
            x,
            deq_mode=deq_mode,
            compute_jac_loss=self.hparams.compute_jac_loss,
            spectral_radius_mode=self.hparams.spectral_radius_mode,
        )[0]
        
        # self.log("train/jac_loss", jac_loss, on_step=False, on_epoch=True, prog_bar=True)
        logits = logits.squeeze()
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, unused: Optional[int] = 0
    ) -> None:
        core_stats: SolverStats = self.core.stats
        self.log(
            "train/f_nstep",
            core_stats.fwd_iters.val,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/f_lowest",
            core_stats.fwd_err.val,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train/b_nstep",
            core_stats.bkwd_iters.val,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train/b_lowest",
            core_stats.bkwd_err.val,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return super().on_train_batch_end(outputs, batch, batch_idx, unused=unused)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        core_stats: SolverStats = self.core.stats
        self.log(
            "val/f_nstep",
            core_stats.fwd_iters.val,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/f_lowest",
            core_stats.fwd_err.val,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        # self.log("val/sradius", sradius, on_step=False, on_epoch=True, prog_bar=True)
        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        core_stats: SolverStats = self.core.stats
        self.log(
            "test/f_nstep",
            core_stats.fwd_iters.val,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/f_lowest",
            core_stats.fwd_err.val,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_epoch_end(self):
        # reset metrics at the end of every epoch!
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

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
            optimizer = torch.optim.SGD(all_params, lr=lr, weight_decay=self.hparams.weight_decay, momentum=0.9)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(all_params, lr=lr, weight_decay=self.hparams.weight_decay)
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


def pred(model, solver, X, thres, stop_mode, eps, no_core=False, latest=False):
    model.eval()
    # target = torch.ones(X.shape[0])
    with torch.no_grad():
        X = X.to(device)
        phi_X = model.model.in_trans(X)
        if no_core:
            state = phi_X
            diff = 0
            nstep = 0
        else:
            state = torch.zeros_like(phi_X)
            func = lambda z: model.core.f(z, phi_X)
            result = solver(func, state, threshold=thres, stop_mode=stop_mode, name="forward", eps=eps)
            if latest:
                nstep = thres
                diff = result[f'{stop_mode}_trace'][-1]
                state = result['latest']
            else:
                nstep = result['nstep']
                diff = result['lowest']
                state = result['result']
        logits = model.model.out_trans(state)
        Z = torch.softmax(logits, dim=1).cpu().numpy()
    return Z, diff, nstep