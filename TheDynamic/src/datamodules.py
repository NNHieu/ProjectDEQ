from typing import Optional, Tuple

import os
import torch
from easy_to_hard_data import PrefixSumDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import transforms


class GMDataModule(LightningDataModule):
    """
    LightningDataModule.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        dim: int,
        rho: float,
        data_dir: str = "data/",
        train_batch_size: int = 100,
        test_batch_size: int = -1,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        if self.hparams.test_batch_size <= 0:
            self.hparams.test_batch_size = self.hparams.train_batch_size
        # self.dims is returned when you call datamodule.size()
        self.dims = (int(self.hparams.dim),)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 2

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            root = os.path.join(self.hparams.data_dir, str(self.hparams.dim))
            train_data = torch.load(
                os.path.join(root, f"train{self.hparams.dim}_{self.hparams.rho}.pth")
            )
            self.data_train = TensorDataset(
                train_data["X"], (train_data["target"].squeeze() + 1) // 2
            )

            eval_data = torch.load(
                os.path.join(root, f"eval{self.hparams.dim}_{self.hparams.rho}.pth")
            )
            self.data_val = TensorDataset(eval_data["X"], (eval_data["target"].squeeze() + 1) // 2)
            self.data_test = self.data_val

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=self.hparams.shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.test_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.test_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
