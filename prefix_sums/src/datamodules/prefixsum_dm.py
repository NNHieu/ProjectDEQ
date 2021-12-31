from typing import Optional, Tuple

import torch
from easy_to_hard_data import PrefixSumDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms


class PrefixSumDataModule(LightningDataModule):
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
        nbit_train: int,
        nbit_eval: int,
        data_dir: str = "data/",
        train_val_split: float = 0.8,
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
        if self.hparams.test_batch_size < 0:
            self.hparams.test_batch_size = self.hparams.train_batch_size

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, int(self.hparams.nbit_train))

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        PrefixSumDataset(self.hparams.data_dir, num_bits=self.hparams.nbit_train, download=True)
        PrefixSumDataset(self.hparams.data_dir, num_bits=self.hparams.nbit_eval, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainvalset = PrefixSumDataset(self.hparams.data_dir, num_bits=self.hparams.nbit_train)
            num_train = int(self.hparams.train_val_split * len(trainvalset))
            # dataset = ConcatDataset(datasets=[trainset, testset])
            self.data_train, self.data_val = random_split(
                dataset=trainvalset,
                lengths=[num_train, int(len(trainvalset) - num_train)],
                generator=torch.Generator().manual_seed(42),
            )
            self.data_test = PrefixSumDataset(
                self.hparams.data_dir, num_bits=self.hparams.nbit_eval
            )

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
