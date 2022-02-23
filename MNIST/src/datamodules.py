from typing import Optional, Tuple, List

import os
import torch
from easy_to_hard_data import PrefixSumDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import transforms
from torchvision.datasets import MNIST

_MNIST_MEAN = [0.1307,]
_MNIST_STDDEV = [0.3081,]

class MnistDM(LightningDataModule):
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
        self.dims = (1,28,28)

        self.data_train: Optional[MNIST] = None
        self.data_val: Optional[MNIST] = None
        self.data_test: Optional[MNIST] = None

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        # download
        MNIST(self.hparams.data_dir, train=True, download=True)
        MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # Assign train/val datasets for use in dataloaders
            if stage == "fit" or stage is None:
                self.data_train = MNIST(self.hparams.data_dir, train=True, transform=self.transform)
                self.data_val = MNIST(self.hparams.data_dir, train=False, transform=self.transform)

            # Assign test dataset for use in dataloader(s)
            if stage == "test" or stage is None:
                self.data_test = MNIST(self.hparams.data_dir, train=False, transform=self.transform)

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

class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.nn.Parameter(torch.tensor(means), requires_grad=False)
        self.sds = torch.nn.Parameter(torch.tensor(sds), requires_grad=False)

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds

def get_normalize_layer():
    return NormalizeLayer(_MNIST_MEAN, _MNIST_STDDEV)