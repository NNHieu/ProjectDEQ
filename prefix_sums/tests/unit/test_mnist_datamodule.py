import os

import pytest
import torch
from src.datamodules.prefixsum_dm import PrefixSumDataModule


@pytest.mark.parametrize("nbit_train", [16, 32, 44])
@pytest.mark.parametrize("nbit_eval", [30, 36, 64])
@pytest.mark.parametrize("train_batch_size", [32, 128])
def test_mnist_datamodule(nbit_train, nbit_eval, train_batch_size):
    datamodule = PrefixSumDataModule(
        nbit_train=nbit_train, nbit_eval=nbit_eval, train_batch_size=train_batch_size
    )
    datamodule.prepare_data()

    assert not datamodule.data_train and not datamodule.data_val and not datamodule.data_test

    assert os.path.exists(os.path.join("data", "prefix_sums_data"))
    assert os.path.exists(os.path.join("data", "prefix_sums_data", "len_32"))

    datamodule.setup()

    assert datamodule.data_train and datamodule.data_val and datamodule.data_test
    # assert (
    #     len(datamodule.data_train) + len(datamodule.data_val) + len(datamodule.data_test) == 70_000
    # )

    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()
    assert datamodule.test_dataloader()

    batch = next(iter(datamodule.train_dataloader()))
    x, y = batch

    assert len(x) == train_batch_size
    assert len(y) == train_batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
