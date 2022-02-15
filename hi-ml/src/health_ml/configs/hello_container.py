#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from torchmetrics import MeanAbsoluteError
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from torch.utils.data import DataLoader, Dataset

from health_ml.lightning_container import LightningContainer


class HelloDataset(Dataset):
    """
    A simple 1dim regression task, read from a data file stored in the test data folder.
    """
    # Creating the data file:
    # import numpy as np
    # import torch
    #
    # N = 100
    # x = torch.rand((N, 1)) * 10
    # y = 0.2 * x + 0.1 * torch.randn(x.size())
    # xy = torch.cat((x, y), dim=1)
    # np.savetxt("health_ml/configs/hellocontainer.csv", xy.numpy(), delimiter=",")
    def __init__(self, raw_data: List[List[float]]) -> None:
        """
        Creates the 1-dim regression dataset.

        :param raw_data: The raw data. This must be numeric data which can be converted into a tensor.
            See the static method  from_path_and_indexes for an example call.
        """
        super().__init__()  # type: ignore
        self.data = torch.tensor(raw_data, dtype=torch.float)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        return {'x': self.data[item][0:1], 'y': self.data[item][1:2]}

    @staticmethod
    def from_path_and_indexes(
            root_folder: Path,
            start_index: int,
            end_index: int) -> 'HelloDataset':
        """
        Static method to instantiate a HelloDataset from the root folder with the start and end indexes.

        :param root_folder: The folder in which the data file lives ("hellocontainer.csv")
        :param start_index: The first row to read.
        :param end_index: The last row to read (exclusive)
        :return: A new instance based on the root folder and the start and end indexes.
        """
        raw_data = np.loadtxt(root_folder / "hellocontainer.csv", delimiter=",")[start_index:end_index]
        return HelloDataset(raw_data)


class HelloDataModule(LightningDataModule):
    """
    A data module that gives the training, validation and test data for a simple 1-dim regression task.
    """
    def __init__(
            self,
            root_folder: Path) -> None:
        super().__init__()
        self.train = HelloDataset.from_path_and_indexes(root_folder, start_index=0, end_index=50)
        self.val = HelloDataset.from_path_and_indexes(root_folder, start_index=50, end_index=70)
        self.test = HelloDataset.from_path_and_indexes(root_folder, start_index=70, end_index=100)

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(self.train, batch_size=5)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(self.val, batch_size=5)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(self.test, batch_size=5)


class HelloRegression(LightningModule):
    """
    A simple 1-dim regression model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Linear(in_features=1, out_features=1, bias=True)  # type: ignore
        self.test_mse: List[torch.Tensor] = []
        self.test_mae = MeanAbsoluteError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        This method is part of the standard PyTorch Lightning interface. For an introduction, please see
        https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html
        It runs a forward pass of a tensor through the model.

        :param x: The input tensor(s)
        :return: The model output.
        """
        return self.model(x)

    def training_step(self, batch: Dict[str, torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore
        """
        This method is part of the standard PyTorch Lightning interface. For an introduction, please see
        https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html
        It consumes a minibatch of training data (coming out of the data loader), does forward propagation, and
        computes the loss.

        :param batch: The batch of training data
        :return: The loss value with a computation graph attached.
        """
        loss = self.shared_step(batch)
        self.log("loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], *args: Any,  # type: ignore
                        **kwargs: Any) -> torch.Tensor:
        """
        This method is part of the standard PyTorch Lightning interface. For an introduction, please see
        https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html
        It consumes a minibatch of validation data (coming out of the data loader), does forward propagation, and
        computes the loss.

        :param batch: The batch of validation data
        :return: The loss value on the validation data.
        """
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        return loss

    def shared_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        This is a convenience method to reduce code duplication, because training, validation, and test step share
        large amounts of code.

        :param batch: The batch of data to process, with input data and targets.
        :return: The MSE loss that the model achieved on this batch.
        """
        input = batch["x"]
        target = batch["y"]
        prediction = self.forward(input)
        return torch.nn.functional.mse_loss(prediction, target)  # type: ignore

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        """
        This method is part of the standard PyTorch Lightning interface. For an introduction, please see
        https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html
        It returns the PyTorch optimizer(s) and learning rate scheduler(s) that should be used for training.
        """
        optimizer = Adam(self.parameters(), lr=1e-1)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
        return [optimizer], [scheduler]

    def on_test_epoch_start(self) -> None:
        """
        This method is part of the standard PyTorch Lightning interface. For an introduction, please see
        https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html
        In this method, you can prepare data structures that need to be in place before evaluating the model on the
        test set (that is done in the test_step).
        """
        self.test_mse = []
        self.test_mae.reset()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:  # type: ignore
        """
        This method is part of the standard PyTorch Lightning interface. For an introduction, please see
        https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html
        It evaluates the model in "inference mode" on data coming from the test set. It could, for example,
        also write each model prediction to disk.

        :param batch: The batch of test data.
        :param batch_idx: The index (0, 1, ...) of the batch when the data loader is enumerated.
        :return: The loss on the test data.
        """
        input = batch["x"]
        target = batch["y"]
        prediction = self.forward(input)
        # This illustrates two ways of computing metrics: Using standard torch
        loss = torch.nn.functional.mse_loss(prediction, target)  # type: ignore
        self.test_mse.append(loss)
        # Metrics computed using PyTorch Lightning objects. Note that these will, by default, attempt
        # to synchronize across GPUs.
        self.test_mae.update(preds=prediction, target=target)
        return loss

    def on_test_epoch_end(self) -> None:
        """
        This method is part of the standard PyTorch Lightning interface. For an introduction, please see
        https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html
        In this method, you can finish off anything to do with evaluating the model on the test set,
        for example writing aggregate metrics to disk.
        """
        average_mse = torch.mean(torch.stack(self.test_mse))
        Path("test_mse.txt").write_text(str(average_mse.item()))
        Path("test_mae.txt").write_text(str(self.test_mae.compute().item()))


class HelloContainer(LightningContainer):
    """
    An example container for using the hi-ml runner. This container has methods
    to generate the actual Lightning model, and read out the datamodule that will be used for training.
    The number of training epochs is controlled at container level.
    You can train this model by running `python health_ml/runner.py --model=HelloContainer` on the local box,
    or via `python health_ml/runner.py --model=HelloContainer --azureml=True` in AzureML
    """

    def __init__(self) -> None:
        super().__init__()
        self.local_dataset_dir = Path(__file__).parent
        self.max_epochs = 20

    # This method must be overridden by any subclass of LightningContainer. It returns the model that you wish to
    # train, as a LightningModule
    def create_model(self) -> LightningModule:
        return HelloRegression()

    # This method must be overridden by any subclass of LightningContainer. It returns a data module, which
    # in turn contains 3 data loaders for training, validation, and test set.
    def get_data_module(self) -> LightningDataModule:
        assert self.local_dataset_dir is not None
        return HelloDataModule(
            root_folder=self.local_dataset_dir)  # type: ignore