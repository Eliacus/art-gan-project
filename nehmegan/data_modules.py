from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from nehmegan import ROOT


class GANImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        dataset: str,
        image_size: tuple = (64, 64),
    ):
        """Generic datamodule for image datasets using the pytorchvision ImageFolder class

        Args:
            batch_size (int): Data batch size
            num_workers (int): Number of cpu workers
            image_size (tuple): Desired image size
        """
        super().__init__()
        self.dataset = dataset
        self.data_path = Path.joinpath(ROOT, "data", dataset, "images")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        self.normalization_schema = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(*self.normalization_schema),
            ]
        )

    def setup(self, stage: Optional[str] = None):
        self.ds = ImageFolder(self.data_path, self.transforms)

    def train_dataloader(self):
        return DataLoader(
            self.ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )