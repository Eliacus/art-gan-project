from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src import ROOT


class GANImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        data_path: str,
        image_size: tuple = (64, 64),
    ):
        """Generic datamodule for image datasets using the pytorchvision ImageFolder class

        Args:
            batch_size (int): Data batch size
            num_workers (int): Number of cpu workers
            image_size (tuple): Desired image size
        """
        super().__init__()
        self.data_dir = Path.joinpath(ROOT, data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        self.normalization_schema = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

        # Transforms
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(*self.normalization_schema),
            ]
        )

    def setup(self, stage: Optional[str] = None):
        self.ds = ImageFolder(self.data_dir, self.transforms)

    def train_dataloader(self):
        return DataLoader(
            self.ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )


class CelebADataModule(GANImageDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        image_size: tuple = (64, 64),
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            data_path="data/celeba/img_align_celeba",
            image_size=image_size,
        )


class ArtDataModule(GANImageDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        image_size: tuple = (64, 64),
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            data_path="data/art_dataset/resized",
            image_size=image_size,
        )


data_modules = {
    "celeba": CelebADataModule,
    "art": ArtDataModule,
}
