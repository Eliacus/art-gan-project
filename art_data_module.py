import os
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).parent.resolve()

PATH_DATASETS = Path.joinpath(ROOT, "data/art_dataset/resized")
NUM_WORKERS = int(os.cpu_count() / 2)
BATCH_SIZE = 128
IMAGE_SIZE = (64, 64)


class ArtDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        image_size: tuple = IMAGE_SIZE,
    ):
        super().__init__()
        self.data_dir = PATH_DATASETS
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        # Internally set variables
        # NOTE: Maybe theese should be moved somewhere else?
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
