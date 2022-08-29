import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from nehmegan.data_utils import verify_and_get_dataset_paths


class GANImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        dataset_name: str,
        image_size: tuple = (64, 64),
    ):
        """
        Generic datamodule for image datasets using the pytorchvision ImageFolder class

        Args:
            batch_size (int): Data batch size
            num_workers (int): Number of cpu workers
            dataset (str): dataset name. This requires there to be a folder named data/dataset which in turn contains
                           a folder images/ which contains all the dataset images.
            image_size (tuple): Desired image size
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_root_path, self.dataset_images_path = verify_and_get_dataset_paths(
            self.dataset_name
        )

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

    def setup(self):
        self.dataset = ImageFolder(self.dataset_images_path, self.transforms)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )
