import os
from pathlib import Path
from tkinter import W

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm

from inception import InceptionV3
from nehmegan.data_utils import verify_and_get_dataset_paths

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}


class FID:
    def __init__(
        self,
        dataset_name: str,
        image_size: int,
        batch_size: int = 256,
        num_workers: int = 1,
        device: str = "cpu",
    ) -> None:

        self.dataset_name = dataset_name
        self.dataset_root_path, self.dataset_images_path = verify_and_get_dataset_paths(
            dataset_name
        )
        self.dataset_statistics_filepath = Path.joinpath(
            self.dataset_root_path, "inception_statistics.npz"
        )

        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.device = device

        self.inception_output_dim = 2048
        self.inception_block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[
            self.inception_output_dim
        ]
        self.inception_model = InceptionV3(
            output_blocks=(self.inception_block_idx,)
        ).to(self.device)
        self.inception_model.eval()

        self.transforms = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

    def calculate_fid(self, images_batch):
        batch_mu, batch_sigma = self.compute_batch_statistics(images_batch)
        dataset_mu, dataset_sigma = self.get_dataset_statistics()
        fid_score = FID.calculate_frechet_distance(
            mu1=dataset_mu, sigma1=dataset_sigma, mu2=batch_mu, sigma2=batch_sigma
        )
        return fid_score

    def compute_batch_statistics(self, images_batch):
        # TODO: Verify that the dimensions make sense
        # Input dim: [batch_size, 128, 1, 1]
        # Why do we need the two empty dimensions?
        activations = self.inception_model(images_batch)[0]

        squeezed_activations = activations.squeeze().detach().cpu().numpy()
        batch_mu = np.mean(squeezed_activations, axis=0)
        batch_sigma = np.cov(squeezed_activations, rowvar=False)
        return batch_mu, batch_sigma

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), "Training and test mean vectors have different lengths"
        assert (
            sigma1.shape == sigma2.shape
        ), "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def get_dataset_statistics(self):
        if not os.path.exists(self.dataset_statistics_filepath):
            print("Dataset statistics not found. Calculating...")
            self.generate_dataset_statistics()

        return self.load_dataset_statistics()

    def load_dataset_statistics(self):
        dataset_statics = np.load(self.dataset_statistics_filepath)
        return dataset_statics["mu"], dataset_statics["sigma"]

    def generate_dataset_statistics(self):
        mu, sigma = self.compute_dataset_statistics()
        np.savez(self.dataset_statistics_filepath, mu=mu, sigma=sigma)

    def compute_dataset_statistics(self):
        m, s = self.calculate_dataset_activation_statistics()

        return m, s

    def calculate_dataset_activation_statistics(self):
        """Calculation of the statistics used by the FID.
        Params:
        -- files       : List of image files paths
        -- model       : Instance of inception model
        -- batch_size  : The images numpy array is split into batches with
                        batch size batch_size. A reasonable batch size
                        depends on the hardware.
        -- dims        : Dimensionality of features returned by Inception
        -- device      : Device to run calculations
        -- num_workers : Number of parallel dataloader workers
        Returns:
        -- mu    : The mean over samples of the activations of the pool_3 layer of
                the inception model.
        -- sigma : The covariance matrix of the activations of the pool_3 layer of
                the inception model.
        """
        activations = self.get_dataset_activations()
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def get_dataset_activations(self):
        """Calculates the activations of the pool_3 layer for all images.
        Params:
        -- files       : List of image files paths
        -- model       : Instance of inception model
        -- batch_size  : Batch size of images for the model to process at once.
                        Make sure that the number of samples is a multiple of
                        the batch size, otherwise some samples are ignored. This
                        behavior is retained to match the original FID score
                        implementation.
        -- dims        : Dimensionality of features returned by Inception
        -- device      : Device to run calculations
        -- num_workers : Number of parallel dataloader workers
        Returns:
        -- A numpy array of dimension (num images, dims) that contains the
        activations of the given tensor when feeding inception with the
        query tensor.
        """

        dataset = ImageFolder(self.dataset_root_path, self.transforms)

        self.ensure_batch_size_smaller_than_data_size(dataset_size=len(dataset))

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

        pred_arr = np.empty((len(dataset), self.inception_output_dim))
        start_idx = 0

        for batch, _ in tqdm(dataloader):
            batch = batch.to(self.device)

            with torch.no_grad():
                pred = self.inception_model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()

            pred_arr[start_idx : start_idx + pred.shape[0]] = pred

            start_idx = start_idx + pred.shape[0]

        return pred_arr

    def ensure_batch_size_smaller_than_data_size(self, dataset_size):
        if self.batch_size > dataset_size:
            print(
                (
                    "Warning: batch size is bigger than the data size. "
                    "Setting batch size to data size"
                )
            )
            self.batch_size = dataset_size
