from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from data_module import BATCH_SIZE


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        self.model = nn.Sequential(
            # latent_dim x 1 x 1
            nn.ConvTranspose2d(
                self.latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64 x 32 x 32
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # 3 x 64 x 64
        )

    def forward(self, z):
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # 3 x 64 x 64
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 32 x 32
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 16 x 16
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(inplace=False),
            # 256 x 8 x 8
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 4 x 4
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # 1 x 1 x 1
            nn.Flatten(),
            nn.Sigmoid(),
        )

    def forward(self, img):
        validity = self.model(img)

        return validity


class GAN(pl.LightningModule):
    def __init__(
        self,
        channels,
        width,
        height,
        latent_dim: int = 128,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = BATCH_SIZE,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_shape = (channels, width, height)

        # Networks
        self.generator = Generator(
            latent_dim=self.hparams.latent_dim,
            img_shape=self.data_shape,
        )
        self.discriminator = Discriminator()

        self.validation_z = torch.randn(8, *self.data_shape)

    def forward(self, z):
        return self.generator(z)

    def adverserial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # Sample noise
        z = torch.randn(imgs.shape(0), *self.data_shape).type_as(imgs)

        # train generator
        if optimizer_idx == 0:

            # Generate images from generator
            self.generated_imgs = self(z)

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image("generated_images", grid, 0)

            # Create the ground thruth (i.e all fake)
            validities = torch.ones(imgs.size(1), 1).type_as(imgs)

            # Forward the images through the discriminator
            predicted_validities = self.discriminator(self.generated_imgs)

            # Calculate adverserial loss
            g_loss = self.adverserial_loss(predicted_validities, validities)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # train discriminator
        if optimizer_idx == 1:

            # How well can the discriminator identify real images
            valid = torch.ones(imgs.size(0), 1).type_as(imgs)
            predicted_validites_real = self.discriminator(imgs)
            real_loss = self.adverserial_loss(predicted_validites_real, valid)

            # How well can discriminator identify fake images
            fake = torch.zeros(imgs.size(0), 1).type_as(imgs)
            generated_imgs = self(z).detach()
            predicted_validites_fake = self.discriminator(generated_imgs)
            fake_loss = self.adverserial_loss(predicted_validites_fake, fake)

            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

        return [opt_g, opt_d], []

    def on_epoch_end(self):
        z = self.validation_Z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_images = self(z)
        grid = torchvision.utils.make_grid(sample_images)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
