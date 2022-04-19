from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from art_data_module import BATCH_SIZE
from utils import accuracy


# TODO: Rename all non-explanatory variables
class Generator(nn.Module):
    def __init__(self, nz, num_generator_features, num_channels):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, num_generator_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_generator_features * 8),
            nn.ReLU(True),
            # state size. (num_generator_features*8) x 4 x 4
            nn.ConvTranspose2d(
                num_generator_features * 8, num_generator_features * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(num_generator_features * 4),
            nn.ReLU(True),
            # state size. (num_generator_features*4) x 8 x 8
            nn.ConvTranspose2d(
                num_generator_features * 4, num_generator_features * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(num_generator_features * 2),
            nn.ReLU(True),
            # state size. (num_generator_features*2) x 16 x 16
            nn.ConvTranspose2d(
                num_generator_features * 2, num_generator_features, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(num_generator_features),
            nn.ReLU(True),
            # state size. (num_generator_features) x 32 x 32
            nn.ConvTranspose2d(num_generator_features, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (num_channels) x 64 x 64
        )

    def forward(self, input):
        return self.model(input)


class Discriminator(nn.Module):
    def __init__(self, num_channels, num_discriminator_features):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # input is (num_channels) x 64 x 64
            nn.Conv2d(num_channels, num_discriminator_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_discriminator_features) x 32 x 32
            nn.Conv2d(
                num_discriminator_features, num_discriminator_features * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(num_discriminator_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_discriminator_features*2) x 16 x 16
            nn.Conv2d(
                num_discriminator_features * 2, num_discriminator_features * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(num_discriminator_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_discriminator_features*4) x 8 x 8
            nn.Conv2d(
                num_discriminator_features * 4, num_discriminator_features * 8, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(num_discriminator_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_discriminator_features*8) x 4 x 4
            nn.Conv2d(num_discriminator_features * 8, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.model(input)


class DCGAN(pl.LightningModule):
    def __init__(
        self,
        width,
        height,
        num_channels,
        nz,
        num_generator_features,
        num_discriminator_features,
        latent_dim: int = 128,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = BATCH_SIZE,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_shape = (num_channels, width, height)

        # Networks
        self.generator = Generator(nz, num_generator_features, num_channels)
        self.discriminator = Discriminator(num_channels, num_discriminator_features)

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        self.validation_z = torch.randn(4, self.hparams.latent_dim, 1, 1)

    def forward(self, z):
        return self.generator(z)

    def adverserial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # Sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim, 1, 1).type_as(imgs)

        # train generator
        if optimizer_idx == 0:

            # Generate images from generator
            self.generated_imgs = self(z)

            # log sampled images
            # sample_imgs = self.generated_imgs[:6]
            # grid = torchvision.utils.make_grid(sample_imgs)
            # self.logger.experiment.add_image("generated_images", grid, 0)

            # Create the ground thruth (i.e all fake)
            validities = torch.ones(imgs.size(0), 1).type_as(imgs)

            # Forward the images through the discriminator
            predicted_validities = self.discriminator(self(z))

            # Calculate adverserial loss
            g_loss = self.adverserial_loss(predicted_validities, validities)
            generator_acc = accuracy(validities.view(-1), predicted_validities.view(-1))
            tqdm_dict = {"g_loss": g_loss}
            self.log("generator_accuracy", generator_acc)
            self.log("g_loss", g_loss)
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # train discriminator
        if optimizer_idx == 1:
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adverserial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adverserial_loss(self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            self.log("d_loss", d_loss)
            output = {"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

        return [opt_g, opt_d], []

    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_images = self(z)
        grid = torchvision.utils.make_grid(sample_images)
        self.logger.log_image(key="validation_images", images=[grid])


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
