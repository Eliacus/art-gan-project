import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        self.model = nn.Sequential(
            # latent_dim x 1 x 1
        )


class Discriminator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        self.model = nn.Sequential(
            # 3 x 64 x 64
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 32 x 32
            nn.Conv2d(
                in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False
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
