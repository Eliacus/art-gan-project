import os

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from data_modules import GANImageDataModule
from DCGAN import DCGAN

# Dataset
dataset = "celeba"
# dataset = "art"

# Number of channels in the training images. For color images this is 3
num_channels = 3

# Size of z latent vector (i.e. size of generator input)
latent_dim = 128

# Size of feature maps in generator
num_generator_features = 128

# Size of feature maps in discriminator
num_discriminator_features = 64

NUM_WORKERS = int(os.cpu_count() - 1)
BATCH_SIZE = 256

wandb_logger = WandbLogger(project="gan-project")
dm = GANImageDataModule(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, dataset=dataset)

model = DCGAN(
    num_channels=num_channels,
    latent_dim=latent_dim,
    num_generator_features=num_discriminator_features,
    num_discriminator_features=num_discriminator_features,
)

trainer = Trainer(
    gpus=1,
    max_epochs=1000,
    precision=16,
    log_every_n_steps=30,
    logger=wandb_logger,
    track_grad_norm=2,
)
trainer.fit(model, dm)
