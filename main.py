from art_data_module import ArtDataModule, BATCH_SIZE, IMAGE_SIZE
from celeba_datamodule import CelebADataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from cnn_model import DCGAN

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 128

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

wandb_logger = WandbLogger(project="gan-project")
dm = CelebADataModule(batch_size=BATCH_SIZE, num_workers=10, image_size=IMAGE_SIZE)

model = DCGAN(IMAGE_SIZE[0], IMAGE_SIZE[1], nc, nz, ngf, ndf)

trainer = Trainer(gpus=1, max_epochs=1000, log_every_n_steps=30, logger=wandb_logger)
trainer.fit(model, dm)
