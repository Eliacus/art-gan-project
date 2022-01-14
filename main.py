from data_module import ArtDataModule, BATCH_SIZE, IMAGE_SIZE
from pytorch_lightning import Trainer
from cnn_model import GAN

dm = ArtDataModule("resized", batch_size=BATCH_SIZE, num_workers=10, image_size=IMAGE_SIZE)

model = GAN(3, IMAGE_SIZE[0], IMAGE_SIZE[1], 100)

trainer = Trainer(gpus=1, max_epochs=1000, progress_bar_refresh_rate=20)
trainer.fit(model, dm)
