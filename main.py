from data_module import ArtDataModule
from pytorch_lightning import Trainer
from cnn_model import GAN, BATCH_SIZE

image_size = (64, 64)
dm = ArtDataModule("rezied", batch_size=BATCH_SIZE, num_workers=10, image_size=image_size)

model = GAN(3, image_size[0], image_size[1], 128)

trainer = Trainer(gpus=1, max_epochs=5, progress_bar_refresh_rate=20)
trainer.fit(model, dm)
