from data_module import ArtDataModule
from utils import show_batch


datamod = ArtDataModule()

datamod.setup()
dl = datamod.train_dataloader()

show_batch(dl, datamod.normalization_schema)
