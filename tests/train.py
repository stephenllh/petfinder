import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from src.data_module import PawpularityDataModule
from src.learner import PawpularityRegressor


def train():
    dataframe = pd.read_csv("./input/train.csv")
    data_module = PawpularityDataModule(dataframe, batch_size=32)
    regressor = PawpularityRegressor()
    ModelCheckpoint(monitor="val_loss")
    wandb_logger = WandbLogger(project="pawpularity")
    trainer = Trainer(
        logger=wandb_logger, gpus=1, max_epochs=1, precision=16, fast_dev_run=True
    )
    trainer.fit(regressor, data_module)


if __name__ == "__main__":
    train()
