import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.utilities.seed import seed_everything
import warnings
from data_module import PawpularityDataModule
from learner import PawpularityRegressor
from utils import get_neptune_api_token


warnings.filterwarnings(
    "ignore", message="Consider increasing the value of the `num_workers` argument`"
)


def train():
    dataframe = pd.read_csv("./input/train.csv")
    data_module = PawpularityDataModule(dataframe, batch_size=32)
    config = {"name": "swin_transformer", "lr": 0.003}
    regressor = PawpularityRegressor(config)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    neptune_logger = NeptuneLogger(
        api_key=get_neptune_api_token(),
        project_name="stephenllh/pawpularity",
    )
    for script_file in ["model.py", "learner.py"]:
        neptune_logger.experiment.log_artifact(f"./src/{script_file}")

    trainer = Trainer(
        callbacks=[checkpoint_callback],
        logger=neptune_logger,
        gpus=1,
        max_epochs=10,
        precision=16,
        limit_train_batches=1,
        limit_val_batches=1,
    )
    trainer.fit(regressor, data_module)


if __name__ == "__main__":
    seed_everything(0, True)
    train()
