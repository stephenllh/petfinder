import pandas as pd
import cv2
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from data_module import get_transforms
from learner import PawpularityRegressor


class PawpularityTestDataset:
    def __init__(self, dataframe, get_transforms_func):
        self.dataframe = dataframe
        self.get_transforms_func = get_transforms_func

    def __getitem__(self, idx):
        filename = self.dataframe.iloc[idx]["Id"]
        image = cv2.imread(f"./input/test/{filename}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmentations = self.get_transforms_func(is_train=False)
        image = augmentations(image=image)["image"]
        return image, filename

    def __len__(self):
        return len(self.dataframe)


def predict():
    test_dataframe = pd.read_csv("./input/test.csv")
    test_dataset = PawpularityTestDataset(
        test_dataframe, get_transforms_func=get_transforms
    )
    test_dataloader = DataLoader(test_dataset, batch_size=16, num_workers=4)

    regressor = PawpularityRegressor.load_from_checkpoint(
        "checkpoints/epoch=0-step=557.ckpt"
    )
    trainer = Trainer(gpus=1, precision=32, logger=None)
    predictions, ids = trainer.predict(regressor, dataloaders=test_dataloader)[0]
    prediction_dict = {"Id": ids, "Pawpularity": predictions.tolist()}
    submission_dataframe = pd.DataFrame.from_dict(prediction_dict)
    submission_dataframe.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    predict()
