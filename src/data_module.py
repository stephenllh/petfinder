import cv2
from sklearn.model_selection import train_test_split
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule


class PawpularityDataset:
    def __init__(self, dataframe, is_train, visualize, get_transforms_func):
        df_split = train_test_split(dataframe, test_size=0.1, random_state=0)
        df = df_split[0] if is_train else df_split[1]
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.visualize = visualize
        self.get_transforms_func = get_transforms_func

    def __getitem__(self, idx):
        filename = self.df.iloc[idx]["Id"]
        image = cv2.imread(f"./input/train/{filename}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmentations = self.get_transforms_func(self.is_train, self.visualize)
        image = augmentations(image=image)["image"]
        target = self.df.iloc[idx]["Pawpularity"] / 100.0
        target = target.astype("float32")
        return image, target

    def __len__(self):
        return len(self.df)


def get_transforms(is_train=True, visualize=False):
    tfms = [
        alb.LongestMaxSize(max_size=224),
        alb.PadIfNeeded(min_height=224, min_width=224),
    ]
    if is_train:
        tfms += [
            alb.HorizontalFlip(p=0.5),
            alb.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.02,
                p=0.75,
            ),
        ]

    if not visualize:
        tfms += [alb.Normalize(), ToTensorV2()]

    return alb.Compose(tfms)


class PawpularityDataModule(LightningDataModule):
    def __init__(self, dataframe, batch_size):
        super().__init__()
        self.dataframe = dataframe
        self.batch_size = batch_size

    def train_dataloader(self):
        train_dataset = PawpularityDataset(
            self.dataframe,
            is_train=True,
            visualize=False,
            get_transforms_func=get_transforms,
        )
        return DataLoader(train_dataset, self.batch_size)

    def val_dataloader(self):
        val_dataset = PawpularityDataset(
            self.dataframe,
            is_train=False,
            visualize=False,
            get_transforms_func=get_transforms,
        )
        return DataLoader(val_dataset, self.batch_size)

    def predict_dataloader(self):
        val_dataset = PawpularityDataset(
            self.dataframe,
            is_train=False,
            visualize=False,
            get_transforms_func=get_transforms,
        )
        return DataLoader(val_dataset, self.batch_size)
