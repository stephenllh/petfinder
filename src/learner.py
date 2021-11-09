import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from model import RegressorNet


class PawpularityRegressor(LightningModule):
    def __init__(self, config=None):
        super().__init__()
        self.save_hyperparameters()
        # self.config = config
        self.model = RegressorNet()
        self.diff_lr = False  # TODO

    def forward(self, x):
        return self.model(x) * 100.0

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = F.mse_loss(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        rmse = 100 * torch.sqrt(F.mse_loss(preds, y))
        self.log("train_rmse", rmse, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = F.mse_loss(preds, y)
        self.log("val_loss", loss)
        rmse = 100 * torch.sqrt(F.mse_loss(preds, y))
        self.log("val_rmse", rmse, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx):
        image, id = batch
        return self.model(image) * 100.0, id

    def configure_optimizers(self):
        if not self.diff_lr:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        else:
            optimizer = torch.optim.Adam(
                [
                    {"params": self.model.backbone.parameters(), "lr": 1e-5},
                    {"params": self.model.linear.parameters(), "lr": 1e-4},
                ],
                lr=1e-3,
            )
        return optimizer
