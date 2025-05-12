import torch
from torch import nn
import lightning.pytorch as L
from torchmetrics import MetricCollection
from torchmetrics.retrieval import (
    RetrievalPrecision,
    RetrievalRecall,
    RetrievalNormalizedDCG,
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
)


class UcoRecSys(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        threshold: float = 8.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        k: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.threshold = threshold
        self.lr = lr
        self.weight_decay = weight_decay

        # Define retrieval metrics with user grouping via indexes
        metrics = MetricCollection(
            RetrievalPrecision(top_k=k, adaptive_k=True),
            RetrievalRecall(top_k=k),
            RetrievalNormalizedDCG(top_k=k),
            RetrievalHitRate(top_k=k),
            RetrievalMAP(top_k=k),
            RetrievalMRR(top_k=k),
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, batch):
        score = self.model(batch)
        return {
            "user_id": batch["user_id"].long(),
            "item_id": batch["item_id"].long(),
            "prediction": score.detach(),
            "rating": batch["rating"].float(),
        }

    def step(self, batch, metrics, prefix):
        preds = self.model(batch)
        loss = self.loss_fn(preds, batch["rating"].float())

        # Binarize ratings to relevance labels
        target = (batch["rating"] >= self.threshold).int()
        user_ids = batch["user_id"].long()

        metrics.update(
            preds,
            target,
            indexes=user_ids,
        )

        self.log(f"{prefix}_loss", loss, prog_bar=True)
        self.log(f"{prefix}_rmse", torch.sqrt(loss), prog_bar=True)

        return loss

    def training_step(self, batch):
        return self.step(batch, self.train_metrics, "train")

    def on_train_epoch_start(self):
        self.train_metrics.reset()

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())

    def validation_step(self, batch):
        self.step(batch, self.val_metrics, "val")

    def on_validation_epoch_start(self):
        self.val_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute())

    def test_step(self, batch):
        self.step(batch, self.test_metrics, "test")

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())

    def predict_step(self, batch):
        score = self.model(batch)
        return {
            "user_id": batch["user_id"].long(),
            "item_id": batch["item_id"].long(),
            "prediction": score.detach(),
            "rating": batch["rating"].float(),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
