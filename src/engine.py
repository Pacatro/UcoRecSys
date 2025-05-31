import torch
from torch import nn
import lightning.pytorch as L
from torchmetrics import MetricCollection, Metric
from torchmetrics.retrieval import (
    RetrievalPrecision,
    RetrievalRecall,
    RetrievalNormalizedDCG,
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
)


class RetrievalFBetaScore(Metric):
    def __init__(
        self, top_k: int = 10, beta: float = 1.0, adaptive_k: bool = True, **kwargs
    ):
        super().__init__(**kwargs)
        self.beta = beta
        self.top_k = top_k

        self.precision = RetrievalPrecision(top_k=top_k, adaptive_k=adaptive_k)
        self.recall = RetrievalRecall(top_k=top_k)

    def update(self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor):
        self.precision.update(preds, target, indexes=indexes)
        self.recall.update(preds, target, indexes=indexes)

    def compute(self):
        precision = self.precision.compute()
        recall = self.recall.compute()

        return ((1 + self.beta**2) * precision * recall) / (
            (self.beta**2 * precision) + recall
        )

    def reset(self):
        self.precision.reset()
        self.recall.reset()


class UcoRecSys(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        threshold: float = 8.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        k: int = 10,
        loss_fn: nn.Module | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.loss_fn = nn.MSELoss() if loss_fn is None else loss_fn
        self.threshold = threshold
        self.lr = lr
        self.weight_decay = weight_decay

        metrics = MetricCollection(
            {
                f"Precison@{k}": RetrievalPrecision(top_k=k, adaptive_k=True),
                f"Recall@{k}": RetrievalRecall(top_k=k),
                f"F1@{k}": RetrievalFBetaScore(top_k=k, beta=1.0, adaptive_k=True),
                f"NDCG@{k}": RetrievalNormalizedDCG(top_k=k),
                f"HitRate@{k}": RetrievalHitRate(top_k=k),
                f"MAP@{k}": RetrievalMAP(top_k=k),
                f"MRR@{k}": RetrievalMRR(top_k=k),
            }
        )

        self.test_metrics = metrics.clone(prefix="test/")
        self.val_metrics = metrics.clone(prefix="val/")

    def forward(self, batch):
        score = self.model(batch)
        return {
            "user_id": batch["user_id"].long(),
            "item_id": batch["item_id"].long(),
            "prediction": score.detach(),
            "rating": batch["rating"].float(),
        }

    def step(self, batch, prefix, metrics=None):
        preds = self.model(batch)
        loss = self.loss_fn(preds, batch["rating"].float())

        if metrics is not None:
            target = (batch["rating"] >= self.threshold).int()
            user_ids = batch["user_id"].long()

            metrics.update(
                preds,
                target,
                indexes=user_ids,
            )

        self.log(f"{prefix}/mse", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            f"{prefix}/rmse",
            torch.sqrt(loss),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def training_step(self, batch):
        return self.step(batch, "train")

    def validation_step(self, batch):
        self.step(batch, "val", metrics=self.val_metrics)

    def test_step(self, batch):
        self.step(batch, "test", metrics=self.test_metrics)

    def on_validation_epoch_start(self):
        self.val_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute())

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())

    def predict_step(self, batch):
        return self.forward(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/mse"},
        }
