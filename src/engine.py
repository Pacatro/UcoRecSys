import torch
from torch import nn
import lightning.pytorch as L
import matplotlib.pyplot as plt
from torchmetrics import MetricCollection, Metric
from torchmetrics.retrieval import (
    RetrievalPrecision,
    RetrievalRecall,
    RetrievalNormalizedDCG,
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
)
# from torchmetrics.regression import R2Score, ExplainedVariance


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
        val_metrics_path: str = "val_metrics.png",
        train_metrics_path: str = "train_metrics.png",
        train_losses_path: str = "train_losses.png",
        encoders: dict | None = None,
        plot: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.loss_fn = nn.MSELoss() if loss_fn is None else loss_fn
        self.threshold = threshold
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_metrics_path = val_metrics_path
        self.train_metrics_path = train_metrics_path
        self.train_losses_path = train_losses_path
        self.plot = plot
        self.encoders = encoders

        ranking_metrics = MetricCollection(
            {
                f"Precision@{k}": RetrievalPrecision(top_k=k, adaptive_k=True),
                f"Recall@{k}": RetrievalRecall(top_k=k),
                f"F1@{k}": RetrievalFBetaScore(top_k=k, beta=1.0, adaptive_k=True),
                f"NDCG@{k}": RetrievalNormalizedDCG(top_k=k),
                f"HR@{k}": RetrievalHitRate(top_k=k),
                f"MAP@{k}": RetrievalMAP(top_k=k),
                f"MRR@{k}": RetrievalMRR(top_k=k),
            }
        )
        # predicted_metrics = MetricCollection(
        #     {
        #         "R2": R2Score(),
        #         "Explained Variance": ExplainedVariance(),
        #     }
        # )

        self.val_ranking_metrics = ranking_metrics.clone(prefix="val/")
        # self.val_predicted_metrics = predicted_metrics.clone(prefix="val/")
        self.test_ranking_metrics = ranking_metrics.clone(prefix="test/")
        # self.test_predicted_metrics = predicted_metrics.clone(prefix="test/")

        # For plotting
        self.train_metrics_history = []
        self.val_metrics_history = []
        self.train_losses = []
        self.train_ranking_metrics = (
            ranking_metrics.clone(prefix="train/") if self.plot else None
        )

    def forward(self, batch):
        score = self.model(batch)

        if self.encoders:
            user_id_tensor = batch["user_id"]
            item_id_tensor = batch["item_id"]

            # Asegúrate de que los tensores estén en la CPU y convertidos a NumPy
            user_id_array = user_id_tensor.detach().cpu().numpy()
            item_id_array = item_id_tensor.detach().cpu().numpy()

            user_id = self.encoders["user_id"].inverse_transform(
                user_id_array.reshape(-1, 1)
            )
            item_id = self.encoders["item_id"].inverse_transform(
                item_id_array.reshape(-1, 1)
            )
        else:
            user_id = batch["user_id"].long()
            item_id = batch["item_id"].long()

        return {
            "user_id": user_id.ravel(),
            "item_id": item_id.ravel(),
            "prediction": score.detach(),
            "relevant": score.detach() > self.threshold,
        }

    def step(
        self,
        batch: dict,
        prefix: str,
        ranking_metrics: MetricCollection | None = None,
        # predicted_metrics: MetricCollection | None = None,
    ):
        ratings = batch["rating"]
        user_ids = batch["user_id"].long()

        preds = self.model(batch)
        loss = self.loss_fn(preds, ratings.float())
        self.train_losses.append(loss.item())

        if ranking_metrics is not None:
            target = (ratings >= self.threshold).int()
            ranking_metrics.update(
                preds,
                target,
                indexes=user_ids,
            )

        # if predicted_metrics is not None:
        #     predicted_metrics.update(preds, ratings)

        self.log(f"{prefix}/MSE", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            f"{prefix}/RMSE",
            torch.sqrt(loss),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def training_step(self, batch):
        return self.step(batch, "train", ranking_metrics=self.train_ranking_metrics)

    def validation_step(self, batch):
        self.step(
            batch,
            "val",
            ranking_metrics=self.val_ranking_metrics,
            # predicted_metrics=self.val_predicted_metrics,
        )

    def test_step(self, batch):
        self.step(
            batch,
            "test",
            ranking_metrics=self.test_ranking_metrics,
            # predicted_metrics=self.test_predicted_metrics,
        )

    def on_train_epoch_start(self):
        if self.plot:
            self.train_ranking_metrics.reset()

    def on_train_epoch_end(self):
        if self.plot:
            train_ranking_metrics = self.train_ranking_metrics.compute()
            self.train_metrics_history.append(train_ranking_metrics)
            self.log_dict(train_ranking_metrics)

    def on_validation_epoch_start(self):
        self.val_ranking_metrics.reset()
        # self.val_predicted_metrics.reset()

    def on_validation_epoch_end(self):
        val_ranking_metrics = self.val_ranking_metrics.compute()
        # val_predicted_metrics = self.val_predicted_metrics.compute()
        # val_metrics = val_ranking_metrics.update(val_predicted_metrics)
        self.val_metrics_history.append(val_ranking_metrics)
        # self.log_dict(val_predicted_metrics)
        self.log_dict(val_ranking_metrics)

    def on_test_epoch_start(self):
        self.test_ranking_metrics.reset()
        # self.test_predicted_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_ranking_metrics.compute())
        # self.log_dict(self.test_predicted_metrics.compute())

    def on_fit_end(self):
        if self.plot:
            fig, ax = plt.subplots(figsize=(12, 8))
            self.val_ranking_metrics.plot(
                val=self.val_metrics_history, ax=ax, together=True
            )
            fig.savefig(self.val_metrics_path)

            fig, ax = plt.subplots(figsize=(12, 8))
            self.train_ranking_metrics.plot(
                val=self.train_metrics_history, ax=ax, together=True
            )
            fig.savefig(self.train_metrics_path)

            self.plot_train_losses()

    def predict_step(self, batch):
        return self.forward(batch)

    def plot_train_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Training Loss (MSE)")
        plt.xlabel("Batch Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss Evolution")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.train_losses_path)
        plt.close()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/MSE"},
        }
