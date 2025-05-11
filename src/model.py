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


class GMFMLP(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        cat_cardinalities: dict[str, int],
        numeric_features: list[str],
        emb_dim: int = 32,
        hidden_dims: list[int] = [64, 32, 16],
        dropout: float = 0.5,
        global_mean: float = 7.0,
        min_rating: float = 1.0,
        max_rating: float = 10.0,
    ):
        super().__init__()

        # CF embeddings
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)

        # Content Categories embeddings (CB)
        self.cat_embeddings = nn.ModuleDict(
            {
                key: nn.Embedding(card, emb_dim // 2)
                for key, card in cat_cardinalities.items()
            }
        )

        # MLP
        n_cat = len(self.cat_embeddings)
        n_num = len(numeric_features)
        mlp_input = 2 * emb_dim + n_cat * (emb_dim // 2) + n_num
        layers = []
        for h in hidden_dims:
            layers += [
                nn.Linear(mlp_input, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            mlp_input = h

        layers.append(nn.Linear(mlp_input, hidden_dims[-1]))
        self.mlp = nn.Sequential(*layers)

        # GMF + MLP
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.combine = nn.Linear(emb_dim + hidden_dims[-1], 1)

        # Hyperparameters
        self.numeric_features = numeric_features
        self.global_mean = global_mean
        self.min_rating = min_rating
        self.max_rating = max_rating

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        u = batch["user_id"].long()
        i = batch["item_id"].long()
        u_emb = self.user_embedding(u)
        i_emb = self.item_embedding(i)

        gmf = u_emb * i_emb

        cat_vecs = [emb(batch[key].long()) for key, emb in self.cat_embeddings.items()]
        cat_embs = (
            torch.cat(cat_vecs, dim=1)
            if cat_vecs
            else torch.zeros(u.size(0), 0, device=u_emb.device)
        )

        num_vecs = [batch[n].unsqueeze(1).float() for n in self.numeric_features]
        num_embs = (
            torch.cat(num_vecs, dim=1)
            if num_vecs
            else torch.zeros(u.size(0), 0, device=u_emb.device)
        )

        mlp_in = torch.cat([u_emb, i_emb, cat_embs, num_embs], dim=1)
        mlp_vec = self.mlp(mlp_in)
        combo = torch.cat([gmf, mlp_vec], dim=1)
        score = self.combine(combo).squeeze(1)

        score = (
            score
            + self.user_bias(u).squeeze(1)
            + self.item_bias(i).squeeze(1)
            + self.global_mean
        )

        return score.clamp(min=self.min_rating, max=self.max_rating)


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
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.threshold = threshold
        self.lr = lr
        self.weight_decay = weight_decay
        # metrics
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
        target = (batch["rating"] >= self.threshold).int()
        metrics.update(preds, target, indexes=batch["user_id"].long())

        self.log(f"{prefix}_loss", loss, prog_bar=(prefix != "train"))

        if prefix != "train":
            self.log(f"{prefix}_rmse", torch.sqrt(loss), prog_bar=True)

        return loss

    def training_step(self, batch):
        return self.step(batch, self.train_metrics, "train")

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch):
        self.step(batch, self.val_metrics, "val")

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, batch):
        self.step(batch, self.test_metrics, "test")

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def predict_step(self, batch):
        score = self.model(batch)
        return {
            "user_id": batch["user_id"].long(),
            "item_id": batch["item_id"].long(),
            "prediction": score.detach(),
            "rating": batch["rating"].float(),
        }

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=3
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"},
        }
