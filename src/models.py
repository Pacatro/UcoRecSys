import torch
from torch import nn


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
