import torch
from torch import nn


class BasicNeuralMF(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        additional_features: list[str],
        emb_dim: int = 128,
        hidden_dims: list[int] = [256, 128, 64, 32, 16],
        dropout: float = 0.1,
        min_rating: float = 1.0,
        max_rating: float = 10.0,
    ):
        super().__init__()

        # CF embeddings
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)

        # MLP
        n_num = len(additional_features)
        mlp_input = 2 * emb_dim + n_num
        layers = []
        for h in hidden_dims:
            layers += [
                nn.Linear(mlp_input, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            mlp_input = h

        layers.append(nn.Linear(mlp_input, 1))
        self.mlp = nn.Sequential(*layers)

        # Hyperparameters
        self.additional_features = additional_features
        self.min_rating = min_rating
        self.max_rating = max_rating

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        u = batch["user_id"].long()
        i = batch["item_id"].long()
        u_emb = self.user_embedding(u)
        i_emb = self.item_embedding(i)

        num_vecs = [batch[n].unsqueeze(1).float() for n in self.additional_features]
        num_embs = (
            torch.cat(num_vecs, dim=1)
            if num_vecs
            else torch.zeros(u.size(0), 0, device=u_emb.device)
        )

        x = torch.cat([u_emb, i_emb, num_embs], dim=1)
        score = self.mlp(x)

        return score.clamp(min=self.min_rating, max=self.max_rating).squeeze(1)


class NeuralMF(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        cat_cardinalities: dict[str, int],
        numeric_features: list[str],
        emb_dim: int = 128,
        hidden_dims: list[int] = [256, 128, 64, 32, 16],
        dropout: float = 0.1,
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

        layers.append(nn.Linear(mlp_input, 1))
        self.mlp = nn.Sequential(*layers)

        # Hyperparameters
        self.numeric_features = numeric_features
        self.min_rating = min_rating
        self.max_rating = max_rating

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        u = batch["user_id"].long()
        i = batch["item_id"].long()
        u_emb = self.user_embedding(u)
        i_emb = self.item_embedding(i)

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

        x = torch.cat([u_emb, i_emb, cat_embs, num_embs], dim=1)
        score = self.mlp(x)

        return score.clamp(min=self.min_rating, max=self.max_rating).squeeze(1)


class GMFMLP(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        cat_cardinalities: dict[str, int],
        numeric_features: list[str],
        emb_dim: int = 128,
        hidden_dims: list[int] = [256, 128, 64, 32, 16],
        dropout: float = 0.1,
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


class DeepHybridModel(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        cat_cardinalities: dict[str, int],
        numeric_features: list[str],
        emb_dim: int = 32,
        hidden_dims: list[int] = [64, 32, 16],
        dropout: float = 0.5,
        min_rating: float = 0,
        max_rating: float = 10,
    ):
        super().__init__()

        self.min_rating = min_rating
        self.max_rating = max_rating

        # MF embeddings for IDs
        mf_dim = emb_dim
        self.user_mf_emb = nn.Embedding(n_users, mf_dim)
        self.item_mf_emb = nn.Embedding(n_items, mf_dim)

        # DNN embeddings for IDs
        self.user_dnn_emb = nn.Embedding(n_users, emb_dim)
        self.item_dnn_emb = nn.Embedding(n_items, emb_dim)

        # Categorical feature embeddings
        self.cat_feats = list(cat_cardinalities.keys())
        self.cat_embs = nn.ModuleDict(
            {
                fname: nn.Embedding(cardinality, emb_dim)
                for fname, cardinality in cat_cardinalities.items()
            }
        )

        # Continuous feature normalization
        self.numeric_feats = numeric_features
        self.num_norm = (
            nn.BatchNorm1d(len(numeric_features)) if numeric_features else None
        )

        # Build DNN on concatenated embeddings + continuous features
        dnn_input_dim = emb_dim * (2 + len(self.cat_feats))
        if numeric_features:
            dnn_input_dim += len(numeric_features)

        layers = []
        for hidden in hidden_dims:
            layers.append(nn.Linear(dnn_input_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            dnn_input_dim = hidden
        self.dnn = nn.Sequential(*layers)

        # Final hybrid layer
        hybrid_input_dim = 1 + hidden_dims[-1]
        self.hybrid_layer = nn.Sequential(
            nn.Linear(hybrid_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        user_id = batch["user_id"].long()
        item_id = batch["item_id"].long()

        # MF branch
        u_mf = self.user_mf_emb(user_id)
        i_mf = self.item_mf_emb(item_id)
        mf_score = (u_mf * i_mf).sum(dim=1, keepdim=True)

        # DNN branch
        u_dnn = self.user_dnn_emb(user_id)
        i_dnn = self.item_dnn_emb(item_id)

        # Categorical embeddings
        cat_embs = [
            self.cat_embs[fname](batch[fname].long()) for fname in self.cat_feats
        ]

        # Numeric features
        if self.numeric_feats:
            nums = [batch[fname].unsqueeze(1) for fname in self.numeric_feats]
            num_tensor = torch.cat(nums, dim=1)
            num_norm = self.num_norm(num_tensor)

        # Concatenate all inputs for DNN
        terms = [u_dnn, i_dnn] + cat_embs
        if self.numeric_feats:
            terms.append(num_norm)
        x = torch.cat(terms, dim=1)
        dnn_out = self.dnn(x)

        # Hybrid fusion
        hybrid = torch.cat([mf_score, dnn_out], dim=1)
        out = self.hybrid_layer(hybrid).squeeze(1)

        return torch.clamp(out, self.min_rating, self.max_rating)
