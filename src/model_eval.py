import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from typing import Callable

from dataset import ELearningDataModule
from engine import UcoRecSys
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import lightning as L


def cross_validate_loo(
    df: pd.DataFrame,
    target: str,
    model_cls: Callable[..., nn.Module],
    batch_size: int = 128,
    epochs: int = 50,
    threshold: float = 7.5,
    patience: int = 5,
    delta: float = 0.001,
    k: int = 10,
) -> tuple[list, dict]:
    """
    Performs a Leave-One-Out cross-validation on the given data with the given model.

    :param df: The data to be used for cross-validation.
    :param target: The target column to be used for cross-validation.
    :param model_cls: Constructor of the model to be used for cross-validation.
    :param batch_size: The batch size to be used for cross-validation.
    :param epochs: The total of epochs to be used for cross-validation.
    :param threshold: The threshold to be used for evaluation.
    :param patience: Patience to be used for early stopping.
    :param delta: Delta to be used for early stopping.
    :param k: The size of the ranking to be evaluated.
    """
    loo = LeaveOneOut()
    metrics_all = []

    for idx, (train_idx, test_idx) in enumerate(loo.split(df), start=1):
        print(f"\n--- Fold {idx}/{len(df)} (LOO) ---")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        dm = ELearningDataModule(
            df,
            target=target,
            batch_size=batch_size,
            threshold=threshold,
            balance=False,
            train_frac=0.98,
            val_frac=0.01,
        )

        dm.train_df = train_df
        dm.test_df = test_df
        dm.setup(stage="fit")
        dm.setup(stage="test")

        cat_cardinalities = dm.cat_cardinalities
        numeric_features = dm.numeric_features
        model = model_cls(
            n_users=dm.num_users,
            n_items=dm.num_items,
            cat_cardinalities=cat_cardinalities,
            numeric_features=numeric_features,
            emb_dim=32,
            hidden_dims=[64, 32, 16],
            dropout=0.5,
            min_rating=dm.min_rating,
            max_rating=dm.max_rating,
        )

        recsys = UcoRecSys(
            model=model,
            threshold=threshold,
            k=k,
        )

        cb_early = EarlyStopping(
            monitor="val_loss", patience=patience, mode="min", min_delta=delta
        )
        cb_ckpt = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
        trainer = L.Trainer(
            max_epochs=epochs, callbacks=[cb_early, cb_ckpt], enable_progress_bar=True
        )

        trainer.fit(recsys, datamodule=dm)

        test_metrics = trainer.test(recsys, datamodule=dm, verbose=False)
        metrics_all.append(test_metrics[0])

        torch.cuda.empty_cache()

    df_metrics = pd.DataFrame(metrics_all)
    avg_metrics = df_metrics.mean().to_dict()
    return metrics_all, avg_metrics
