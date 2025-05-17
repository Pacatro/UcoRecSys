import pandas as pd
import torch
from torch.utils.data import DataLoader
import lightning as L
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from typing import Callable, Optional

from dataset import ELearningDataset
from engine import UcoRecSys
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


def cross_validate(
    model_class: Callable[..., torch.nn.Module],
    df: pd.DataFrame,
    splitter: BaseCrossValidator = KFold,
    n_splits: int = 5,
    target: str = "rating",
    batch_size: int = 32,
    max_epochs: int = 50,
    early_stopping_patience: int = 0,
    early_stopping_delta: float = 0.0,
    k: int = 10,
    threshold: float = 7.5,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """
    Perform cross-validation over the DataFrame using the given splitter.
    """
    fold_metrics: list[dict[str, float]] = []

    for fold in range(n_splits):
        print(f"=== Fold {fold + 1}/{n_splits} ===")

        dm = ELearningCVDataModule(
            df=df,
            target=target,
            fold=fold,
            splitter=splitter,
            batch_size=batch_size,
            threshold=threshold,
        )
        dm.setup()

        model = model_class(
            n_users=dm.num_users,
            n_items=dm.num_items,
            numeric_features=dm.numeric_features,
            cat_cardinalities=dm.cat_cardinalities,
        )

        recsys = UcoRecSys(model=model, k=k, threshold=threshold)

        callbacks = [
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                dirpath=f"lightning_logs/fold_{fold}",
                filename="best",
            )
        ]
        if early_stopping_patience:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    min_delta=early_stopping_delta,
                )
            )

        trainer = L.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            deterministic=True,
            default_root_dir=f"lightning_logs/fold_{fold}",
            callbacks=callbacks,
        )

        trainer.fit(recsys, datamodule=dm)

        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        fold_metrics.append(metrics)
        print(f"Fold {fold + 1} metrics: {metrics}")

    # Average
    dfm = pd.DataFrame(fold_metrics)
    avg_metrics = dfm.mean().to_dict()
    return fold_metrics, avg_metrics


class ELearningCVDataModule(L.LightningDataModule):
    """
    DataModule for cross validation on tabular e-learning data.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        fold: int = 0,
        splitter: BaseCrossValidator = KFold,
        n_splits=5,
        batch_size: int = 128,
        threshold: float = 7.5,
        num_workers: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["df"])
        self.df = df.reset_index(drop=True)
        self.target = target
        self.fold = fold
        self.splitter = splitter(n_splits=n_splits, shuffle=True, random_state=42)
        self.batch_size = batch_size
        self.threshold = threshold
        self.num_workers = num_workers
        self.num_users = self.df["user_id"].nunique()
        self.num_items = self.df["item_id"].nunique()

        # datasets
        self.data_train = None
        self.data_val = None

    def setup(self, stage: Optional[str] = None):
        if self.data_train is None and self.data_val is None:
            splits = list(self.splitter.split(self.df))

            assert 0 <= self.hparams.fold < len(splits), "Fold index out of range"

            train_idx, val_idx = splits[self.hparams.fold]
            df_train = self.df.iloc[train_idx].reset_index(drop=True)
            df_val = self.df.iloc[val_idx].reset_index(drop=True)

            le_user = LabelEncoder().fit(self.df["user_id"])
            le_item = LabelEncoder().fit(self.df["item_id"])
            for d in [df_train, df_val]:
                d["user_id"] = le_user.transform(d["user_id"])
                d["item_id"] = le_item.transform(d["item_id"])

            encoders = {}
            scalers = {}
            self.cat_cardinalities = {}
            self.numeric_features = []

            for col in df_train.columns:
                if col in [self.hparams.target, "user_id", "item_id"]:
                    continue

                if isinstance(df_train[col].dtype, pd.CategoricalDtype):
                    le = LabelEncoder().fit(df_train[col])
                    encoders[col] = le
                    self.cat_cardinalities[col] = len(le.classes_)
                else:
                    ms = MinMaxScaler().fit(df_train[[col]])
                    scalers[col] = ms
                    self.numeric_features.append(col)

            self.data_train = ELearningDataset(
                df_train,
                encoders=encoders,
                scalers=scalers,
            )
            self.data_val = ELearningDataset(
                df_val,
                encoders=encoders,
                scalers=scalers,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
