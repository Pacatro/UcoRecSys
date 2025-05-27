import pandas as pd
import lightning as L
from sklearn.model_selection import KFold, LeaveOneOut
from typing import Literal
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from dataset import ELearningDataModule
from engine import UcoRecSys


def cross_validate(
    df: pd.DataFrame,
    model_class: type,
    n_splits: int = 5,
    random_state: int = 42,
    epochs: int = 100,
    cv_type: Literal["kfold", "loo"] = "kfold",
    callbacks: list[L.Callback] = [],
    k: int = 10,
    batch_size: int = 128,
    patience: int = 5,
    delta: float = 0.001,
    verbose: bool = False,
) -> pd.DataFrame:
    cv = (
        KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        if cv_type == "kfold"
        else LeaveOneOut()
    )
    fold_metrics = []
    n_folds = cv.get_n_splits(X=df)

    for fold, (train_idx, test_idx) in enumerate(cv.split(df), start=1):
        print(f"Fold {fold}/{n_folds}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[test_idx].reset_index(drop=True)

        dm = ELearningDataModule(
            df=pd.concat([train_df, val_df], ignore_index=True),
            batch_size=batch_size,
            test_size=0,
            val_size=len(val_df) / (len(train_df) + len(val_df)),
        )
        dm.setup("fit")

        model = model_class(
            cat_cardinalities=dm.cat_cardinalities,
            cont_features=dm.cont_features,
            n_users=dm.num_users,
            n_items=dm.num_items,
        )

        recsys = UcoRecSys(model=model, threshold=dm.threshold)

        callbacks = [
            EarlyStopping(
                monitor="val/loss",
                patience=patience,
                mode="min",
                min_delta=delta,
                verbose=False,
            ),
            ModelCheckpoint(
                monitor="val/loss", mode="min", save_top_k=1, filename="best-model"
            ),
        ]

        trainer = L.Trainer(
            max_epochs=epochs,
            accelerator="auto",
            devices="auto",
            callbacks=callbacks,
            log_every_n_steps=10,
            enable_model_summary=False,
            inference_mode=False,
            enable_progress_bar=verbose,
        )

        trainer.fit(recsys, datamodule=dm)

        recsys = UcoRecSys.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            model=model,
            k=k,
            threshold=dm.threshold,
        )

        metrics = trainer.validate(recsys, datamodule=dm)
        print(metrics[0])
        fold_metrics.append(metrics[0])

    avg_metrics = pd.DataFrame(fold_metrics).mean()
    return avg_metrics
