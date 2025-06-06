import pandas as pd
import lightning as L
from sklearn.model_selection import KFold, LeaveOneOut
from typing import Literal
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from datamodule import ELearningDataModule
from engine import UcoRecSys


def cross_validate(
    df: pd.DataFrame,
    model_class: type,
    lr: float = 0.001,
    n_splits: int = 5,
    random_state: int = 42,
    epochs: int = 100,
    cv_type: Literal["kfold", "loo"] = "kfold",
    k: int = 10,
    batch_size: int = 128,
    patience: int = 5,
    delta: float = 0.001,
    ignored_cols: list[str] = [],
    plot: bool = False,
    verbose: bool = False,
) -> pd.Series:
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
            ignored_cols=ignored_cols,
        )
        dm.setup("fit")

        model = model_class(
            cat_cardinalities=dm.cat_cardinalities,
            cont_features=dm.cont_features,
            n_users=dm.num_users,
            n_items=dm.num_items,
        )

        recsys = UcoRecSys(model=model, threshold=dm.threshold, plot=plot, lr=lr)

        earlystop = EarlyStopping(
            monitor="val/MSE",
            patience=patience,
            mode="min",
            min_delta=delta,
            verbose=verbose,
        )
        ckpt = ModelCheckpoint(
            monitor="val/MSE", mode="min", save_top_k=1, filename="best-model"
        )

        trainer = L.Trainer(
            max_epochs=epochs,
            accelerator="auto",
            devices="auto",
            callbacks=[earlystop, ckpt],
            log_every_n_steps=10,
            enable_model_summary=False,
            inference_mode=False,
            enable_progress_bar=verbose,
        )

        trainer.fit(recsys, datamodule=dm)

        recsys = UcoRecSys.load_from_checkpoint(
            ckpt.best_model_path,
            model=model,
            k=k,
            threshold=dm.threshold,
        )

        metrics = trainer.validate(recsys, datamodule=dm)[0]

        fold_metrics.append(metrics)

    all_metrics = pd.DataFrame(fold_metrics)
    avg_metrics = all_metrics.mean()
    std_metrics = all_metrics.std()

    result = pd.DataFrame({"mean": avg_metrics, "std": std_metrics})
    return result
