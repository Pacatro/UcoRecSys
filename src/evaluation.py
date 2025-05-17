import pandas as pd
import lightning as L
from sklearn.model_selection import KFold, LeaveOneOut
from typing import Literal

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
) -> pd.DataFrame:
    cv = (
        KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        if cv_type == "kfold"
        else LeaveOneOut()
    )
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(df), start=1):
        print(f"Fold {fold}/{n_splits}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[test_idx].reset_index(drop=True)

        dm = ELearningDataModule(
            df=pd.concat([train_df, val_df], ignore_index=True),
            test_size=0,
            val_size=len(val_df) / (len(train_df) + len(val_df)),
        )
        dm.setup("fit")

        model = model_class(
            cat_cardinalities=dm.cat_cardinalities,
            numeric_features=dm.numeric_features,
            n_users=dm.num_users,
            n_items=dm.num_items,
        )

        recsys = UcoRecSys(model=model)

        trainer = L.Trainer(
            max_epochs=epochs,
            accelerator="auto",
            devices="auto",
            callbacks=callbacks,
            log_every_n_steps=10,
            enable_model_summary=False,
            inference_mode=False,
            enable_progress_bar=False,
        )

        trainer.fit(recsys, datamodule=dm)
        recsys = UcoRecSys.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            model=model,
        )
        metrics = trainer.validate(recsys, datamodule=dm)
        print(metrics[0])
        fold_metrics.append(metrics[0])

    avg_metrics = pd.DataFrame(fold_metrics).mean()
    return avg_metrics
