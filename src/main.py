import pandas as pd
import json
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
from torch import nn

import db
from engine import UcoRecSys
from models import GMFMLP, DeepHybridModel, NeuralMF
from dataset import ELearningDataModule
from model_eval import cross_validate_loo
from config import (
    EPOCHS,
    BATCH_SIZE,
    DELTA,
    PATIENCE,
    FAST_DEV_RUN,
    BINARIZE,
    BALANCE,
    K,
    THRESHOLD,
    FEATURES,
    TARGET,
)


def load_data(features: list[str], target: str) -> pd.DataFrame:
    explicit_df_en = pd.read_csv("./data/explicit_ratings_en.csv")
    explicit_df_fr = pd.read_csv("./data/explicit_ratings_fr.csv")

    items_en = pd.read_csv("./data/items_en.csv")
    items_fr = pd.read_csv("./data/items_fr.csv")

    df_explicit = pd.concat([explicit_df_en, explicit_df_fr], ignore_index=True)
    df_items = pd.concat([items_en, items_fr], ignore_index=True)

    df_explicit["created_at"] = pd.to_datetime(df_explicit["created_at"])
    df_items = df_items.drop(columns=["created_at"])

    df = pd.merge(df_explicit, df_items, on="item_id", how="inner")

    df["Difficulty"] = df["Difficulty"].fillna("Undefined").astype("category")
    df["type"] = df["type"].fillna("Undefined").astype("category")

    df.rename(
        columns={"Difficulty": "difficulty", "type": "item_type"},
        inplace=True,
    )

    df = df.sort_values(by="created_at", ascending=False)

    return df[features + [target]]


def get_test_predictions(
    trainer: L.Trainer,
    model: UcoRecSys,
    dm: L.LightningDataModule,
    threshold: float,
) -> pd.DataFrame:
    raw_outputs = trainer.predict(model, datamodule=dm)

    flat = []
    for batch_out in raw_outputs:
        B = batch_out["user_id"].size(0)
        for i in range(B):
            flat.append(
                {
                    "user_id": int(batch_out["user_id"][i].item()),
                    "item_id": int(batch_out["item_id"][i].item()),
                    "prediction": float(batch_out["prediction"][i].item()),
                    "rating": float(batch_out["rating"][i].item()),
                }
            )

    df = pd.DataFrame(flat)

    df["target"] = (df["rating"] >= threshold).astype(int)
    df["pred_target"] = (df["prediction"] >= threshold).astype(int)

    return df


def eval_model(models: list[nn.Module], dm: L.LightningDataModule):
    for model in models:
        recsys = UcoRecSys(
            model=model,
            k=K,
            threshold=THRESHOLD,
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            mode="min",
            min_delta=DELTA,
            verbose=True,
        )

        checkpoint = ModelCheckpoint(
            monitor="val_loss", mode="min", save_top_k=1, filename="best-model"
        )

        trainer = L.Trainer(
            max_epochs=EPOCHS,
            accelerator="auto",
            devices="auto",
            callbacks=[early_stop, checkpoint],
            log_every_n_steps=10,
            fast_dev_run=FAST_DEV_RUN,
        )

        trainer.fit(recsys, datamodule=dm)

        if not FAST_DEV_RUN:
            recsys = UcoRecSys.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path,
                model=model,
                k=K,
                threshold=THRESHOLD,
            )

        trainer.test(model=recsys, datamodule=dm)
        df = get_test_predictions(trainer, recsys, dm, THRESHOLD)

        image_path = (
            f"ratings_vs_predictions_{model.__class__.__name__}_{dm.num_features}"
        )
        image_path = (image_path if not BALANCE else image_path + "_balanced") + ".png"
        ax = df.plot(kind="scatter", x="rating", y="prediction", s=32, alpha=0.8)
        fig = ax.get_figure()
        fig.savefig(image_path, dpi=300, bbox_inches="tight")


def main():
    if not Path(db.DB_FILE_PATH).exists():
        db.csv_to_sql(verbose=True)

    print(f"Using batch size of {BATCH_SIZE}")
    print(f"Using {len(FEATURES)} features: {FEATURES}")
    print(f"k = {K}")
    print(f"Patience = {PATIENCE}, delta = {DELTA}")
    print(f"Threshold = {THRESHOLD}")
    print(f"Epochs = {EPOCHS}")
    print(f"Balance: {BALANCE}")
    print(f"Binarize: {BINARIZE}\n")

    df = load_data(features=FEATURES, target=TARGET)

    dm = ELearningDataModule(df, target=TARGET, batch_size=BATCH_SIZE, balance=BALANCE)
    print(f"Dataset sparsity: {dm.sparsity}")
    dm.setup("fit")
    print(dm.train_dataset.df.shape, dm.val_dataset.df.shape)
    print(dm.train_dataset.df)

    model = NeuralMF(
        n_users=dm.num_users,
        n_items=dm.num_items,
        numeric_features=dm.numeric_features,
        cat_cardinalities=dm.cat_cardinalities,
    )

    recsys = UcoRecSys(
        model=model,
        k=K,
        threshold=THRESHOLD,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        mode="min",
        min_delta=DELTA,
        verbose=True,
    )

    checkpoint = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1, filename="best-model"
    )

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices="auto",
        callbacks=[early_stop, checkpoint],
        log_every_n_steps=10,
        fast_dev_run=FAST_DEV_RUN,
    )

    trainer.fit(recsys, datamodule=dm)

    if not FAST_DEV_RUN:
        recsys = UcoRecSys.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            model=model,
            k=K,
            threshold=THRESHOLD,
        )

    dm.setup("test")
    trainer.test(model=recsys, datamodule=dm)


if __name__ == "__main__":
    main()
