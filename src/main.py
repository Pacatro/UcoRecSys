import pandas as pd
import json
from pathlib import Path
from argparse import ArgumentParser
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import db
from evaluation import cross_validate
import config
from dataset import ELearningDataModule
from engine import UcoRecSys
from models import NeuralHybrid

# HACER CROSS VALIDTAION CON SETS DE INTERACCIONES DE USUARIOS
# Cómo entreanr? --> Entrenamiento normal y corriente


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

    return df[features + [target]]


def inference(df: pd.DataFrame):
    dm = ELearningDataModule(
        df, target=config.TARGET, batch_size=config.BATCH_SIZE, balance=config.BALANCE
    )

    print(f"Dataset sparsity: {dm.sparsity}")

    dm.setup("fit")
    print(dm.train_dataset.df.shape, dm.val_dataset.df.shape)

    model = NeuralHybrid(
        n_users=dm.num_users,
        n_items=dm.num_items,
        numeric_features=dm.numeric_features,
        cat_cardinalities=dm.cat_cardinalities,
    )

    recsys = UcoRecSys(
        model=model,
        k=config.K,
        threshold=config.THRESHOLD,
    )

    early_stop = EarlyStopping(
        monitor="val/loss",
        patience=config.PATIENCE,
        mode="min",
        min_delta=config.DELTA,
        verbose=True,
    )

    checkpoint = ModelCheckpoint(
        monitor="val/loss", mode="min", save_top_k=1, filename="best-model"
    )

    trainer = L.Trainer(
        logger=TensorBoardLogger(
            name="ucorecsys", log_graph=True, save_dir="lightning_logs"
        ),
        max_epochs=config.EPOCHS,
        accelerator="auto",
        devices="auto",
        callbacks=[early_stop, checkpoint],
        log_every_n_steps=10,
        fast_dev_run=config.FAST_DEV_RUN,
    )

    dm.setup("fit")
    trainer.fit(recsys, datamodule=dm)

    if not config.FAST_DEV_RUN:
        recsys = UcoRecSys.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            model=model,
            k=config.K,
            threshold=config.THRESHOLD,
        )

    dm.setup("test")
    test_metrics = trainer.test(model=recsys, datamodule=dm)[0]
    with open("inference_results.json", "w") as f:
        json.dump(test_metrics.to_dict(), f)


def eval_model(df: pd.DataFrame):
    early_stop = EarlyStopping(
        monitor="val/loss",
        patience=config.PATIENCE,
        mode="min",
        min_delta=config.DELTA,
        verbose=False,
    )

    checkpoint = ModelCheckpoint(
        monitor="val/loss", mode="min", save_top_k=1, filename="best-model"
    )

    avg_metrics = cross_validate(
        df=df,
        model_class=NeuralHybrid,
        n_splits=5,
        random_state=42,
        epochs=config.EPOCHS,
        callbacks=[checkpoint, early_stop],
    )

    if avg_metrics is not None:
        with open("eval_results.json", "w") as f:
            json.dump(avg_metrics.to_dict(), f)


def main():
    if not Path(db.DB_FILE_PATH).exists():
        db.csv_to_sql(verbose=True)

    model_parser = ArgumentParser(prog="ucorecsys")

    model_parser.add_argument(
        "-i", "--inference", action="store_true", help="Run inference"
    )
    model_parser.add_argument(
        "-e", "--eval", action="store_true", help="Run evaluation"
    )

    args = model_parser.parse_args()

    print(f"Using batch size of {config.BATCH_SIZE}")
    print(f"Using {len(config.FEATURES)} features: {config.FEATURES}")
    print(f"Balance: {config.BALANCE}")
    print(f"k = {config.K}")
    print(f"Patience = {config.PATIENCE}, delta = {config.DELTA}")
    print(f"Threshold = {config.THRESHOLD}")
    print(f"Epochs = {config.EPOCHS}\n")

    df = load_data(features=config.FEATURES, target=config.TARGET)

    if args.inference:
        inference(df)
    elif args.eval:
        eval_model(df)


if __name__ == "__main__":
    main()
