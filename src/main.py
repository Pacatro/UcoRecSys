import pandas as pd
import json
from pathlib import Path
from argparse import ArgumentParser
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from typing import Literal
from sklearn.impute import SimpleImputer
from surprise import (
    Reader,
    Dataset,
    SVDpp,
    NormalPredictor,
    BaselineOnly,
    KNNBasic,
    KNNWithMeans,
    KNNWithZScore,
    KNNBaseline,
    SVD,
    NMF,
    SlopeOne,
    CoClustering,
)

import db
from evaluation import cross_validate
from surprise_eval import cross_validation, preprocess_ratings
import config
from dataset import ELearningDataModule
from engine import UcoRecSys
from models import NeuralHybrid


def load_mars(features: list[str], target: str) -> pd.DataFrame:
    explicit_df_en = pd.read_csv("./data/mars_dataset/explicit_ratings_en.csv")
    explicit_df_fr = pd.read_csv("./data/mars_dataset/explicit_ratings_fr.csv")

    items_en = pd.read_csv("./data/mars_dataset/items_en.csv")
    items_fr = pd.read_csv("./data/mars_dataset/items_fr.csv")

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


def load_doris(features: list[str], target: str) -> pd.DataFrame:
    course_info = pd.read_excel("./data/doris_dataset/CourseInformationTable.xlsx")
    course_selection = pd.read_excel("./data/doris_dataset/CourseSelectionTable.xlsx")
    student_info = pd.read_excel("./data/doris_dataset/StudentInformationTable.xlsx")

    # Rename columns
    course_info = course_info.rename(columns={"CourseId": "item_id"})
    student_info = student_info.rename(columns={"StudentId": "user_id"})
    course_selection = course_selection.rename(
        columns={"StudedntId": "user_id", "Score": "rating", "CourseId": "item_id"}
    )

    merged_df = pd.merge(course_selection, student_info, on="user_id", how="left")
    df = pd.merge(merged_df, course_info, on="item_id", how="left")

    imp = SimpleImputer(strategy="mean")
    df[target] = imp.fit_transform(df[target].values.reshape(-1, 1))
    return df[features + [target]]


def load_itm(features: list[str], target: str) -> pd.DataFrame:
    ratings_df = pd.read_csv("./data/itm_dataset/ratings.csv")
    ratings_df = ratings_df.rename(
        columns={"UserID": "user_id", "Item": "item_id", "Rating": "rating"}
    )
    return ratings_df[features + [target]]


def load_data(
    dataset_name: Literal["mars", "doris", "itm"], features: list[str], target: str
) -> pd.DataFrame:
    match dataset_name:
        case "mars":
            return load_mars(features, target)
        case "doris":
            return load_doris(features, target)
        case "itm":
            return load_itm(features, target)


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
        cont_features=dm.cont_features,
        cat_cardinalities=dm.cat_cardinalities,
    )

    print(model)

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
        logger=TensorBoardLogger(name="ucorecsys", save_dir="lightning_logs"),
        max_epochs=config.EPOCHS,
        accelerator="auto",
        devices="auto",
        callbacks=[early_stop, checkpoint],
        log_every_n_steps=10,
        fast_dev_run=config.FAST_DEV_RUN,
    )

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
        json.dump(test_metrics, f, indent=2)


def eval_model(df: pd.DataFrame, cv_type: Literal["kfold", "loo"] = "kfold"):
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
        cv_type=cv_type,
    )

    if avg_metrics is not None:
        with open(f"{cv_type}_eval_results.json", "w") as f:
            json.dump(avg_metrics.to_dict(), f, indent=2)


def surprise_eval(
    df: pd.DataFrame,
    min_rating: int = 1,
    max_rating: int = 10,
    cv_type: Literal["kfold", "loo"] = "kfold",
):
    reader = Reader(rating_scale=(min_rating, max_rating))
    df = preprocess_ratings(df)
    data = Dataset.load_from_df(df, reader)

    algos = [
        SVDpp,
        NormalPredictor,
        BaselineOnly,
        KNNBasic,
        KNNWithMeans,
        KNNWithZScore,
        KNNBaseline,
        SVD,
        NMF,
        SlopeOne,
        CoClustering,
    ]

    algos_metrics = {}
    for algo in algos:
        results = cross_validation(
            algo_class=algo,
            data=data,
            n_splits=5,
            k=config.K,
            cv_type=cv_type,
            epochs=config.EPOCHS,
        )
        algos_metrics[algo.__name__] = results

    with open("surprise_metrics.json", "w") as f:
        json.dump(algos_metrics, f, indent=2)


def main():
    if not Path(db.DB_FILE_PATH).exists():
        db.csv_to_sql(verbose=True)

    model_parser = ArgumentParser(prog="ucorecsys")

    model_parser.add_argument(
        "-i", "--inference", action="store_true", help="Run inference"
    )
    model_parser.add_argument(
        "-e",
        "--eval",
        action="store",
        help="Evaluate the proposed model, if -s is activate, then performs the type of evaluation for surprise algorithms",
        choices=["kfold", "loo"],
        default="kfold",
    )
    model_parser.add_argument(
        "-s",
        "--surprise",
        action="store_true",
        help="Evaluate the surprise algorithms",
    )
    model_parser.add_argument(
        "-ds",
        "--dataset",
        action="store",
        help="Name of the dataset to load",
        choices=["mars", "doris", "itm"],
        default="mars",
    )

    args = model_parser.parse_args()

    print(f"Using batch size of {config.BATCH_SIZE}")
    print(f"Using {args.dataset} dataset")
    print(f"Using {len(config.FEATURES)} features: {config.FEATURES}")
    print(f"Balance: {config.BALANCE}")
    print(f"k = {config.K}")
    print(f"Patience = {config.PATIENCE}, delta = {config.DELTA}")
    print(f"Threshold = {config.THRESHOLD}")
    print(f"Epochs = {config.EPOCHS}\n")

    df = load_data(args.dataset, features=config.FEATURES, target=config.TARGET)

    if args.inference:
        inference(df)
    elif args.surprise:
        surprise_eval(
            df,
            cv_type=args.eval,
            min_rating=df[config.TARGET].min(),
            max_rating=df[config.TARGET].max(),
        )
    elif args.eval:
        print("Eval mode:", args.eval)
        eval_model(df, args.eval)
    else:
        print(df)


if __name__ == "__main__":
    main()
