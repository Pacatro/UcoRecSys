import pandas as pd
import numpy as np
from pathlib import Path
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from typing import Literal
from sklearn.preprocessing import MinMaxScaler
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
import config
from evaluation import cross_validate
from surprise_eval import cross_validation, preprocess_ratings
from dataset import ELearningDataModule
from engine import UcoRecSys
from models import NeuralHybrid
from args_parser import model_parser


def load_mars() -> pd.DataFrame:
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

    features = [
        "user_id",
        "item_id",
        "item_type",
        "difficulty",
        "nb_views",
        "watch_percentage",
        "rating",
    ]

    return df[features]


def load_itm() -> pd.DataFrame:
    ratings_df = pd.read_csv("./data/itm_dataset/ratings.csv")
    items_df = pd.read_csv("./data/itm_dataset/items.csv")
    merged_df = pd.merge(left=items_df, right=ratings_df, how="inner", on="Item")
    merged_df = merged_df.rename(
        columns={"UserID": "user_id", "Item": "item_id", "Rating": "rating"}
    )
    merged_df["Class"] = merged_df["Class"].astype("category")
    merged_df["Semester"] = merged_df["Semester"].astype("category")
    merged_df["Lockdown"] = merged_df["Lockdown"].astype("category")
    merged_df["Title"] = merged_df["Title"].astype("category")

    features = [
        "user_id",
        "item_id",
        "Title",
        "Semester",
        "Class",
        "App",
        "Lockdown",
        "Ease",
        "rating",
    ]

    return merged_df[features]


def load_coursera() -> pd.DataFrame:
    df = pd.read_csv("./data/coursera_dataset/Coursera.csv")
    num_users = 2000
    num_interactions = 20000
    user_ids = np.random.randint(1, num_users + 1, size=num_interactions)
    course_titles = np.random.choice(df["Course Name"], size=num_interactions)
    interaction_types = np.random.choice(
        ["view", "enroll", "complete", "rate"],
        size=num_interactions,
        p=[0.4, 0.3, 0.2, 0.1],
    )

    # Asegurar que todos tengan un rating (aunque no sea realista para algunas interacciones)
    ratings = np.random.uniform(1, 5, size=num_interactions)

    interactions = pd.DataFrame(
        {
            "user_id": user_ids,
            "Course Name": course_titles,
            "interaction_type": interaction_types,
            "rating": ratings,
        }
    )

    df_merged = interactions.merge(df, on="Course Name", how="left", indicator=True)
    df_merged = df_merged[df_merged["_merge"] == "both"].drop(columns=["_merge"])

    df_merged = df_merged.rename(columns={"Course Name": "item_id"})
    df_merged["item_id"] = df_merged["item_id"].astype("category")
    df_merged["user_id"] = df_merged["user_id"].astype("category")
    df_merged["Difficulty Level"] = df_merged["Difficulty Level"].astype("category")
    df_merged["University"] = df_merged["University"].astype("category")

    features = [
        "user_id",
        "item_id",
        "Difficulty Level",
        "University",
        "rating",
    ]

    return df_merged[features]


def load_doris() -> pd.DataFrame:
    course_info = pd.read_csv("./data/doris_dataset/CourseInformationTable.csv")
    course_selection = pd.read_csv("./data/doris_dataset/CourseSelectionTable.csv")
    merged_df = pd.merge(
        left=course_info, right=course_selection, how="inner", on="CourseId"
    )
    merged_df.rename(
        columns={"CourseId": "item_id", "StudentId": "user_id", "Score": "rating"},
        inplace=True,
    )
    merged_df["rating"] = merged_df["rating"].fillna(0)
    merged_df["rating"] = MinMaxScaler(feature_range=(0, 10)).fit_transform(
        merged_df[["rating"]]
    )
    merged_df["Type"] = merged_df["Type"].fillna("Undefined").astype("category")
    merged_df["CourseCollege"] = (
        merged_df["CourseCollege"].fillna("Undefined").astype("category")
    )
    features = ["user_id", "item_id", "Type", "rating"]
    return merged_df[features]


def load_data(
    dataset_name: Literal["mars", "itm", "coursera", "doris"],
) -> pd.DataFrame:
    match dataset_name:
        case "mars":
            return load_mars()
        case "itm":
            return load_itm()
        case "coursera":
            return load_coursera()
        case "doris":
            return load_doris()


def inference(
    df: pd.DataFrame,
    dataset_name: str,
    target: str,
    batch_size: int,
    balance: bool,
    k: int = 10,
    verbose: bool = False,
):
    dm = ELearningDataModule(
        df,
        target=target,
        batch_size=batch_size,
        balance=balance,
    )

    dm.setup("fit")

    model = NeuralHybrid(
        n_users=dm.num_users,
        n_items=dm.num_items,
        cont_features=dm.cont_features,
        cat_cardinalities=dm.cat_cardinalities,
    )

    if verbose:
        print(f"Dataset sparsity: {dm.sparsity}")
        print(f"Dataset threshold: {dm.threshold}")
        print(dm.train_dataset.df)
        print(model)

    recsys = UcoRecSys(
        model=model,
        k=k,
        threshold=dm.threshold,
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

    recsys = UcoRecSys.load_from_checkpoint(
        checkpoint.best_model_path,
        model=model,
        k=k,
        threshold=dm.threshold,
    )

    dm.setup("test")
    test_metrics = trainer.test(model=recsys, datamodule=dm)
    pd.DataFrame(test_metrics).to_csv(f"inference_{dataset_name}_results_k={k}.csv")


def eval_model(
    df: pd.DataFrame,
    batch_size: int,
    dataset: str,
    k: int,
    cv_type: Literal["kfold", "loo"] = "kfold",
    verbose: bool = False,
):
    avg_metrics = cross_validate(
        df=df,
        model_class=NeuralHybrid,
        n_splits=5,
        random_state=42,
        epochs=config.EPOCHS,
        cv_type=cv_type,
        batch_size=batch_size,
        k=k,
        patience=config.PATIENCE,
        delta=config.DELTA,
        verbose=verbose,
    )

    if avg_metrics is not None:
        avg_metrics.to_csv(f"{cv_type}_eval_{dataset}_results_k={k}.csv")


def surprise_eval(
    df: pd.DataFrame,
    dataset: str,
    k: int,
    target: str = "rating",
    min_rating: int = 1,
    max_rating: int = 10,
    cv_type: Literal["kfold", "loo"] = "kfold",
):
    reader = Reader(rating_scale=(min_rating, max_rating))
    df = preprocess_ratings(df)
    data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)

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
            k=k,
            cv_type=cv_type,
            threshold=df[target].mean(),
        )
        algos_metrics[algo.__name__] = results

    pd.DataFrame(algos_metrics).to_csv(
        f"surprise_{dataset}_{cv_type}_metrics_k={k}.csv"
    )


def main():
    if not Path(db.DB_FILE_PATH).exists():
        db.csv_to_sql(verbose=True)

    args = model_parser.parse_args()

    print(f"Using {args.dataset} dataset")
    print(f"Balance: {config.BALANCE}")
    print(f"Patience = {config.PATIENCE}, delta = {config.DELTA}")

    df = load_data(args.dataset)
    batch_size = config.BATCH_SIZE if args.dataset in ["mars", "doris"] else 32
    print(f"Batch size: {batch_size}")

    if args.inference:
        inference(
            df,
            target=config.TARGET,
            dataset_name=args.dataset,
            batch_size=batch_size,
            balance=config.BALANCE,
            k=config.K,
            verbose=args.verbose,
        )
    elif args.eval:
        print("CV type:", args.cvtype)
        eval_model(df, batch_size, args.dataset, config.K, args.cvtype, args.verbose)
    else:
        print(df)

    if args.surprise:
        surprise_eval(
            df,
            dataset=args.dataset,
            cv_type=args.cvtype,
            min_rating=df[config.TARGET].min(),
            max_rating=df[config.TARGET].max(),
            k=config.K,
        )


if __name__ == "__main__":
    main()
