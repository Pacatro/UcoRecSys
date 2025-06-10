import os
import pandas as pd
import numpy as np
from pathlib import Path
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from typing import Literal
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

import config
from evaluation import cross_validate
from surprise_eval import cross_validation, preprocess_ratings
from datasets import load_data
from datamodule import ELearningDataModule
from engine import UcoRecSys
from model import NeuralHybrid
from args_parser import build_parser
from stats import friedman_test


def train_model(
    df: pd.DataFrame,
    dataset_name: str,
    target: str,
    epochs: int,
    lr: float,
    batch_size: int,
    top_k: int,
    output_model: str,
    ignored_cols: list[str] = [],
    balance: bool = False,
    plot: bool = False,
    verbose: bool = False,
):
    dm = ELearningDataModule(
        df,
        target=target,
        batch_size=batch_size,
        balance=balance,
        ignored_cols=ignored_cols,
    )

    dm.setup("fit")

    model = NeuralHybrid(
        n_users=dm.num_users,
        n_items=dm.num_items,
        cont_features=dm.cont_features,
        cat_cardinalities=dm.cat_cardinalities,
    )

    if verbose:
        # print(f"[TRAIN] Dataset {dataset_name}:\n{dm.df}\n")
        print(f"[TRAIN] Dataset {dataset_name} sparsity: {dm.sparsity}")
        print(f"[TRAIN] Dataset {dataset_name} threshold: {dm.threshold}")
        print(f"[TRAIN] Dataset {dataset_name} lenght: {len(dm.df)}")
        # print(f"[TRAIN] Train dataset:\n{dm.train_dataset.df}\n")
        print(f"[TRAIN] Model:\n{model}\n")

    recsys = UcoRecSys(
        model=model,
        k=top_k,
        threshold=dm.threshold,
        plot=plot,
        lr=lr,
    )

    early_stop = EarlyStopping(
        monitor="val/MSE",
        patience=config.PATIENCE,
        mode="min",
        min_delta=config.DELTA,
        verbose=True,
    )

    checkpoint = ModelCheckpoint(
        monitor="val/MSE", mode="min", save_top_k=1, filename="best-model"
    )

    trainer = L.Trainer(
        logger=TensorBoardLogger(name="ucorecsys", save_dir="lightning_logs"),
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[early_stop, checkpoint],
        log_every_n_steps=10,
    )

    trainer.fit(recsys, datamodule=dm)

    dm.setup("test")
    trainer.test(model=recsys, datamodule=dm)

    # Guardar ruta del mejor modelo
    best_path = checkpoint.best_model_path

    Path(best_path).rename(output_model)
    print(f"Modelo entrenado guardado en: {output_model}")


def generate_new_interactions(
    df: pd.DataFrame,
    samples: int = 130,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> pd.DataFrame:
    users = df["user_id"].unique()
    items = df["item_id"].unique()

    if samples == 0:
        return df

    rng = np.random.default_rng(seed=42)
    user_choice = rng.choice(users, size=samples * 2, replace=False)
    item_choice = rng.choice(items, size=samples * 2, replace=False)

    candidates = pd.DataFrame({"user_id": user_choice, "item_id": item_choice})

    positives = df[[user_col, item_col]].drop_duplicates()
    merged = candidates.merge(
        positives, on=[user_col, item_col], how="left", indicator=True
    )
    negatives = (
        merged[merged["_merge"] == "left_only"]
        .drop(columns="_merge")
        .drop_duplicates()
        .head(samples)
        .reset_index(drop=True)
    )

    # Add randomized values for extra columns
    item_types = ["tutorial", "use_case", "webcast"]
    difficulties = ["Beginner", "Intermediate", "Advanced", "Undefined"]
    negatives["item_type"] = rng.choice(item_types, size=len(negatives))
    negatives["difficulty"] = rng.choice(difficulties, size=len(negatives))
    negatives["nb_views"] = rng.integers(0, 2000, size=len(negatives)).astype(float)
    negatives["watch_percentage"] = rng.integers(0, 101, size=len(negatives)).astype(
        float
    )

    return negatives


def recommend(
    dm: ELearningDataModule, top_k: int, model_path: str, user_id: int | None = None
) -> pd.DataFrame:
    model = NeuralHybrid(
        n_users=dm.num_users,
        n_items=dm.num_items,
        cont_features=dm.cont_features,
        cat_cardinalities=dm.cat_cardinalities,
    )

    recsys = UcoRecSys.load_from_checkpoint(
        model_path, model=model, encoders=dm.encoders
    )

    dm.setup("predict")
    trainer = L.Trainer()
    predictions = trainer.predict(recsys, datamodule=dm)[0]

    preds_df = pd.DataFrame(predictions).sort_values(by=["prediction"], ascending=False)

    return (
        preds_df.head(top_k)
        if user_id is None
        else preds_df[preds_df["user_id"] == user_id].head(top_k)
    )


def inference(
    df: pd.DataFrame,
    model_path: str,
    target: str,
    batch_size: int,
    top_k: int,
    balance: bool = False,
    ignored_cols: list[str] = [],
    verbose: bool = False,
):
    predict_df = generate_new_interactions(df)
    if verbose:
        print(predict_df[predict_df["user_id"] == 564609])

    dm = ELearningDataModule(
        df,
        predict_df=predict_df,
        target=target,
        batch_size=batch_size,
        balance=balance,
        ignored_cols=ignored_cols,
    )

    user_preds = recommend(dm, top_k, model_path)
    print(user_preds)


def eval_model(
    df: pd.DataFrame,
    batch_size: int,
    dataset: str,
    k: int,
    epochs: int,
    n_splits: int,
    patience: int,
    delta: float,
    ignored_cols: list[str] = [],
    cv_type: Literal["kfold", "loo"] = "kfold",
    plot: bool = False,
    results_folder: str = "results",
    verbose: bool = False,
):
    avg_metrics = cross_validate(
        df=df,
        model_class=NeuralHybrid,
        n_splits=n_splits,
        random_state=42,
        epochs=epochs,
        cv_type=cv_type,
        batch_size=batch_size,
        k=k,
        patience=patience,
        delta=delta,
        ignored_cols=ignored_cols,
        plot=plot,
        verbose=verbose,
    )

    results_path = (
        f"{results_folder}/metrics_{cv_type}_k={n_splits}_{dataset}_top-{k}.csv"
    )
    avg_metrics.to_csv(results_path)

    if verbose:
        print(f"Resultados guardados en {results_path}")


def surprise_eval(
    df: pd.DataFrame,
    dataset: str,
    k: int,
    n_splits: int = 5,
    target: str = "rating",
    min_rating: int = 1,
    max_rating: int = 10,
    cv_type: Literal["kfold", "loo"] = "kfold",
    results_folder: str = "results",
    seeds: list[int] = [42],
):
    reader = Reader(rating_scale=(min_rating, max_rating))
    df = preprocess_ratings(df)
    data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)

    deterministic_algos = [
        NormalPredictor,
        BaselineOnly,
        KNNBasic,
        KNNWithMeans,
        KNNWithZScore,
        KNNBaseline,
        SlopeOne,
    ]

    stochastic_algos = [SVD, SVDpp, NMF, CoClustering]

    detailed_results = {}
    threshold = df[target].mean()

    for algo in deterministic_algos:
        print(f"Running {algo.__name__} {cv_type} cross validation")
        results = cross_validation(
            algo_class=algo,
            data=data,
            n_splits=n_splits,
            k=k,
            cv_type=cv_type,
            threshold=threshold,
        )
        detailed_results[algo.__name__] = results

    for algo in stochastic_algos:
        for random_state in seeds:
            print(
                f"Running {algo.__name__} (SEED: {random_state}) {cv_type} cross validation"
            )
            results = cross_validation(
                algo_class=algo,
                data=data,
                n_splits=n_splits,
                k=k,
                cv_type=cv_type,
                threshold=threshold,
                random_state=random_state,
            )
            detailed_results[f"{algo.__name__} (Seed: {random_state})"] = results

    combined_df = pd.DataFrame(
        {algo: results["Mean+-Std"] for algo, results in detailed_results.items()}
    )

    results_path = (
        f"{results_folder}/surprise_{cv_type}_k={n_splits}_{dataset}_top-{k}.csv"
    )

    combined_df.to_csv(results_path)

    print(f"Resultados guardados en {results_path}")


def get_stats_tests(top_k: int, verbose: bool = False):
    datasets = ["mars", "itm"]
    files = [
        f"./results/metrics_kfold_k=5_mars_top-{top_k}.csv",
        f"./results/metrics_kfold_k=5_itm_top-{top_k}.csv",
        f"./results/surprise_kfold_k=5_mars_top-{top_k}.csv",
        f"./results/surprise_kfold_k=5_itm_top-{top_k}.csv",
    ]

    models = [
        "NormalPredictor",
        "KNNBasic",
        "KNNWithMeans",
        "KNNWithZScore",
        "KNNBaseline",
        "SlopeOne",
        "SVD (Seed: 0)",
        "SVDpp (Seed: 0)",
        "NMF (Seed: 0)",
        "CoClustering (Seed: 0)",
        "SVD (Seed: 1)",
        "SVDpp (Seed: 1)",
        "NMF (Seed: 1)",
        "CoClustering (Seed: 1)",
        "SVD (Seed: 42)",
        "SVDpp (Seed: 42)",
        "NMF (Seed: 42)",
        "CoClustering (Seed: 42)",
        "Modelo Propuesto",
    ]

    stats_results = {dataset: {"p_value": 0, "stat": 0} for dataset in datasets}

    for dataset in datasets:
        stat, p = friedman_test(files, models, dataset, top_k, verbose=verbose)
        print(f"Results for dataset {dataset} in top-{top_k} are:")
        print(f"Stat: {stat}, p: {p}")
        stats_results[dataset]["p_value"] = p
        stats_results[dataset]["stat"] = stat

    stastics_path = f"{config.RESULTS_FOLDER}/stats/stats_{top_k}.csv"
    pd.DataFrame.from_dict(stats_results, orient="index").to_csv(stastics_path)


def main():
    if not Path(config.RESULTS_FOLDER).exists():
        os.mkdir(config.RESULTS_FOLDER)

    if not Path(f"{config.RESULTS_FOLDER}/stats").exists():
        os.mkdir(f"{config.RESULTS_FOLDER}/stats")

    parser = build_parser()
    args = parser.parse_args()

    df = load_data(args.dataset)

    # Modo INFERENCE
    if args.inference:
        inference(
            df=df,
            model_path=args.inference,
            target=config.TARGET_COL,
            batch_size=args.batch_size,
            top_k=args.top_k,
            verbose=args.verbose,
        )
    # Modo TRAIN
    elif args.train:
        train_model(
            df=df,
            dataset_name=args.dataset,
            target=config.TARGET_COL,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            top_k=args.top_k,
            output_model=args.output_model,
            plot=args.plot,
            verbose=args.verbose,
        )
    # Modo EVAL
    elif args.eval:
        eval_model(
            df=df,
            epochs=args.epochs,
            n_splits=args.k_splits,
            delta=config.DELTA,
            patience=config.PATIENCE,
            batch_size=args.batch_size,
            dataset=args.dataset,
            k=args.top_k,
            cv_type=args.cvtype,
            plot=args.plot,
            results_folder=config.RESULTS_FOLDER,
            verbose=args.verbose,
        )

    # Modo SURPRISE
    if args.surprise:
        surprise_eval(
            df=df,
            dataset=args.dataset,
            cv_type=args.cvtype,
            n_splits=args.k_splits,
            min_rating=df[config.TARGET_COL].min(),
            max_rating=df[config.TARGET_COL].max(),
            k=args.top_k,
            results_folder=config.RESULTS_FOLDER,
            target=config.TARGET_COL,
            seeds=args.seeds,
        )

    # Modo Test estadísticos
    if args.stats_test:
        get_stats_tests(top_k=args.top_k, verbose=args.verbose)


if __name__ == "__main__":
    main()
