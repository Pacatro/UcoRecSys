import os
import pandas as pd
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


def train_model(
    df: pd.DataFrame,
    dataset_name: str,
    target: str,
    epochs: int,
    lr: float,
    batch_size: int,
    balance: bool,
    top_k: int,
    output_model: str,
    ignored_cols: list[str] = [],
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
        print(f"[TRAIN] Dataset {dataset_name}:\n{dm.df}\n")
        print(f"[TRAIN] Dataset {dataset_name} sparsity: {dm.sparsity}")
        print(f"[TRAIN] Dataset {dataset_name} threshold: {dm.threshold}")
        print(f"[TRAIN] Train shape: {dm.train_dataset.df.shape}")
        print(f"[TRAIN] Val shape: {dm.val_dataset.df.shape}")
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
    # Copiar o renombrar según output_model
    Path(best_path).rename(output_model)
    if verbose:
        print(f"Modelo entrenado guardado en: {output_model}")


def inference(
    df: pd.DataFrame,
    model_path: str,
    target: str,
    batch_size: int,
    balance: bool,
    k: int = 10,
    ignored_cols: list[str] = [],
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
        print(f"[INFERENCE] Dataset sparsity: {dm.sparsity}")
        print(f"[INFERENCE] Dataset threshold: {dm.threshold}")

    recsys = UcoRecSys(
        model=model,
        k=k,
        threshold=dm.threshold,
    )

    # Cargar checkpoint proporcionado
    recsys = UcoRecSys.load_from_checkpoint(
        model_path,
        model=model,
        k=k,
        threshold=dm.threshold,
    )

    # Configurar y ejecutar prueba
    dm.setup("test")
    trainer = L.Trainer(
        logger=TensorBoardLogger(name="ucorecsys", save_dir="lightning_logs"),
        accelerator="auto",
        devices="auto",
    )

    trainer.test(model=recsys, datamodule=dm)


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


def main():
    if not Path(config.RESULTS_FOLDER).exists():
        os.mkdir(config.RESULTS_FOLDER)

    parser = build_parser()
    args = parser.parse_args()

    df = load_data(args.dataset)

    # Modo INFERENCE
    if args.inference:
        inference(
            df=df,
            dataset_name=args.dataset,
            model_path=args.inference,
            target=config.TARGET_COL,
            batch_size=args.batch_size,
            balance=args.balance,
            k=args.top_k,
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
            balance=args.balance,
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


if __name__ == "__main__":
    main()
