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

import db
import config
from evaluation import cross_validate
from surprise_eval import cross_validation, preprocess_ratings
from datasets import load_data
from datamodule import ELearningDataModule
from engine import UcoRecSys
from models import NeuralHybrid
from args_parser import model_parser


def inference(
    df: pd.DataFrame,
    dataset_name: str,
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
    ignored_cols: list[str] = [],
    cv_type: Literal["kfold", "loo"] = "kfold",
    verbose: bool = False,
):
    avg_metrics = cross_validate(
        df=df,
        model_class=NeuralHybrid,
        n_splits=config.K_FOLD,
        random_state=42,
        epochs=config.EPOCHS,
        cv_type=cv_type,
        batch_size=batch_size,
        k=k,
        patience=config.PATIENCE,
        delta=config.DELTA,
        ignored_cols=ignored_cols,
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

    ignored_cols = ["Semester", "Grade"] if args.dataset == "doris" else []

    if args.inference:
        inference(
            df,
            target=config.TARGET,
            dataset_name=args.dataset,
            batch_size=batch_size,
            balance=config.BALANCE,
            k=config.K,
            ignored_cols=ignored_cols,
            verbose=args.verbose,
        )
    elif args.eval:
        print("CV type:", args.cvtype)
        eval_model(
            df=df,
            batch_size=batch_size,
            dataset=args.dataset,
            k=config.K,
            cv_type=args.cvtype,
            ignored_cols=ignored_cols,
            verbose=args.verbose,
        )
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
