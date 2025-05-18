import json
import pandas as pd
import torch
from pathlib import Path
from surprise.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from torchmetrics import MetricCollection
from typing import Callable
from surprise import (
    Reader,
    Dataset,
    accuracy,
    AlgoBase,
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
    Prediction,
)
from torchmetrics.retrieval import (
    RetrievalPrecision,
    RetrievalRecall,
    RetrievalNormalizedDCG,
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
)

import db
import config


def preprocess_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica preprocesamiento únicamente a la columna 'rating',
    dejando intactos los identificadores.
    """
    le_user = LabelEncoder()
    le_course = LabelEncoder()

    df.loc[:, "user_id"] = le_user.fit_transform(df.user_id.values)
    df.loc[:, "item_id"] = le_course.fit_transform(df.item_id.values)
    return df


def calc_metrics(
    preds: list[Prediction], k: int = 10, threshold: float = 8
) -> dict[str, float]:
    indexes = torch.tensor([pred.uid for pred in preds])
    predictions = torch.tensor([pred.est for pred in preds])
    target = torch.tensor([pred.r_ui >= threshold for pred in preds])

    metrics = MetricCollection(
        RetrievalPrecision(top_k=k, adaptive_k=True),
        RetrievalRecall(top_k=k),
        RetrievalNormalizedDCG(top_k=k),
        RetrievalHitRate(top_k=k),
        RetrievalMAP(top_k=k),
        RetrievalMRR(top_k=k),
    )

    metrics.update(predictions, target, indexes=indexes)
    metrics_results = {k: v.item() for k, v in metrics.compute().items()}
    rmse = accuracy.rmse(preds, verbose=False)
    mse = accuracy.mse(preds, verbose=False)

    return {
        "rmse": rmse,
        "mse": mse,
        **metrics_results,
    }


def cross_validation(
    algo_class: Callable[..., AlgoBase],
    data: Dataset,
    n_splits: int = 5,
    k: int = 10,
    threshold: float = 8.0,
    verbose: bool = False,
) -> dict[str, float]:
    print(f"Running {algo_class.__name__} cross validation")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []
    for fold, (trainset, testset) in enumerate(kf.split(data), start=1):
        print(f"Fold {fold}/{n_splits}")
        algo = algo_class()
        algo.fit(trainset)
        preds = algo.test(testset)
        metrics = calc_metrics(preds, k=k, threshold=threshold)
        if verbose:
            print(metrics)
        fold_metrics.append(metrics)

    avg_metrics = pd.DataFrame(fold_metrics).mean()
    return avg_metrics.to_dict()


def main():
    if not Path(db.DB_FILE_PATH).exists():
        db.csv_to_sql(verbose=True)

    df_explicit_ratings_en = pd.read_csv("data/explicit_ratings_en.csv")
    df_explicit_ratings_fr = pd.read_csv("data/explicit_ratings_fr.csv")
    df_explicit_ratings = pd.concat([df_explicit_ratings_en, df_explicit_ratings_fr])

    # Seleccionar únicamente las columnas relevantes
    final_df = df_explicit_ratings[["user_id", "item_id", "rating"]]
    final_df = preprocess_ratings(final_df)

    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(final_df, reader)

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
        results = cross_validation(algo_class=algo, data=data, n_splits=5, k=config.K)
        algos_metrics[algo.__name__] = results

    with open("surprise_metrics.json", "w") as f:
        json.dump(algos_metrics, f, indent=2)


if __name__ == "__main__":
    main()
