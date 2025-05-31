import pandas as pd
import torch
from typing import Literal, Callable
from surprise.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from torchmetrics import MetricCollection
from torchmetrics.retrieval import (
    RetrievalPrecision,
    RetrievalRecall,
    RetrievalNormalizedDCG,
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
)
from surprise import (
    Dataset,
    accuracy,
    AlgoBase,
    Prediction,
)

from engine import RetrievalFBetaScore


def preprocess_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica preprocesamiento únicamente a la columna 'rating',
    dejando intactos los identificadores.
    """
    le_user = LabelEncoder()
    le_course = LabelEncoder()
    df["user_id"] = df["user_id"].astype("category")
    df["item_id"] = df["item_id"].astype("category")

    df["user_id"] = le_user.fit_transform(df["user_id"])
    df["item_id"] = le_course.fit_transform(df["item_id"])
    return df


def calc_metrics(
    preds: list[Prediction], k: int = 10, threshold: float = 8
) -> dict[str, float]:
    indexes = torch.tensor([pred.uid for pred in preds])
    predictions = torch.tensor([pred.est for pred in preds])
    target = torch.tensor([pred.r_ui >= threshold for pred in preds])

    metrics = MetricCollection(
        {
            f"Precison@{k}": RetrievalPrecision(top_k=k, adaptive_k=True),
            f"Recall@{k}": RetrievalRecall(top_k=k),
            f"F1@{k}": RetrievalFBetaScore(top_k=k, beta=1.0, adaptive_k=True),
            f"NDCG@{k}": RetrievalNormalizedDCG(top_k=k),
            f"HitRate@{k}": RetrievalHitRate(top_k=k),
            f"MAP@{k}": RetrievalMAP(top_k=k),
            f"MRR@{k}": RetrievalMRR(top_k=k),
        }
    )

    metrics.update(predictions, target, indexes=indexes)
    metrics_results = {k: v.item() for k, v in metrics.compute().items()}
    rmse = accuracy.rmse(preds, verbose=False)
    mse = accuracy.mse(preds, verbose=False)
    metrics.reset()

    return {
        "rmse": float(rmse),
        "mse": float(mse),
        **metrics_results,
    }


def cross_validation(
    algo_class: Callable[..., AlgoBase],
    data: Dataset,
    n_splits: int = 5,
    k: int = 10,
    threshold: float = 8.0,
    cv_type: Literal["kfold", "loo"] | None = None,
) -> dict[str, float]:
    cv_type = cv_type if not None else "kfold"
    print(f"Running {algo_class.__name__} cross validation ({cv_type})")
    cv = (
        KFold(n_splits=n_splits, shuffle=True, random_state=42)
        if cv_type == "kfold"
        else LeaveOneOut(n_splits=n_splits)
    )
    n_folds = cv.get_n_folds()
    fold_metrics = []
    for fold, (trainset, testset) in enumerate(cv.split(data), start=1):
        print(f"Fold {fold}/{n_folds}")
        algo = algo_class()
        algo.fit(trainset)
        preds = algo.test(testset)
        metrics = calc_metrics(preds, k=k, threshold=threshold)
        fold_metrics.append(metrics)

    avg_metrics = pd.DataFrame(fold_metrics).mean()
    return avg_metrics.to_dict()
