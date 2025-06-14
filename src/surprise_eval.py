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
# from torchmetrics.regression import R2Score, ExplainedVariance

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
    preds: list[Prediction],
    k: int = 10,
    threshold: float = 8,
) -> dict[str, float]:
    # Extract prediction data
    indexes = torch.tensor([pred.uid for pred in preds])
    predictions = torch.tensor([pred.est for pred in preds])
    target = torch.tensor([pred.r_ui >= threshold for pred in preds])

    # Ranking metrics
    ranking_metrics = MetricCollection(
        {
            f"Precision@{k}": RetrievalPrecision(top_k=k, adaptive_k=True),
            f"Recall@{k}": RetrievalRecall(top_k=k),
            f"F1@{k}": RetrievalFBetaScore(top_k=k, beta=1.0, adaptive_k=True),
            f"NDCG@{k}": RetrievalNormalizedDCG(top_k=k),
            f"HitRate@{k}": RetrievalHitRate(top_k=k),
            f"MAP@{k}": RetrievalMAP(top_k=k),
            f"MRR@{k}": RetrievalMRR(top_k=k),
        }
    )

    ranking_metrics.update(predictions, target, indexes=indexes)

    ranking_results = {k: v.item() for k, v in ranking_metrics.compute().items()}

    # Surprise library metrics
    rmse = float(accuracy.rmse(preds, verbose=False))
    mse = float(accuracy.mse(preds, verbose=False))

    # Clear stateful metrics
    ranking_metrics.reset()

    return {
        "MSE": mse,
        "RMSE": rmse,
        **ranking_results,
        # **prediction_results,
    }


def cross_validation(
    algo_class: Callable[..., AlgoBase],
    data: Dataset,
    n_splits: int = 5,
    k: int = 10,
    threshold: float = 8.0,
    cv_type: Literal["kfold", "loo"] | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    cv_type = cv_type if cv_type is not None else "kfold"

    cv = (
        KFold(n_splits=n_splits, shuffle=True, random_state=42)
        if cv_type == "kfold"
        else LeaveOneOut(n_splits=n_splits)
    )

    n_folds = cv.get_n_folds()
    fold_metrics = []

    for fold, (trainset, testset) in enumerate(cv.split(data), start=1):
        print(f"Fold {fold}/{n_folds}")
        algo = (
            algo_class()
            if random_state is None
            else algo_class(random_state=random_state)
        )
        algo.fit(trainset)
        preds = algo.test(testset)
        metrics = calc_metrics(preds, k=k, threshold=threshold)
        fold_metrics.append(metrics)

    fold_metrics = pd.DataFrame(fold_metrics)

    avg_metrics = fold_metrics.mean()
    std_metrics = fold_metrics.std()

    results = pd.DataFrame(
        {
            "Mean": avg_metrics,
            "Std": std_metrics,
            "Mean+-Std": [
                f"{mean:.4f}/{std:.4f}" for mean, std in zip(avg_metrics, std_metrics)
            ],
        }
    )

    return results
