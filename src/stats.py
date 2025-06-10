import pandas as pd
from scipy import stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt

import config


def extract_metrics_values(results_topk_paths: list[str], dataset: str) -> pd.DataFrame:
    metrics_results: dict[str, dict[str, float]] = {}

    paths = [p for p in results_topk_paths if f"_{dataset}_" in p]
    if not paths:
        raise ValueError(f"No se encontraron ficheros para el dataset '{dataset}'")

    for results_path in paths:
        df = pd.read_csv(results_path, index_col=0)
        df.index = df.index.str.replace(r"^val/", "", regex=True)

        is_surprise = "surprise" in results_path

        if not is_surprise:
            for metric in df.index:
                value = df.at[metric, "mean"]
                value *= -1 if metric in ["MSE", "RMSE"] else value
                metrics_results.setdefault(metric, {})["Modelo Propuesto"] = float(
                    value
                )
        else:
            for metric in df.index:
                for model in df.columns:
                    raw = str(df.at[metric, model])
                    val = float(raw.split("/")[0])
                    val *= -1 if metric in ["MSE", "RMSE"] else val
                    metrics_results.setdefault(metric, {})[model] = val

    result_df = pd.DataFrame.from_dict(metrics_results, orient="index")
    return result_df


def friedman_test(
    files: list[str], models: list[str], dataset: str, topk: int, verbose: bool = False
):
    print(f"Friedman test for {dataset} dataset")
    df = extract_metrics_values(files, dataset)
    df_filtered = df[models].dropna()

    if df_filtered.shape[0] < 2:
        raise ValueError(
            "Se necesitan al menos dos datasets con valores completos para todos los modelos"
        )

    scores = [df_filtered[model].values for model in models]

    stat, p = stats.friedmanchisquare(*scores)

    if p < 0.05:
        if verbose:
            print(
                f"\nP-value {p:.4f} < 0.05, los modelos son significativamente diferentes\n"
            )

        # Post-hoc Nemenyi
        nemenyi = sp.posthoc_nemenyi_friedman(df_filtered.values)
        nemenyi.index = nemenyi.columns = df_filtered.columns
        if verbose:
            print("Post hoc Nemenyi Friedman test:\n", nemenyi, "\n")

        # Diagrama de diferencia crítica
        avg_ranks = df_filtered.rank(axis=1, method="average", ascending=False).mean()
        plt.figure(figsize=(10, 4))
        plt.title("Critical Difference Diagram")
        sp.critical_difference_diagram(avg_ranks, nemenyi)
        plt.tight_layout()
        plt.savefig(f"{config.RESULTS_FOLDER}/stats/CDD_{dataset}_{topk}.png")
        # plt.show()

    if verbose:
        print(df_filtered)

    return stat, p
