import pandas as pd
from scipy import stats
import scikit_posthocs as sp


def extract_metrics_values(results_topk_paths: list[str], metric: str) -> pd.DataFrame:
    metrics_results = {}

    for results_path in results_topk_paths:
        dataset = results_path.split("_")[3]

        if dataset not in metrics_results:
            metrics_results[dataset] = {}

        results_df = pd.read_csv(results_path, index_col=0)
        results_df.index = results_df.index.str.replace(r"^val/", "", regex=True)

        if "surprise" not in results_path:
            if metric == "Precision@10":
                metric = "Precison@10"
            if metric in results_df.index:
                metrics_results[dataset]["UcoRecSys"] = results_df.at[metric, "mean"]
        else:
            models = results_df.columns
            for model in models:
                if metric == "Precison@10":
                    metric = "Precision@10"
                if metric in results_df.index:
                    metric_result = float(
                        str(results_df.at[metric, model]).split("/")[0]
                    )
                    metrics_results[dataset][model] = metric_result

    return pd.DataFrame.from_dict(metrics_results, orient="index")


def friedman_test(
    files: list[str], models: list[str], metric: str, verbose: bool = False
):
    df = extract_metrics_values(files, metric)
    if verbose:
        print(df)
    df_filtered = df[models].dropna()

    if df_filtered.shape[0] < 2:
        raise ValueError(
            "Se necesitan al menos dos datasets con valores completos para todos los modelos"
        )

    scores = [df_filtered[model].values for model in models]

    stat, p = stats.friedmanchisquare(*scores)

    if p < 0.05:
        print(f"\nP-value {p} < 0.05, no hay evidencia de que los modelos sean iguales")
        nemeyi = sp.posthoc_nemenyi(df_filtered.values)
        nemeyi.index = nemeyi.columns = df_filtered.columns
        print("\n", nemeyi, "\n")

    return stat, p
