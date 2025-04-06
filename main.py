import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from surprise import Reader, Dataset, SVDpp, accuracy
from surprise.model_selection import train_test_split
from evidently.metric_preset import RecsysPreset
from evidently.report import Report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

import db


def get_top_k_preds(predictions, k=10):
    """
    Transforma una lista de objetos Prediction de Surprise en un DataFrame
    con el formato requerido por Evidently para métricas de sistemas de recomendación.

    Cada fila contendrá:
      - user_id: identificador del usuario
      - item_id: identificador del ítem recomendado
      - prediction: puntuación estimada por el modelo
      - rank: posición en la lista de recomendaciones (1 = mejor)
      - target: 1 si el rating real (r_ui) es mayor o igual a 7, 0 en caso contrario

    :param predictions: Lista de objetos Prediction de Surprise.
    :param k: Top K de recomendaciones. Por defecto 10.
    """
    top_n = defaultdict(list)
    for pred in predictions:
        top_n[pred.uid].append(pred)

    rows = []
    for _, preds in top_n.items():
        # Ordena las predicciones para cada usuario de mayor a menor score
        preds.sort(key=lambda x: x.est, reverse=True)
        for rank, pred in enumerate(preds[:k], start=1):
            rows.append(
                {
                    "user_id": pred.uid,
                    "item_id": pred.iid,
                    "prediction": pred.est,
                    "rank": rank,
                    "target": int(
                        pred.r_ui >= 7
                    ),  # Ajusta este umbral según tu criterio
                }
            )
    return pd.DataFrame(rows)


def generate_report(preds: pd.DataFrame, k: int = 10, report_path: str = "report.html"):
    """
    Genera un reporte de evidently con los resultados del sistema de recomendación.

    :param preds: Las predicciones del sistema.
    :param k: Top K de recomendaciones.
    :param report_path: El path donde guardar el reporte.
    """
    pred_df = get_top_k_preds(preds, k=k)

    metrics = RecsysPreset(k=10)
    report = Report(metrics=[metrics])
    report.run(reference_data=None, current_data=pred_df)

    report.save_html(filename=report_path)


def preprocess_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica preprocesamiento únicamente a la columna 'rating',
    dejando intactos los identificadores.
    """
    imp = SimpleImputer(missing_values=np.nan, strategy="median")
    df.loc[:, "rating"] = imp.fit_transform(df[["rating"]])

    # Escalar ratings al rango 1-10
    scaler = MinMaxScaler(feature_range=(1, 10))
    df.loc[:, "rating"] = scaler.fit_transform(df[["rating"]])

    return df


def main():
    preprocess = len(sys.argv) > 1 and sys.argv[1]

    if not Path(db.DB_FILE_PATH).exists():
        db.csv_to_sql(verbose=True)

    df_explicit_ratings_en = pd.read_csv("data/explicit_ratings_en.csv")
    df_explicit_ratings_fr = pd.read_csv("data/explicit_ratings_fr.csv")
    df_explicit_ratings = pd.concat([df_explicit_ratings_en, df_explicit_ratings_fr])

    # Seleccionar únicamente las columnas relevantes
    final_df = df_explicit_ratings[["user_id", "item_id", "rating"]]

    if preprocess:
        print("Preprocesando...")
        final_df = preprocess_ratings(final_df)

    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(final_df, reader)

    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

    algo = SVDpp()
    algo.fit(trainset)

    predictions = algo.test(testset)

    accuracy.mse(predictions)
    accuracy.rmse(predictions)
    accuracy.mae(predictions)
    accuracy.fcp(predictions)

    generate_report(predictions, k=10)


if __name__ == "__main__":
    main()
