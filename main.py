import db
import pandas as pd
import numpy as np
from pathlib import Path
from surprise import KNNBasic, Reader, Dataset, accuracy
from surprise.model_selection import cross_validate, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


def preprocess_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica preprocesamiento únicamente a la columna 'rating',
    dejando intactos los identificadores.
    """
    imp = SimpleImputer(missing_values=np.nan, strategy="median")
    df["rating"] = imp.fit_transform(df[["rating"]])

    # Escalar ratings al rango 1-10 (si se requiere normalizar, pero si ya vienen en esa escala,
    # quizás solo convenga hacer una pequeña transformación)
    # Aquí se aplica MinMaxScaler para preservar la escala, ajustando al rango [1, 10]
    scaler = MinMaxScaler(feature_range=(1, 10))
    df["rating"] = scaler.fit_transform(df[["rating"]])

    return df


def main():
    # Cargar datos y, si es necesario, crear la base de datos
    if not Path(db.DB_FILE_PATH).exists():
        db.csv_to_sql(verbose=True)

    df_explicit_ratings_en = pd.read_csv("data/explicit_ratings_en.csv")
    df_explicit_ratings_fr = pd.read_csv("data/explicit_ratings_fr.csv")
    df_explicit_ratings = pd.concat([df_explicit_ratings_en, df_explicit_ratings_fr])

    # Seleccionar únicamente las columnas relevantes
    final_df = df_explicit_ratings[["user_id", "item_id", "rating"]]

    # Aplicar preprocesamiento solo a la columna 'rating'
    final_df = preprocess_ratings(final_df)

    # Cargar en el Dataset de Surprise. Aseguramos que la escala se mantenga en [1,10]
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(
        final_df[["user_id", "item_id", "rating"]], reader=reader
    )

    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # Ejemplo de ajustar hiperparámetros: podrías especificar similitud 'cosine' y
    # ajustar el número de vecinos
    sim_options = {"name": "cosine", "user_based": True}
    algo = KNNBasic(sim_options=sim_options, k=40, min_k=1)

    # Entrenar y evaluar
    # algo.fit(trainset)
    # preds = algo.test(testset)
    #
    # accuracy.rmse(preds)
    # accuracy.mae(preds)
    # accuracy.mse(preds)
    # accuracy.fcp(preds)
    #
    results = cross_validate(
        algo, data, measures=["rmse", "mae", "fcp"], cv=5, verbose=True
    )
    print("Mean RMSE:", np.mean(results["test_rmse"]))


if __name__ == "__main__":
    main()
