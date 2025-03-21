import db
import pandas as pd
import numpy as np
from pathlib import Path

# from surprise import KNNBasic, Reader, Dataset, accuracy
# from surprise.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import torch
from model import CourseRec, train_model, evaluate_model


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


def test_with_nn(df: pd.DataFrame) -> float:
    # Remapeamos los identificadores a números más bajos
    df.loc[:, "user_id"], _ = pd.factorize(df["user_id"])
    df.loc[:, "item_id"], _ = pd.factorize(df["item_id"])

    # Separamos el dataset en entrenamiento y prueba
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Convertir las columnas a tensores.
    train_users = torch.tensor(train_df["user_id"].values, dtype=torch.long)
    train_courses = torch.tensor(train_df["item_id"].values, dtype=torch.long)
    train_ratings = torch.tensor(train_df["rating"].values, dtype=torch.float)

    test_users = torch.tensor(test_df["user_id"].values, dtype=torch.long)
    test_courses = torch.tensor(test_df["item_id"].values, dtype=torch.long)
    test_ratings = torch.tensor(test_df["rating"].values, dtype=torch.float)

    num_users = df["user_id"].nunique()
    num_items = df["item_id"].nunique()

    model = CourseRec(num_users=num_users, num_items=num_items, embedding_size=10)
    train_model(model, train_users, train_courses, train_ratings)

    loss = evaluate_model(model, test_users, test_courses, test_ratings)

    print(f"Loss (RMSE): {loss}")

    return loss


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

    loss = test_with_nn(final_df)

    with open("results.txt", "a") as f:
        f.write(f"Loss (RMSE): {loss}")

    # reader = Reader(rating_scale=(1, 10))
    # data = Dataset.load_from_df(final_df[["user_id", "item_id", "rating"]], reader=reader)
    # trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    # sim_options = {"name": "cosine", "user_based": True}
    # algo = KNNBasic(sim_options=sim_options, k=40, min_k=1)
    # algo.fit(trainset)
    # preds = algo.test(testset)
    # accuracy.rmse(preds)
    # accuracy.mae(preds)
    # results = cross_validate(algo, data, measures=["rmse", "mae", "fcp"], cv=5, verbose=True)
    # print("Mean RMSE:", np.mean(results["test_rmse"]))


if __name__ == "__main__":
    main()
