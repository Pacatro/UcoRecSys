# import lightning as L
import pandas as pd
from pathlib import Path

# from engine import UcoRecSys
import db
from models import NeuralHybrid
from model_eval import cross_validate
from config import (
    EPOCHS,
    BATCH_SIZE,
    DELTA,
    PATIENCE,
    BINARIZE,
    BALANCE,
    K,
    THRESHOLD,
    FEATURES,
    TARGET,
)

# HACER CROSS VALIDTAION CON SETS DE INTERACCIONES DE USUARIOS
# Cómo entreanr? --> Entrenamiento normal y corriente


def load_data(features: list[str], target: str) -> pd.DataFrame:
    explicit_df_en = pd.read_csv("./data/explicit_ratings_en.csv")
    explicit_df_fr = pd.read_csv("./data/explicit_ratings_fr.csv")

    items_en = pd.read_csv("./data/items_en.csv")
    items_fr = pd.read_csv("./data/items_fr.csv")

    df_explicit = pd.concat([explicit_df_en, explicit_df_fr], ignore_index=True)
    df_items = pd.concat([items_en, items_fr], ignore_index=True)

    df_explicit["created_at"] = pd.to_datetime(df_explicit["created_at"])
    df_items = df_items.drop(columns=["created_at"])

    df = pd.merge(df_explicit, df_items, on="item_id", how="inner")

    df["Difficulty"] = df["Difficulty"].fillna("Undefined").astype("category")
    df["type"] = df["type"].fillna("Undefined").astype("category")

    df.rename(
        columns={"Difficulty": "difficulty", "type": "item_type"},
        inplace=True,
    )

    df = df.sort_values(by="created_at", ascending=False)

    return df[features + [target]]


def main():
    if not Path(db.DB_FILE_PATH).exists():
        db.csv_to_sql(verbose=True)

    print(f"Using batch size of {BATCH_SIZE}")
    print(f"Using {len(FEATURES)} features: {FEATURES}")
    print(f"k = {K}")
    print(f"Patience = {PATIENCE}, delta = {DELTA}")
    print(f"Threshold = {THRESHOLD}")
    print(f"Epochs = {EPOCHS}")
    print(f"Balance: {BALANCE}")
    print(f"Binarize: {BINARIZE}\n")

    df = load_data(features=FEATURES, target=TARGET)

    _, avg_metrics = cross_validate(
        model_class=NeuralHybrid,
        df=df,
        n_splits=5,
        early_stopping_delta=DELTA,
        early_stopping_patience=PATIENCE,
    )

    print(avg_metrics)


if __name__ == "__main__":
    main()
