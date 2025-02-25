import db
import pandas as pd
import numpy as np
from pathlib import Path
from surprise import SVD, Reader, Dataset, accuracy
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms import NMF
from sklearn.impute import SimpleImputer


def main():
    if not Path(db.DB_FILE_PATH).exists():
        db.csv_to_sql(verbose=True)

    df_explicit_ratings_en = pd.read_csv("data/explicit_ratings_en.csv")
    df_explicit_ratings_fr = pd.read_csv("data/explicit_ratings_fr.csv")
    df_explicit_ratings = pd.concat([df_explicit_ratings_en, df_explicit_ratings_fr])

    final_df = df_explicit_ratings[["user_id", "item_id", "rating"]]

    imp = SimpleImputer(missing_values=np.nan, strategy="median")
    imp.fit(final_df)

    data = Dataset.load_from_df(
        df=final_df,
        reader=Reader(rating_scale=(1, 10)),
    )

    trainset, testset = train_test_split(
        data, test_size=0.2, shuffle=True, random_state=42
    )

    model = NMF()

    model.fit(trainset)

    preds = model.test(testset)

    accuracy.rmse(preds)
    accuracy.mae(preds)
    accuracy.mse(preds)
    accuracy.fcp(preds)


if __name__ == "__main__":
    main()
