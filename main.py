import torch
from torch import nn
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import root_mean_squared_error

from model import CourseRecommender, train
from course_dataset import CourseDataset
from config import DEVICE


# TODO: Hacer validación cruzada
def eval_model(model: nn.Module, valid_dataset: CourseDataset) -> float:
    model.eval()

    with torch.no_grad():
        users = torch.tensor(valid_dataset.users, dtype=torch.long, device=DEVICE)
        courses = torch.tensor(valid_dataset.courses, dtype=torch.long, device=DEVICE)
        watch_percentage = torch.tensor(
            valid_dataset.watch_percentage, dtype=torch.float, device=DEVICE
        )
        ratings = torch.tensor(valid_dataset.ratings, dtype=torch.float, device=DEVICE)

        outputs = model(users, courses, watch_percentage)
        preds = outputs.squeeze().cpu().numpy()
        targets = ratings.cpu().numpy()

        rmse = root_mean_squared_error(preds, targets)

    return rmse


def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_eq_10 = df[df["rating"] == 10]
    df_not_eq_10 = df[df["rating"] != 10]

    min_count = min(len(df_eq_10), len(df_not_eq_10))

    df_eq_10_sampled = df_eq_10.sample(n=min_count, random_state=42)
    df_not_eq_10_sampled = df_not_eq_10.sample(n=min_count, random_state=42)

    df = pd.concat([df_eq_10_sampled, df_not_eq_10_sampled])

    return df


def load_data(balance: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    df_explicit_ratings_en = pd.read_csv("./data/explicit_ratings_en.csv")
    df_explicit_ratings_fr = pd.read_csv("./data/explicit_ratings_fr.csv")
    df_explicit_ratings = pd.concat([df_explicit_ratings_en, df_explicit_ratings_fr])

    # Seleccionar únicamente las columnas relevantes
    df = df_explicit_ratings[["user_id", "item_id", "watch_percentage", "rating"]]

    le_user = preprocessing.LabelEncoder()
    le_course = preprocessing.LabelEncoder()

    df.loc[:, "user_id"] = le_user.fit_transform(df.user_id.values)
    df.loc[:, "item_id"] = le_course.fit_transform(df.item_id.values)

    num_users = len(le_user.classes_)
    num_courses = len(le_course.classes_)

    if balance:
        df = balance_dataset(df)

    df_train, df_val = model_selection.train_test_split(
        df, test_size=0.2, random_state=3, stratify=df.rating.values
    )

    return df_train, df_val, num_users, num_courses


def main():
    print(f"Using {DEVICE} device")

    df_train, df_val, num_users, num_courses = load_data()

    train_dataset = CourseDataset(df_train)
    valid_dataset = CourseDataset(df_val)

    model = CourseRecommender(
        num_users=num_users,
        num_courses=num_courses,
        embedding_size=128,
        hidden_dim=256,
        dropout=0.1,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters())
    loss_func = nn.MSELoss()

    print("Training...")
    train(model, optimizer, loss_func, train_dataset, valid_dataset)

    rmse = eval_model(model, valid_dataset)

    print(f"\nRMSE en validación: {rmse:.4f}")

    with open("results.txt", "a") as f:
        f.write(f"{rmse}\n")


if __name__ == "__main__":
    main()
