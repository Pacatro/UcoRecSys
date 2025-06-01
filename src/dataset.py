import pandas as pd
import numpy as np
from typing import Literal
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import lightning as L


class ELearningDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        encoders: dict | None = None,
        scalers: dict | None = None,
    ):
        self.df = df.copy()
        if encoders:
            for col, encoder in encoders.items():
                self.df[col] = encoder.transform(self.df[col])

        if scalers:
            for col, scaler in scalers.items():
                self.df[col] = scaler.transform(self.df[[col]])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx].map(lambda x: torch.tensor(x)).to_dict()


class ELearningDataModule(L.LightningDataModule):
    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 32,
        test_size: float = 0.4,
        val_size: float = 0.1,
        user_col: str = "user_id",
        item_col: str = "item_id",
        target: str = "rating",
        balance: bool = False,
        threshold: float = 0,
        preprocess: bool = True,
        ignored_cols: list[str] = None,
    ):
        super().__init__()
        self.df = df.copy()
        self.user_col = user_col
        self.item_col = item_col
        self.target = target
        self.batch_size = batch_size
        self.threshold = threshold if threshold != 0 else self.df[target].mean()
        self.balance = balance
        self.preprocess = preprocess
        self.test_size = test_size
        self.val_size = val_size

        self.ignored_cols = ignored_cols or []

        self.num_users = self.df[user_col].nunique()
        self.num_items = self.df[item_col].nunique()
        self.min_rating = self.df[target].min()
        self.max_rating = self.df[target].max()
        self.sparsity = 1 - len(self.df) / (self.num_users * self.num_items)

        self.encoders: dict[str, LabelEncoder] = {}
        self.scalers: dict[str, MinMaxScaler] = {}
        self.cat_cardinalities: dict[str, int] = {}
        self.cont_features: list[str] = []

        self._preprocess()

        if self.balance:
            self.train_df = self._balance_dataset(self.train_df)

    def _preprocess(self):
        self.df[self.user_col] = LabelEncoder().fit_transform(self.df[self.user_col])
        self.df[self.item_col] = LabelEncoder().fit_transform(self.df[self.item_col])

        self.protected_cols = set(
            self.ignored_cols + [self.user_col, self.item_col, self.target]
        )

        remaining_cols = [
            col for col in self.df.columns if col not in self.protected_cols
        ]

        self.train_df, self.val_df, self.test_df = self._split_data()

        for col in remaining_cols:
            if isinstance(self.df[col].dtype, pd.CategoricalDtype):
                le = LabelEncoder().fit(self.train_df[col])
                self.encoders[col] = le
                self.cat_cardinalities[col] = self.train_df[col].nunique()
            else:
                scaler = MinMaxScaler().fit(self.train_df[[col]])
                self.scalers[col] = scaler
                self.cont_features.append(col)

        self.num_cat_features = len(self.cat_cardinalities)
        self.num_cont_features = len(self.cont_features)

    def _split_data(self):
        if self.test_size == 0:
            train_df, val_df = train_test_split(
                self.df, test_size=self.val_size, shuffle=True, random_state=42
            )
            return train_df.reset_index(drop=True), val_df.reset_index(drop=True), None

        train_df, test_df = train_test_split(
            self.df, test_size=self.test_size, shuffle=True, random_state=42
        )
        train_df, val_df = train_test_split(
            train_df, test_size=self.val_size, shuffle=True, random_state=42
        )
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    def _balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        df_eq_10 = df[df[self.target] == 10]
        df_not_eq_10 = df[df[self.target] != 10]
        n = min(len(df_eq_10), len(df_not_eq_10))
        df_eq_10_s = df_eq_10.sample(n=n, random_state=42)
        df_not_eq_10_s = df_not_eq_10.sample(n=n, random_state=42)
        return pd.concat([df_eq_10_s, df_not_eq_10_s]).reset_index(drop=True)

    def setup(self, stage: str | None = None):
        match stage:
            case "fit":
                self.train_dataset = ELearningDataset(
                    self.train_df, encoders=self.encoders, scalers=self.scalers
                )
                self.val_dataset = ELearningDataset(
                    self.val_df, encoders=self.encoders, scalers=self.scalers
                )
            case "test":
                if self.test_df is not None:
                    self.test_dataset = ELearningDataset(
                        self.test_df, encoders=self.encoders, scalers=self.scalers
                    )
                else:
                    self.test_dataset = None

    def train_dataloader(self, num_workers: int = 2):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    def val_dataloader(self, num_workers: int = 2):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
        )

    def test_dataloader(self, num_workers: int = 2):
        return (
            DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=num_workers,
            )
            if self.test_dataset is not None
            else None
        )

    def predict_dataloader(self):
        return (
            DataLoader(self.test_dataset, batch_size=self.batch_size)
            if self.test_dataset is not None
            else None
        )


def load_mars() -> pd.DataFrame:
    explicit_df_en = pd.read_csv("./data/mars_dataset/explicit_ratings_en.csv")
    explicit_df_fr = pd.read_csv("./data/mars_dataset/explicit_ratings_fr.csv")

    items_en = pd.read_csv("./data/mars_dataset/items_en.csv")
    items_fr = pd.read_csv("./data/mars_dataset/items_fr.csv")

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

    features = [
        "user_id",
        "item_id",
        "item_type",
        "difficulty",
        "nb_views",
        "watch_percentage",
        "rating",
    ]

    return df[features]


def load_itm() -> pd.DataFrame:
    ratings_df = pd.read_csv("./data/itm_dataset/ratings.csv")
    items_df = pd.read_csv("./data/itm_dataset/items.csv")
    users_df = pd.read_csv("./data/itm_dataset/users.csv")

    merged_df = pd.merge(left=items_df, right=ratings_df, how="inner", on="Item")
    merged_df = pd.merge(left=merged_df, right=users_df, how="inner", on="UserID")
    merged_df = merged_df.rename(
        columns={"UserID": "user_id", "Item": "item_id", "Rating": "rating"}
    )
    merged_df["Class"] = merged_df["Class"].astype("category")
    merged_df["Semester"] = merged_df["Semester"].astype("category")
    merged_df["Lockdown"] = merged_df["Lockdown"].astype("category")
    merged_df["Title"] = merged_df["Title"].astype("category")
    merged_df[" Age"] = merged_df[" Age"].astype("category")

    features = [
        "user_id",
        "item_id",
        "Title",
        "Semester",
        "Class",
        "App",
        "Lockdown",
        "Ease",
        " Age",
        "Married",
        "rating",
    ]

    return merged_df[features]


def load_coursera() -> pd.DataFrame:
    df = pd.read_csv("./data/coursera_dataset/Coursera.csv")
    num_users = 2000
    num_interactions = 20000
    user_ids = np.random.randint(1, num_users + 1, size=num_interactions)
    course_titles = np.random.choice(df["Course Name"], size=num_interactions)
    interaction_types = np.random.choice(
        ["view", "enroll", "complete", "rate"],
        size=num_interactions,
        p=[0.4, 0.3, 0.2, 0.1],
    )

    # Asegurar que todos tengan un rating (aunque no sea realista para algunas interacciones)
    ratings = np.random.uniform(1, 5, size=num_interactions)

    interactions = pd.DataFrame(
        {
            "user_id": user_ids,
            "Course Name": course_titles,
            "interaction_type": interaction_types,
            "rating": ratings,
        }
    )

    df_merged = interactions.merge(df, on="Course Name", how="left", indicator=True)
    df_merged = df_merged[df_merged["_merge"] == "both"].drop(columns=["_merge"])

    df_merged = df_merged.rename(columns={"Course Name": "item_id"})
    df_merged["item_id"] = df_merged["item_id"].astype("category")
    df_merged["user_id"] = df_merged["user_id"].astype("category")
    df_merged["Difficulty Level"] = df_merged["Difficulty Level"].astype("category")
    df_merged["University"] = df_merged["University"].astype("category")

    features = [
        "user_id",
        "item_id",
        "Difficulty Level",
        "University",
        "rating",
    ]

    return df_merged[features]


def load_doris() -> pd.DataFrame:
    course_info = pd.read_csv("./data/doris_dataset/CourseInformationTable.csv")
    students_info = pd.read_csv("./data/doris_dataset/StudentInformationTable.csv")
    course_selection = pd.read_csv("./data/doris_dataset/CourseSelectionTable.csv")
    course_info.drop(columns=["CourseName"], inplace=True)

    merged_df = pd.merge(
        left=course_info, right=course_selection, how="inner", on="CourseId"
    )
    merged_df = pd.merge(
        left=merged_df, right=students_info, how="inner", on="StudentId"
    )
    merged_df.rename(
        columns={"CourseId": "item_id", "StudentId": "user_id", "Score": "rating"},
        inplace=True,
    )
    merged_df["rating"] = merged_df["rating"].fillna(0)
    merged_df["rating"] = MinMaxScaler(feature_range=(0, 10)).fit_transform(
        merged_df[["rating"]]
    )
    merged_df["Type"] = merged_df["Type"].fillna("Undefined").astype("category")
    merged_df["CourseCollege"] = (
        merged_df["CourseCollege"].fillna("Undefined").astype("category")
    )
    merged_df["College"] = merged_df["College"].fillna("Undefined").astype("category")
    merged_df["CourseName"] = (
        merged_df["CourseName"].fillna("Undefined").astype("category")
    )
    merged_df["Education"] = (
        merged_df["Education"].fillna("Undergraduate").astype("category")
    )
    merged_df["Major"] = merged_df["Major"].fillna("Undefined").astype("category")
    features = [
        "user_id",
        "item_id",
        # "Type",
        # "Semester",
        # "CourseName",
        # "College",
        # "CourseCollege",
        # "Grade",
        "Education",
        "Major",
        "rating",
    ]
    return merged_df[features]


def load_data(
    dataset_name: Literal["mars", "itm", "coursera", "doris"],
) -> pd.DataFrame:
    match dataset_name:
        case "mars":
            return load_mars()
        case "itm":
            return load_itm()
        case "coursera":
            return load_coursera()
        case "doris":
            return load_doris()
