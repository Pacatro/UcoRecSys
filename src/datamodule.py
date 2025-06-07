import pandas as pd
from pandas.api.types import is_numeric_dtype
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
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
                self.df[col] = encoder.transform(self.df[[col]])

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
        predict_df: pd.DataFrame | None = None,
        batch_size: int = 32,
        test_size: float = 0.4,
        val_size: float = 0.1,
        user_col: str = "user_id",
        item_col: str = "item_id",
        target: str = "rating",
        balance: bool = False,
        threshold: float = 0,
        preprocess: bool = True,
        ignored_cols: list[str] | None = None,
    ):
        super().__init__()
        self.df = df.copy()
        self.predict_df = predict_df
        self.user_col = user_col
        self.item_col = item_col
        self.target = target
        self.batch_size = batch_size
        self.threshold = threshold if threshold != 0 else self.df[target].mean()
        self.balance = balance
        self.preprocess = preprocess
        self.test_size = test_size
        self.val_size = val_size
        self.ignored_cols = ignored_cols

        self._preprocess()

        self.num_users = self.df[user_col].nunique()
        self.num_items = self.df[item_col].nunique()
        self.min_rating = self.df[target].min()
        self.max_rating = self.df[target].max()
        self.sparsity = 1 - len(self.df) / (self.num_users * self.num_items)

        if self.balance:
            self.train_df = self._balance_dataset(self.df)

    def _preprocess(self):
        self.encoders = {}
        self.scalers = {}
        self.cat_cardinalities = {}
        self.cont_features = []

        num_nan_targets = self.df[self.target].isna().sum()

        if num_nan_targets > 0:
            self.df = self.df.dropna(subset=[self.target])

        assert self.df[self.target].isna().sum() == 0

        user_encoder = OrdinalEncoder().fit(self.df[[self.user_col]])
        item_encoder = OrdinalEncoder().fit(self.df[[self.item_col]])
        self.encoders[self.user_col] = user_encoder
        self.encoders[self.item_col] = item_encoder

        self.protected_cols = set(
            self.ignored_cols + [self.user_col, self.item_col, self.target]
        )

        remaining_cols = [
            col for col in self.df.columns if col not in self.protected_cols
        ]

        for col in remaining_cols:
            if is_numeric_dtype(self.df[col]):
                scaler = MinMaxScaler().fit(self.df[[col]])
                self.scalers[col] = scaler
                self.cont_features.append(col)

            else:
                self.df[col] = self.df[col].astype("category")
                num_nans = self.df[col].isna().sum()
                if num_nans > 0:
                    self.df[col] = self.df[col].cat.add_categories("Undefined")
                    self.df[col] = self.df[col].fillna("Undefined")

                le = OrdinalEncoder().fit(self.df[[col]])
                self.encoders[col] = le
                self.cat_cardinalities[col] = self.df[col].nunique()

        self.num_cat_features = len(self.cat_cardinalities)
        self.num_cont_features = len(self.cont_features)

        self.train_df, self.val_df, self.test_df = self._split_data()

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
            case "predict":
                assert self.predict_df is not None
                self.predict_dataset = ELearningDataset(
                    self.predict_df, encoders=self.encoders, scalers=self.scalers
                )

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
            DataLoader(self.predict_dataset, batch_size=self.batch_size)
            if self.predict_dataset is not None
            else None
        )
