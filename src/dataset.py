import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightning as L


class ELearningDataset(Dataset):
    def __init__(self, df: pd.DataFrame, encoders: dict = None, scalers: dict = None):
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
        target: str,
        batch_size: int = 32,
        threshold: float = 7.5,
        balance: bool = False,
        train_frac: float = 0.6,
        val_frac: float = 0.2,
    ):
        super().__init__()
        self.df = df.copy()
        self.interactions_cols = ["user_id", "item_id", "rating"]
        self.content_cols = df.columns.difference(self.interactions_cols).tolist() + [
            "item_id"
        ]
        self.target = target
        self.batch_size = batch_size
        self.threshold = threshold
        self.balance = balance
        self.num_users = df["user_id"].nunique()
        self.num_items = df["item_id"].nunique()
        self.global_mean = df["rating"].mean()
        self.train_frac = train_frac
        self.val_frac = val_frac

        self.encoders = {}
        self.scalers = {}
        self.cat_cardinalities = {}
        self.numeric_features = []

        for col in df.columns:
            if col != target:
                if isinstance(df[col].dtype, pd.CategoricalDtype):
                    self.encoders[col] = LabelEncoder().fit(df[col])
                    if col not in ["user_id", "item_id"]:
                        self.cat_cardinalities[col] = df[col].nunique()
                else:
                    self.scalers[col] = StandardScaler().fit(df[[col]])
                    self.numeric_features.append(col)

        self.num_cat_features = len(self.cat_cardinalities)
        self.num_num_features = len(self.numeric_features)

        self.train_df, self.val_df, self.test_df = self._split_data()

        if balance:
            self.train_df = self._balance_dataset(self.train_df)

    def _split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(self.df)
        train_end = int(self.train_frac * n)
        val_end = int((self.train_frac + self.val_frac) * n)

        train_df = self.df.iloc[:train_end]
        val_df = self.df.iloc[train_end:val_end]
        test_df = self.df.iloc[val_end:]

        return train_df, val_df, test_df

    def _balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        df_eq_10 = df[df["rating"] == 10]
        df_not_eq_10 = df[df["rating"] != 10]

        min_count = min(len(df_eq_10), len(df_not_eq_10))

        df_eq_10_sampled = df_eq_10.sample(n=min_count, random_state=42)
        df_not_eq_10_sampled = df_not_eq_10.sample(n=min_count, random_state=42)

        df = pd.concat([df_eq_10_sampled, df_not_eq_10_sampled])

        return df

    def setup(self, stage: str = None):
        if stage == "fit":
            self.train_dataset = ELearningDataset(
                self.train_df, encoders=self.encoders, scalers=self.scalers
            )
            self.val_dataset = ELearningDataset(
                self.val_df, encoders=self.encoders, scalers=self.scalers
            )
        elif stage == "test":
            self.test_dataset = ELearningDataset(
                self.test_df, encoders=self.encoders, scalers=self.scalers
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
            self.val_dataset, batch_size=self.batch_size, num_workers=num_workers
        )

    def test_dataloader(self, num_workers: int = 2):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=num_workers
        )

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
