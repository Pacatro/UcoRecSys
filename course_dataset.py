import torch
from torch.utils.data import Dataset
import pandas as pd


class CourseDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        users, courses, watch_percentage, ratings = (
            df.user_id.values,
            df.item_id.values,
            df.watch_percentage.values,
            df.rating.values,
        )
        self.users = users
        self.courses = courses
        self.watch_percentage = watch_percentage
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, item):
        users, courses, watch_percentage, ratings = (
            self.users[item],
            self.courses[item],
            self.watch_percentage[item],
            self.ratings[item],
        )
        return {
            "users": torch.tensor(users, dtype=torch.long),
            "courses": torch.tensor(courses, dtype=torch.long),
            "watch_percentage": torch.tensor(watch_percentage, dtype=torch.float),
            "ratings": torch.tensor(ratings, dtype=torch.float),
        }
