import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Callable

from sklearn.preprocessing import StandardScaler


class TimeseriesDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        targets: List[str],
        input_length: int,
        start_date: str = "1749-01",
        end_date: str = "2021-01",
        scaler: Callable = None,
    ) -> None:
        super().__init__()
        df = df[start_date:end_date].copy()
        self.scaler = scaler
        if not self.scaler:
            self.scaler = StandardScaler()
            df[:] = self.scaler.fit_transform(df.values)
        else:
            df[:] = self.scaler.transform(df.values)
        self.features = df[features]
        self.targets = df[targets]
        self.input_length = input_length

    def __len__(self):
        return len(self.features) - self.input_length

    def __getitem__(self, idx):
        x = self.features.iloc[idx : idx + self.input_length].values
        y = self.targets.iloc[idx + self.input_length].values
        date = self.targets.index[idx + self.input_length]
        return x, y, date
