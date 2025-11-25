import pandas as pd
import numpy as np


class EDA:
    """Small EDA helper class providing lightweight analysis utilities."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def moving_average(self, column: str, window: int = 7) -> pd.Series:
        if column not in self.df.columns:
            raise KeyError(column)
        return self.df[column].rolling(window=window, min_periods=1).mean()

    def correlation_matrix(self) -> pd.DataFrame:
        return self.df.select_dtypes(include=[np.number]).corr()

    def basic_stats(self, column: str) -> dict:
        if column not in self.df.columns:
            raise KeyError(column)
        s = self.df[column]
        return {
            'mean': s.mean(),
            'median': s.median(),
            'std': s.std(),
            'missing': int(s.isna().sum())
        }
