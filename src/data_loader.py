from pathlib import Path
import pandas as pd


class DataLoader:
    """Simple data loader for CSV files.

    Usage:
        dl = DataLoader('data/news.csv')
        df = dl.load_csv(parse_dates=['date'])
    """

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.df = None

    def load_csv(self, **kwargs):
        """Load CSV into a pandas DataFrame.

        Any pandas.read_csv kwargs can be passed, e.g., parse_dates.
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        self.df = pd.read_csv(self.filepath, **kwargs)
        return self.df

    def head(self, n=5):
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_csv() first.")
        return self.df.head(n)

    def describe(self):
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_csv() first.")
        return self.df.describe(include='all')

    def save_sample(self, outpath, n=100):
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_csv() first.")
        out = Path(outpath)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.df.head(n).to_csv(out, index=False)
