import pandas as pd
from src.data_loader import DataLoader


def test_load_csv(tmp_path):
    # create a small CSV
    df = pd.DataFrame({'date': ['2020-01-01', '2020-01-02'], 'value': [1, 2]})
    p = tmp_path / 'sample.csv'
    df.to_csv(p, index=False)

    dl = DataLoader(p)
    loaded = dl.load_csv()
    assert list(loaded.columns) == ['date', 'value']
    assert loaded.shape == (2, 2)
