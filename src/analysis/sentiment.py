from __future__ import annotations
from typing import Iterable
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from textblob import TextBlob
    _HAS_TEXTBLOB = True
except Exception:
    _HAS_TEXTBLOB = False


def _polarity_textblob(text: str) -> float:
    if not _HAS_TEXTBLOB:
        raise RuntimeError("TextBlob is not installed")
    try:
        return float(TextBlob(str(text)).sentiment.polarity)
    except Exception:
        return float('nan')


def compute_sentiment(df: pd.DataFrame, text_col: str = 'headline', date_col: str = 'date') -> pd.DataFrame:
    """Return a copy of `df` with a `sentiment` column (polarity -1..1).

    - `date_col` is parsed to pandas datetime and normalized to date (no time).
    - If `text_col` is missing, raises ValueError.
    """
    if text_col not in df.columns:
        raise ValueError(f"Text column not found: {text_col}")

    out = df.copy()
    # parse dates conservatively
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors='coerce')
        out['date_only'] = out[date_col].dt.date
    else:
        # try index
        if hasattr(out.index, 'to_datetime') or out.index.dtype != object:
            out = out.reset_index()
            out[date_col] = pd.to_datetime(out.iloc[:, 0], errors='coerce')
            out['date_only'] = out[date_col].dt.date
        else:
            out['date_only'] = None

    # compute sentiment polarity
    out['sentiment'] = out[text_col].apply(lambda t: _polarity_textblob(t) if _HAS_TEXTBLOB else float('nan'))
    return out


def aggregate_sentiment_by_date(sent_df: pd.DataFrame, date_field: str = 'date_only') -> pd.Series:
    """Aggregate sentiment by date (mean). Returns a Series indexed by pd.Timestamp (date).
    """
    if date_field not in sent_df.columns:
        raise ValueError(f"Date field not found: {date_field}")
    s = sent_df.dropna(subset=['sentiment']).groupby(sent_df[date_field]).sentiment.mean()
    # convert index to datetime.date -> pd.Timestamp at midnight for alignment
    idx = [pd.to_datetime(d) for d in s.index]
    return pd.Series(s.values, index=pd.DatetimeIndex(idx))


__all__ = [
    'compute_sentiment',
    'aggregate_sentiment_by_date',
]
