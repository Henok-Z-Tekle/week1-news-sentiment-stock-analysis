"""Quantitative analysis helpers: fetch data, compute technical indicators and simple financial metrics.

This module prefers `talib` when available, and falls back to the `ta` library (pure-python)
to calculate RSI/MACD/EMA/SMA so the code works on environments where TA-Lib C library is not installed.

Functions:
- fetch_data: download OHLCV using yfinance
- add_technical_indicators: adds SMA/EMA/RSI/MACD and signal columns
- daily_returns / sharpe_ratio
"""
from __future__ import annotations
from typing import Optional, List
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import talib
    _HAS_TALIB = True
except Exception:
    _HAS_TALIB = False

try:
    # `ta` is a pure-python library (pip install ta)
    import ta
    _HAS_TA = True
except Exception:
    _HAS_TA = False


def fetch_data(ticker: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
    """Fetch historical OHLCV data for `ticker` using yfinance.

    Returns a DataFrame indexed by Datetime with columns: Open, High, Low, Close, Volume
    """
    import yfinance as yf
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise RuntimeError(f'No data downloaded for {ticker} with period={period} interval={interval}')
    df.index = pd.to_datetime(df.index)
    return df


def _rsi_talib(close: pd.Series, timeperiod: int = 14) -> pd.Series:
    return pd.Series(talib.RSI(close.values, timeperiod=timeperiod), index=close.index)


def _macd_talib(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd, macdsig, macdhist = talib.MACD(close.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
    return pd.Series(macd, index=close.index), pd.Series(macdsig, index=close.index), pd.Series(macdhist, index=close.index)


def _rsi_ta(close: pd.Series, window: int = 14) -> pd.Series:
    return ta.momentum.RSIIndicator(close, window=window).rsi()


def _macd_ta(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd = ta.trend.MACD(close, window_slow=slow, window_fast=fast, window_sign=signal)
    return macd.macd(), macd.macd_signal(), macd.macd_diff()


def add_technical_indicators(df: pd.DataFrame, close_col: str = 'Close', ma_windows: Optional[List[int]] = None) -> pd.DataFrame:
    """Return a DataFrame with added technical indicator columns.

    Adds SMA/EMA for windows in `ma_windows`, RSI (14), MACD (12,26,9).
    If talib is available it will be used for indicators; otherwise `ta` is used as fallback.
    """
    if ma_windows is None:
        ma_windows = [7, 21, 50]

    out = df.copy()
    close = out[close_col]

    # moving averages
    for w in ma_windows:
        out[f'sma_{w}'] = close.rolling(window=w, min_periods=1).mean()
        out[f'ema_{w}'] = close.ewm(span=w, adjust=False).mean()

    # RSI
    try:
        if _HAS_TALIB:
            out['rsi_14'] = _rsi_talib(close, timeperiod=14)
        elif _HAS_TA:
            out['rsi_14'] = _rsi_ta(close, window=14)
        else:
            out['rsi_14'] = np.nan
    except Exception:
        out['rsi_14'] = np.nan

    # MACD
    try:
        if _HAS_TALIB:
            macd, macdsig, macdhist = _macd_talib(close)
        elif _HAS_TA:
            macd, macdsig, macdhist = _macd_ta(close)
        else:
            macd = macdsig = macdhist = pd.Series(np.nan, index=close.index)

        out['macd'] = macd
        out['macd_signal'] = macdsig
        out['macd_hist'] = macdhist
    except Exception:
        out['macd'] = out['macd_signal'] = out['macd_hist'] = np.nan

    return out


def daily_returns(df: pd.DataFrame, close_col: str = 'Close') -> pd.Series:
    return df[close_col].pct_change().fillna(0)


def sharpe_ratio(df: pd.DataFrame, close_col: str = 'Close', freq: int = 252) -> float:
    r = daily_returns(df, close_col)
    # remove zeros or nan
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return float('nan')
    # annualized Sharpe (assuming rf=0)
    return (r.mean() / r.std()) * np.sqrt(freq)


def save_indicators(df: pd.DataFrame, outpath: str | Path):
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath)


__all__ = [
    'fetch_data',
    'add_technical_indicators',
    'daily_returns',
    'sharpe_ratio',
    'save_indicators',
]
