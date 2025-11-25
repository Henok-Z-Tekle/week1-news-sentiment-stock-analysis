"""Process local yfinance CSVs, add technical indicators, compute metrics, and save outputs.

This script will:
- Read CSVs from `data/yfinance_data/Data/`.
- Standardize columns and parse dates.
- Add technical indicators using `add_technical_indicators` (talib if available, otherwise `ta`).
- Compute simple financial metrics (daily returns, Sharpe) and attempt to use `pynance` if available.
- Save processed CSVs and a small JSON summary per ticker into `outputs/task2/local/`.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
# When running this script directly ensure project root is on sys.path
import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.quant.indicators import add_technical_indicators, sharpe_ratio


DATA_DIR = Path('data/yfinance_data/Data')
OUT_DIR = Path('outputs/task2/local')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure standard OHLCV columns and datetime index.

    Accepts files with common column names and returns DataFrame with
    ['Open','High','Low','Close','Volume'] and datetime index.
    """
    cols = {c.lower(): c for c in df.columns}
    # Map typical names
    mapping = {}
    for name in ['open', 'high', 'low', 'close', 'adj close', 'adj_close', 'volume']:
        if name in cols:
            mapping[cols[name]] = 'Close' if name in ('adj close', 'adj_close') else name.capitalize()

    # If Adjusted Close exists but Close also exists, keep Close
    # Apply mapping conservatively
    df2 = df.copy()
    # Normalize column names
    df2.columns = [c.strip() for c in df2.columns]
    # Ensure datetime index
    if 'Date' in df2.columns:
        df2['Date'] = pd.to_datetime(df2['Date'])
        df2 = df2.set_index('Date')
    elif df2.index.dtype == object:
        try:
            df2.index = pd.to_datetime(df2.index)
        except Exception:
            pass

    # Try to rename common columns to standard ones
    rename_map = {}
    for c in df2.columns:
        lc = c.lower()
        if lc == 'adj close':
            if 'Close' not in df2.columns:
                rename_map[c] = 'Close'
        elif lc in ('close', 'close*'):
            rename_map[c] = 'Close'
        elif lc == 'open':
            rename_map[c] = 'Open'
        elif lc == 'high':
            rename_map[c] = 'High'
        elif lc == 'low':
            rename_map[c] = 'Low'
        elif lc == 'volume':
            rename_map[c] = 'Volume'

    if rename_map:
        df2 = df2.rename(columns=rename_map)

    # Ensure required cols exist
    for required in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if required not in df2.columns:
            df2[required] = np.nan

    # Sort index
    df2 = df2.sort_index()
    return df2


def try_pynance_metrics(df: pd.DataFrame) -> dict:
    """Attempt to compute additional metrics using pynance if available.

    Returns a dict of metrics; if pynance not available, compute fallback metrics.
    """
    metrics = {}
    try:
        import pynance
        # Example: try to compute annualized volatility or other metrics if present.
        # pynance API varies; use fallback if specific functions not found.
        if hasattr(pynance, 'volatility'):
            metrics['pynance_volatility'] = float(pynance.volatility(df['Close']))
        else:
            # fallback simple ann vol
            dr = df['Close'].pct_change().dropna()
            metrics['annual_vol'] = float(dr.std() * (252 ** 0.5))
    except Exception:
        # fallback metrics
        dr = df['Close'].pct_change().dropna()
        metrics['annual_vol'] = float(dr.std() * (252 ** 0.5)) if not dr.empty else None

    return metrics


def process_all():
    summaries = {}
    files = sorted(DATA_DIR.glob('*.csv'))
    if not files:
        print('No CSV files found in', DATA_DIR)
        return summaries

    for p in files:
        try:
            print('Loading', p.name)
            df = pd.read_csv(p)
            df = standardize_df(df)
            # drop rows without close
            df = df[~df['Close'].isna()]
            if df.empty:
                summaries[p.name] = {'error': 'no valid rows after standardization'}
                continue

            outdf = add_technical_indicators(df)
            # compute metrics
            sr = sharpe_ratio(outdf)
            extra = try_pynance_metrics(outdf)

            # save CSV and summary
            out_csv = OUT_DIR / f'{p.stem}_indicators.csv'
            outdf.to_csv(out_csv)
            summary = {
                'file': p.name,
                'rows': int(len(outdf)),
                'sharpe': float(sr) if pd.notna(sr) else None,
            }
            summary.update(extra)
            summaries[p.name] = summary
            with open(OUT_DIR / f'{p.stem}_summary.json', 'w', encoding='utf8') as f:
                json.dump(summary, f, indent=2)
            print('Wrote', out_csv)
        except Exception as e:
            summaries[p.name] = {'error': str(e)}

    # save overall
    with open(OUT_DIR / 'all_summaries.json', 'w', encoding='utf8') as f:
        json.dump(summaries, f, indent=2)

    return summaries


if __name__ == '__main__':
    s = process_all()
    print('Processed', len(s), 'files')
