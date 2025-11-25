"""Task 3: compute sentiment from news and correlate with stock daily returns.

Usage: run from project root, script will look for a news CSV at `data/newsData/raw_analyst_ratings.csv`
and processed stock indicator CSVs in `outputs/task2/local/*_indicators.csv`.

Outputs are written to `outputs/task3/` as per-ticker summary JSONs and an overall `correlations.csv`.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.sentiment import compute_sentiment, aggregate_sentiment_by_date


OUT = Path('outputs/task3')
OUT.mkdir(parents=True, exist_ok=True)
STOCK_DIR = Path('outputs/task2/local')
NEWS_CANDIDATE = Path('data/newsData/raw_analyst_ratings.csv')


def find_text_column(df: pd.DataFrame) -> str | None:
    for c in ['headline', 'title', 'text', 'content']:
        if c in df.columns:
            return c
    # try case-insensitive match
    lowmap = {col.lower(): col for col in df.columns}
    for c in ['headline', 'title', 'text', 'content']:
        if c in lowmap:
            return lowmap[c]
    return None


def stock_daily_returns(df: pd.DataFrame, close_col: str = 'Close') -> pd.Series:
    s = df[close_col].pct_change().dropna()
    s.index = pd.to_datetime(s.index)
    s.index = s.index.normalize()
    return s


def run(news_path: Path | None = None):
    news_path = news_path or NEWS_CANDIDATE
    if not news_path.exists():
        print('News CSV not found at', news_path)
        return {}

    news = pd.read_csv(news_path)
    txtcol = find_text_column(news)
    if txtcol is None:
        print('No text column found in news CSV; expected one of headline/title/text/content')
        return {}

    datecol = None
    # find a date-like column
    for c in ['date', 'published', 'timestamp']:
        if c in news.columns:
            datecol = c
            break

    if datecol is None:
        # look for first column that looks like a date
        for c in news.columns:
            try:
                pd.to_datetime(news[c].dropna().iloc[0])
                datecol = c
                break
            except Exception:
                continue

    if datecol is None:
        print('No date column detected in news CSV; add a date column.')
        return {}

    sent = compute_sentiment(news, text_col=txtcol, date_col=datecol)
    agg = aggregate_sentiment_by_date(sent)

    summaries = {}
    rows = []
    files = sorted(STOCK_DIR.glob('*_indicators.csv'))
    if not files:
        print('No processed stock indicator CSVs found in', STOCK_DIR)
        return {}

    for p in files:
        try:
            df = pd.read_csv(p, parse_dates=['Date'], index_col='Date')
            ticker = p.stem.replace('_indicators','')
            dr = stock_daily_returns(df)
            # align dates
            common_index = dr.index.intersection(agg.index)
            if len(common_index) < 2:
                corr = None
            else:
                corr = float(dr.reindex(common_index).corr(agg.reindex(common_index)))

            summary = {
                'ticker': ticker,
                'rows_stock': int(len(df)),
                'rows_sentiment_days': int(len(agg)),
                'overlap_days': int(len(common_index)),
                'pearson_corr': corr,
            }
            summaries[ticker] = summary
            rows.append({'ticker': ticker, 'corr': corr, 'overlap_days': int(len(common_index))})

            # save per-ticker summary
            with open(OUT / f'{ticker}_sentiment_summary.json', 'w', encoding='utf8') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            summaries[p.stem] = {'error': str(e)}

    # save overall CSV
    df_out = pd.DataFrame(rows).set_index('ticker')
    df_out.to_csv(OUT / 'correlations.csv')
    with open(OUT / 'all_sentiment_summaries.json', 'w', encoding='utf8') as f:
        json.dump(summaries, f, indent=2)

    print('Wrote', OUT)
    return summaries


if __name__ == '__main__':
    run()
