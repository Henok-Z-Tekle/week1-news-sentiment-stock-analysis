from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT = Path('outputs/task2')
LOCAL = OUT / 'local'
PLOTS = OUT / 'plots'
PLOTS.mkdir(parents=True, exist_ok=True)

def main():
    files = sorted(LOCAL.glob('*_indicators.csv'))
    print('Found', len(files), 'indicator CSVs')
    for p in files:
        try:
            print('Loading', p.name)
            df = pd.read_csv(p, parse_dates=['Date'], index_col='Date')
            ticker = p.stem.replace('_indicators','')
            # Plot Close with SMA/EMA if present
            plt.figure(figsize=(12,4))
            plt.plot(df.index, df['Close'], label='Close', color='black')
            if 'sma_21' in df.columns:
                plt.plot(df.index, df['sma_21'], label='SMA 21')
            if 'ema_21' in df.columns:
                plt.plot(df.index, df['ema_21'], label='EMA 21')
            plt.title(f'{ticker} Close and MA')
            plt.legend()
            plt.grid(True)
            fname = PLOTS / f'{ticker}_close_ma.png'
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
            print('Wrote', fname)
            # RSI and MACD
            fig, axes = plt.subplots(2,1, figsize=(12,6), sharex=True)
            axes[0].plot(df.index, df.get('rsi_14', pd.Series(np.nan, index=df.index)), color='tab:blue')
            axes[0].set_title(f'{ticker} RSI (14)')
            axes[0].axhline(70, color='red', linewidth=0.7, linestyle='--')
            axes[0].axhline(30, color='green', linewidth=0.7, linestyle='--')
            axes[1].plot(df.index, df.get('macd', pd.Series(np.nan, index=df.index)), label='MACD', color='tab:orange')
            axes[1].plot(df.index, df.get('macd_signal', pd.Series(np.nan, index=df.index)), label='Signal', color='tab:green')
            axes[1].set_title(f'{ticker} MACD')
            axes[1].legend()
            fig.autofmt_xdate()
            fname2 = PLOTS / f'{ticker}_rsi_macd.png'
            plt.tight_layout()
            plt.savefig(fname2)
            plt.close()
            print('Wrote', fname2)
        except Exception as e:
            print('Failed to plot', p.name, e)

if __name__ == '__main__':
    main()
