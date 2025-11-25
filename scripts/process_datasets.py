from pathlib import Path
import pandas as pd
import json

RAW = Path('data/raw')
OUT = Path('outputs/summaries')
OUT.mkdir(parents=True, exist_ok=True)

results = []
for p in sorted(RAW.iterdir()):
    info = {'path': str(p), 'name': p.name, 'size': p.stat().st_size}
    try:
        # try CSV
        df = pd.read_csv(p)
        info['type'] = 'csv'
        info['shape'] = df.shape
        info['columns'] = list(df.columns[:20])
        info['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        # basic stats for numeric cols
        desc = df.select_dtypes(include='number').describe().to_dict()
        info['describe'] = desc
        outf = OUT / (p.name + '_summary.json')
        with open(outf, 'w', encoding='utf8') as f:
            json.dump(info, f, indent=2, default=str)
        results.append(info)
    except Exception as e:
        info['error'] = str(e)
        outf = OUT / (p.name + '_summary.json')
        with open(outf, 'w', encoding='utf8') as f:
            json.dump(info, f, indent=2, default=str)
        results.append(info)

print('Processed', len(results), 'files. Summaries in', OUT)
