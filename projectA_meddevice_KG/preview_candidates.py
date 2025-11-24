import pandas as pd
from pathlib import Path

base = Path("data") / "kge"
print("list of csv files in data kge")
for f in sorted(base.glob("*.csv")):
    print(f)

sample = base / "K101995_K094012_candidates.csv"
if sample.exists():
    df = pd.read_csv(sample)
    print("columns in sample file")
    print(list(df.columns))
    print("first five rows")
    print(df.head())
else:
    print("sample file not found")
