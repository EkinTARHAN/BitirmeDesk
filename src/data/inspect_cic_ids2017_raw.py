from pathlib import Path
import pandas as pd

root = Path("data/raw/cic_ids2017/machine_learning_csv")

csv_files = sorted(root.rglob("*.csv"))

print("CSV file count:", len(csv_files))
print("\nFirst files:")
for p in csv_files[:20]:
    print("-", p)

if not csv_files:
    print("\nHiç CSV bulunamadı.")
    raise SystemExit(0)

sample = csv_files[0]
print("\nSample file:", sample)

df = pd.read_csv(sample, nrows=5)

print("\nColumns:")
for c in df.columns:
    print("-", c)

print("\nShape preview:", df.shape)

print("\nHead:")
print(df.head().to_string())