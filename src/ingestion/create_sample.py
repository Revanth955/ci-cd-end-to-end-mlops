import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "accepted_2007_to_2018Q4.csv"
)

SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sample.csv"
)

print("Reading first 100,000 rows...")

df = pd.read_csv(RAW_DATA_PATH, nrows=100000)

print("Saving sample...")

df.to_csv(SAMPLE_PATH, index=False)

print("✅ Sample created successfully!")
print(df.shape)