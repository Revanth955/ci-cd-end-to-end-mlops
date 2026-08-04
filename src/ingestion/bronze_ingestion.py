import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "accepted_2007_to_2018Q4.csv"
)

BRONZE_PATH = PROJECT_ROOT / "data" / "bronze" / "bronze_data.csv"

if not RAW_DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")

print("✅ Dataset found!")

df = pd.read_csv(RAW_DATA_PATH,nrows=5)

print(df.head())
print(df.shape)
print(df.info())