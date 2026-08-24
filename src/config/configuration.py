from pathlib import Path


# Project root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
BRONZE_DATA_DIR = DATA_DIR / "bronze"
SILVER_DATA_DIR = DATA_DIR / "silver"
GOLD_DATA_DIR = DATA_DIR / "gold"

# Input dataset.
RAW_DATA_FILE = RAW_DATA_DIR / "accepted_2007_to_2018Q4.csv"

# Bronze output directory.
BRONZE_DATA_PATH = BRONZE_DATA_DIR / "bronze_data"

SILVER_DATA_PATH = SILVER_DATA_DIR / "silver_data"

GOLD_DATA_PATH = GOLD_DATA_DIR / "gold_data"