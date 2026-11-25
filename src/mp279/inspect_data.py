
from pathlib import Path
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"

files = {
    "TRAIN": "train.csv",
    "LATER": "later.csv",
    "HOLDOUT": "holdout_features.csv"
}

for name, filename in files.items():
    path = DATA_DIR / filename

    df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print(f"{name} DATASET")
    print("=" * 60)

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nFirst 5 rows:")
    print(df.head())

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        print("\nDate range:", dates.min(), "to", dates.max())

    if "grain_kg" in df.columns:
        print("\nTarget statistics:")
        print(df["grain_kg"].describe())

    if name == "TRAIN":
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        print("\nOriginal train.csv SHA-256:")
        print(sha256)

    for col in ["block", "shop_area"]:
        if col in df.columns:
            print(f"\nUnique values in {col}:")
            print(df[col].value_counts(dropna=False).to_string())
