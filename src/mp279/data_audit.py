
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN = pd.read_csv(ROOT / "data/raw/train.csv")
LATER = pd.read_csv(ROOT / "data/raw/later.csv")

TRAIN["date"] = pd.to_datetime(TRAIN["date"])

# 1. Leakage
print("\n--- LEAKAGE ---")
print("EPOS-target correlation:",
      TRAIN["kg_entered_in_epos_at_close"].corr(TRAIN["grain_kg"]))
print("Mean absolute EPOS-target difference:",
      (TRAIN["kg_entered_in_epos_at_close"] - TRAIN["grain_kg"]).abs().mean())

# 2. Unit mix-up
print("\n--- UNIT MIX-UP ---")
print(TRAIN.groupby("block")["stock_kg_morning"].agg(
    ["min", "median", "mean", "max"]
))

# 3. Sentinel
print("\n--- SENTINEL ---")
print(TRAIN["biometric_failures"].value_counts().head(10))

# 4. Duplicates
print("\n--- DUPLICATES ---")
print("Exact duplicates:", TRAIN.duplicated().sum())
print("Duplicate IDs:", TRAIN["record_id"].duplicated().sum())

# 5. Stuck sensor candidates
print("\n--- STUCK SENSOR CANDIDATES ---")

sensor_columns = [
    "working_hours",
    "biometric_failures",
    "temp_c",
    "stock_kg_morning"
]

for col in sensor_columns:
    ordered = TRAIN.sort_values("date").reset_index(drop=True)
    values = ordered[col]
    run_id = values.ne(values.shift()).cumsum()

    runs = ordered.groupby(run_id).agg(
        value=(col, "first"),
        count=(col, "size"),
        start_date=("date", "min"),
        end_date=("date", "max")
    )

    candidates = runs[runs["count"] >= 5].sort_values(
        "count", ascending=False
    )

    print(f"\n{col}:")
    print(candidates.head(10).to_string(index=False))

# 6. Target typo candidates
print("\n--- TARGET TYPO CANDIDATES ---")

targets = TRAIN["grain_kg"].sort_values().reset_index(drop=True)
gaps = targets.diff().sort_values(ascending=False)

print("Largest target values:")
print(targets.tail(20).to_string(index=False))

print("\nLargest gaps in sorted target:")
print(gaps.head(15).to_string())

# 7. Spelling variants
print("\n--- SPELLING VARIANTS ---")

print("Training labels:")
print(TRAIN["shop_area"].value_counts().to_string())

print("\nClean later labels:")
print(LATER["shop_area"].value_counts().to_string())

print("\nNormalized training labels:")
normalized = TRAIN["shop_area"].str.strip().str.lower().replace({
    "tri.": "tribal",
    "rur.": "rural",
    "urb.": "urban"
})
print(normalized.value_counts().to_string())
print("True category count:", normalized.nunique())

# 8. STUCK SENSOR - check separately for each block
print("\n--- STUCK SENSOR BY BLOCK ---")

for col in ["working_hours", "biometric_failures", "temp_c", "stock_kg_morning"]:
    print(f"\nColumn: {col}")

    for block, group in TRAIN.groupby("block"):
        group = group.sort_values("date").reset_index(drop=True)

        run_id = group[col].ne(group[col].shift()).cumsum()

        runs = group.groupby(run_id).agg(
            value=(col, "first"),
            count=(col, "size"),
            start_date=("date", "min"),
            end_date=("date", "max")
        )

        candidates = runs[runs["count"] >= 3]

        if not candidates.empty:
            print(f"\n{block}:")
            print(candidates.to_string(index=False))


# 9. TARGET TYPO - inspect unusually small and large values
print("\n--- TARGET TYPO INVESTIGATION ---")

print("\nSmallest 30 target values:")
print(
    TRAIN.nsmallest(30, "grain_kg")[
        ["record_id", "date", "block", "grain_kg"]
    ].to_string(index=False)
)

print("\nLargest 30 target values:")
print(
    TRAIN.nlargest(30, "grain_kg")[
        ["record_id", "date", "block", "grain_kg"]
    ].to_string(index=False)
)

print("\nTarget values above 1500:")
print(
    TRAIN[TRAIN["grain_kg"] > 1500][
        ["record_id", "date", "block", "grain_kg"]
    ].to_string(index=False)
)


# 10. Scan all numeric columns for long repeated readings
print("\n--- ALL NUMERIC COLUMNS: STUCK SENSOR SCAN ---")

numeric_cols = TRAIN.select_dtypes(include="number").columns

for col in numeric_cols:
    print(f"\nColumn: {col}")

    for block, group in TRAIN.groupby("block"):
        group = group.sort_values("date").reset_index(drop=True)

        run_id = group[col].ne(group[col].shift()).cumsum()

        runs = group.groupby(run_id).agg(
            value=(col, "first"),
            count=(col, "size"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            unique_days=("date", "nunique")
        )

        candidates = runs[runs["unique_days"] >= 5]

        if not candidates.empty:
            print(f"\n{block}:")
            print(candidates.sort_values(
                "unique_days", ascending=False
            ).head(10).to_string(index=False))


# 11. Check whether target has unusually large values
print("\n--- TARGET VALUES ABOVE 2000 ---")

print(
    TRAIN[TRAIN["grain_kg"] > 2000][
        ["record_id", "date", "block", "grain_kg"]
    ].to_string(index=False)
)

print("\nMaximum target:", TRAIN["grain_kg"].max())


# 10. INVESTIGATE THE UNUSUAL TARGET VALUE
print("\n--- TARGET TYPO FINAL CHECK ---")

row = TRAIN[TRAIN["record_id"] == "R00310"]

print("\nSuspicious record:")
print(row[
    [
        "record_id",
        "date",
        "block",
        "grain_kg",
        "kg_entered_in_epos_at_close"
    ]
].to_string(index=False))

print("\nCompare target with its 10x value:")
if not row.empty:
    value = row["grain_kg"].iloc[0]

    print("Original target:", value)
    print("Target multiplied by 10:", value * 10)

    print("\nOther records with similar target:")
    print(
        TRAIN[
            TRAIN["grain_kg"].between(value * 9, value * 11)
        ][
            ["record_id", "date", "block", "grain_kg"]
        ].to_string(index=False)
    )

print("\nAll targets below 100:")
print(
    TRAIN[TRAIN["grain_kg"] < 100][
        ["record_id", "date", "block", "grain_kg"]
    ].to_string(index=False)
)




print("\n--- R00310 CONTEXT CHECK ---")

TRAIN["date"] = pd.to_datetime(TRAIN["date"])

print(
    TRAIN[
        (TRAIN["block"] == "Block East") &
        (TRAIN["date"].between("2026-02-18", "2026-02-26"))
    ][
        [
            "record_id",
            "date",
            "block",
            "stock_kg_morning",
            "kg_entered_in_epos_at_close",
            "grain_kg"
        ]
    ].sort_values("date").to_string(index=False)
)