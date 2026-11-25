
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_params():
    params_path = ROOT / "params.yaml"

    with open(params_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def clean_dataframe(df, params=None):
    """
    Clean known data-quality issues in a copy of the dataframe.
    """
    if params is None:
        params = load_params()

    df = df.copy()

    cleaning = params["cleaning"]

    # 1. Replace sentinel value with missing value
    sentinel_column = cleaning["sentinel_column"]
    sentinel_value = cleaning["sentinel_value"]

    df[sentinel_column] = df[sentinel_column].replace(
        sentinel_value, np.nan
    )

    # 2. Correct the unit mix-up in Block South
    unit_column = cleaning["unit_mixup_column"]
    affected_group = cleaning["unit_mixup_group"]
    factor = cleaning["unit_mixup_factor"]

    mask = df["block"] == affected_group
    df.loc[mask, unit_column] = df.loc[mask, unit_column] / factor

    # 3. Normalize spelling variants in shop_area
    spelling_column = cleaning["spelling_column"]

    df[spelling_column] = (
        df[spelling_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    spelling_map = {
        "tri.": "tribal",
        "rur.": "rural",
        "urb.": "urban"
    }

    df[spelling_column] = df[spelling_column].replace(spelling_map)

    return df


def load_clean_train():
    params = load_params()
    train_path = ROOT / params["data"]["train_path"]

    df = pd.read_csv(train_path)

    return clean_dataframe(df, params)


def get_features_and_target(df):
    """
    Separate model input features and target.
    """
    target = df["grain_kg"].copy()

    columns_to_drop = [
        "record_id",
        "date",
        "kg_entered_in_epos_at_close",
        "grain_kg"
    ]

    features = df.drop(columns=columns_to_drop, errors="ignore")

    return features, target
