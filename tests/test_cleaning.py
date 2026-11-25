
import numpy as np
import pandas as pd

from src.mp279.cleaning import clean_dataframe, get_features_and_target


PARAMS = {
    "cleaning": {
        "sentinel_column": "biometric_failures",
        "sentinel_value": 9999,
        "unit_mixup_column": "stock_kg_morning",
        "unit_mixup_group": "Block South",
        "unit_mixup_factor": 1000,
        "spelling_column": "shop_area",
    }
}


def sample_data():
    return pd.DataFrame({
        "record_id": ["R1", "R2"],
        "date": ["2026-01-01", "2026-01-02"],
        "cards_attached": [100, 120],
        "working_hours": [8, 9],
        "biometric_failures": [9999, 2],
        "temp_c": [30, 32],
        "stock_kg_morning": [1500, 1500000],
        "days_to_month_end": [10, 9],
        "shop_area": [" TRIBAL ", "rur."],
        "block": ["Block North", "Block South"],
        "kg_entered_in_epos_at_close": [500, 600],
        "grain_kg": [490, 590],
    })


def test_sentinel_replaced():
    cleaned = clean_dataframe(sample_data(), PARAMS)

    assert pd.isna(cleaned.loc[0, "biometric_failures"])


def test_south_stock_unit_corrected():
    cleaned = clean_dataframe(sample_data(), PARAMS)

    assert cleaned.loc[1, "stock_kg_morning"] == 1500


def test_categories_and_features():
    cleaned = clean_dataframe(sample_data(), PARAMS)
    X, y = get_features_and_target(cleaned)

    assert set(cleaned["shop_area"]) == {"tribal", "rural"}
    assert "grain_kg" not in X.columns
    assert "kg_entered_in_epos_at_close" not in X.columns
    assert len(X.columns) == 8
    assert len(y) == 2
