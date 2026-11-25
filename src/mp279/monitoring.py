
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import mean_absolute_error

from src.mp279.cleaning import (
    ROOT,
    clean_dataframe,
    get_features_and_target,
    load_params,
)

NUMERIC_FEATURES = [
    "cards_attached",
    "working_hours",
    "biometric_failures",
    "temp_c",
    "stock_kg_morning",
    "days_to_month_end",
]


def calculate_psi(reference, current, bins=10):
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()

    if reference.empty or current.empty:
        return None

    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    epsilon = 1e-6
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    return float(
        np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    )


def main():
    params = load_params()

    train_path = ROOT / params["data"]["train_path"]
    later_path = ROOT / params["data"]["later_path"]

    train_raw = pd.read_csv(train_path)
    later_raw = pd.read_csv(later_path)

    train = clean_dataframe(train_raw, params)
    later = clean_dataframe(later_raw, params)

    print("\n--- PSI: CLEAN TRAIN vs LATER ---")
    psi_values = {}

    for column in NUMERIC_FEATURES:
        psi = calculate_psi(train[column], later[column])
        psi_values[column] = psi
        print(f"{column}: {psi:.6f}" if psi is not None
              else f"{column}: unavailable")

    available_psi = {
        key: value for key, value in psi_values.items()
        if value is not None
    }

    if available_psi:
        most_drifted = max(available_psi, key=available_psi.get)
        print("\nMost drifted feature:", most_drifted)

    print("\n--- V1 MAE BY BLOCK ON LATER DATA ---")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    model = mlflow.pyfunc.load_model(
        "models:/MP279_Grain_Model_Final/1"
    )

    features, target = get_features_and_target(later)

    predictions = model.predict(features)

    later_result = later.copy()
    later_result["prediction"] = predictions
    later_result["absolute_error"] = (
        target.to_numpy() - predictions
    ).__abs__()

    mae_by_group = (
        later_result.groupby("block")["absolute_error"]
        .mean()
        .to_dict()
    )

    for block, mae in mae_by_group.items():
        print(f"{block}: {mae:.4f}")

    if mae_by_group:
        worst_group = max(mae_by_group, key=mae_by_group.get)
        print("\nWorst-served group:", worst_group)

    print("\n--- PSI VALUES FOR findings.yaml ---")
    for key, value in psi_values.items():
        print(f"{key}: {value}")

    print("\n--- MAE VALUES FOR findings.yaml ---")
    for key, value in mae_by_group.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
