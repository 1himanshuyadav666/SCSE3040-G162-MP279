from pathlib import Path

import pandas as pd
import mlflow

from src.mp279.cleaning import ROOT, clean_dataframe, load_params


def main():
    params = load_params()

    holdout_path = ROOT / params["data"]["holdout_path"]
    output_path = ROOT / "predictions" / "holdout_predictions.csv"

    # Load holdout data
    holdout_raw = pd.read_csv(holdout_path)

    # Keep record IDs before cleaning
    record_ids = holdout_raw["record_id"].copy()

    # Apply the same cleaning used during training
    holdout = clean_dataframe(holdout_raw, params)

    # Select exactly the 8 model input features
    feature_columns = [
        "cards_attached",
        "working_hours",
        "biometric_failures",
        "temp_c",
        "stock_kg_morning",
        "days_to_month_end",
        "shop_area",
        "block",
    ]

    features = holdout[feature_columns].copy()

    # Load registered V2 champion model
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    model = mlflow.pyfunc.load_model(
        "models:/MP279_Grain_Model_Final/2"
    )

    # Generate predictions
    predictions = model.predict(features)

    # Create output CSV
    result = pd.DataFrame({
        "record_id": record_ids,
        "prediction": predictions,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print("Holdout predictions created.")
    print("Rows:", len(result))
    print("Columns:", list(result.columns))
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()