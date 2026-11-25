
import hashlib
import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

from src.mp279.cleaning import (
    ROOT,
    clean_dataframe,
    get_features_and_target,
    load_clean_train,
    load_params,
)
from src.mp279.train import build_pipeline


def main():
    params = load_params()
    config = params["model"]

    model_name = config["registered_model_name"]
    tracking_uri = "sqlite:///mlflow.db"

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # V1 must already exist as version 1
    versions = client.search_model_versions(
        filter_string=f"name='{model_name}'"
    )
    version_numbers = sorted(int(v.version) for v in versions)

    if version_numbers != [1]:
        raise RuntimeError(
            f"Expected only V1 version 1. Found {version_numbers}. "
            "Stop and inspect registry."
        )

    # Load cleaned original training data
    train_df = load_clean_train()

    # Load and clean later data
    later_path = ROOT / params["data"]["later_path"]
    later_df = pd.read_csv(later_path)
    later_df = clean_dataframe(later_df, params)

    # Chronological split: first 70%, last 30%
    later_df["date"] = pd.to_datetime(later_df["date"])
    later_df = later_df.sort_values("date").reset_index(drop=True)

    split_index = int(len(later_df) * 0.70)

    later_train = later_df.iloc[:split_index].copy()
    later_test = later_df.iloc[split_index:].copy()

    # V2 training = original train + first 70% of later data
    combined_train = pd.concat(
        [train_df, later_train],
        ignore_index=True,
    )

    X_train, y_train = get_features_and_target(combined_train)
    X_test, y_test = get_features_and_target(later_test)

    print("Original training rows:", len(train_df))
    print("Later training rows:", len(later_train))
    print("Combined V2 training rows:", len(combined_train))
    print("Temporal test rows:", len(later_test))

    # Evaluate V1 on the same temporal test data
    v1_model = mlflow.sklearn.load_model(
        f"models:/{model_name}/1"
    )

    v1_predictions = v1_model.predict(X_test)
    v1_mae = mean_absolute_error(y_test, v1_predictions)

    print("\nV1 temporal MAE:", round(v1_mae, 4))

    # SHA-256 of original, unmodified train.csv
    train_path = ROOT / params["data"]["train_path"]
    data_sha256 = hashlib.sha256(
        train_path.read_bytes()
    ).hexdigest()

    mlflow.set_experiment(config["experiment_name"])

    best_mae = float("inf")
    best_model_id = None
    best_run_id = None

    # Train all configured V2 models
    for index, model_config in enumerate(
        config["configurations"], start=1
    ):
        pipeline = build_pipeline(
            X_train,
            config["random_state"],
            model_config["n_estimators"],
            model_config["max_depth"],
        )

        with mlflow.start_run(
            run_name=f"v2_config_{index}"
        ) as run:

            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)

            mlflow.set_tag("data_sha256", data_sha256)
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_param("model_version", "v2")
            mlflow.log_param(
                "n_estimators", model_config["n_estimators"]
            )
            mlflow.log_param(
                "max_depth", str(model_config["max_depth"])
            )
            mlflow.log_param("training_rows", len(combined_train))
            mlflow.log_param("later_train_rows", len(later_train))
            mlflow.log_param("temporal_test_rows", len(later_test))
            mlflow.log_param("split_method", "chronological_70_30")

            mlflow.log_metric("mae", mae)

            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format="cloudpickle",
            )

            print(
                f"V2 Run {index}: MAE={mae:.4f}, "
                f"Run ID={run.info.run_id}"
            )

            if mae < best_mae:
                best_mae = mae
                best_model_id = model_info.model_id
                best_run_id = run.info.run_id

    if best_model_id is None:
        raise RuntimeError("No best V2 model was found.")

    improvement = ((v1_mae - best_mae) / v1_mae) * 100

    print("\n========== V1 vs V2 ==========")
    print("V1 temporal MAE:", round(v1_mae, 4))
    print("Best V2 temporal MAE:", round(best_mae, 4))
    print("Improvement:", round(improvement, 2), "%")

    # Register V2 as version 2
    registered_v2 = mlflow.register_model(
        model_uri=f"models:/{best_model_id}",
        name=model_name,
    )

    print("Registered V2 version:", registered_v2.version)

    if int(registered_v2.version) != 2:
        raise RuntimeError(
            "Expected V2 version 2. Stop and inspect registry."
        )

    # Promote V2 only if improvement is at least 8%
    if improvement >= 8:
        client.set_registered_model_alias(
            model_name, "champion", registered_v2.version
        )
        print("Decision: V2 promoted to champion.")
    else:
        print("Decision: V2 not promoted; V1 remains champion.")

    champion = client.get_model_version_by_alias(
        model_name, "champion"
    )

    print("Champion version:", champion.version)
    print("Best V2 run ID:", best_run_id)
    print("\nV2 training complete!")


if __name__ == "__main__":
    main()
