
import hashlib

import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.mp279.cleaning import (
    ROOT,
    get_features_and_target,
    load_clean_train,
    load_params,
)


def build_pipeline(X, random_state, n_estimators, max_depth):
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", model),
    ])


def main():
    params = load_params()
    model_config = params["model"]

    df = load_clean_train()
    X, y = get_features_and_target(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=model_config["test_size"],
        random_state=model_config["random_state"],
    )

    # SHA-256 of original, unmodified train.csv
    train_path = ROOT / params["data"]["train_path"]
    data_sha256 = hashlib.sha256(
        train_path.read_bytes()
    ).hexdigest()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(model_config["experiment_name"])

    best_mae = float("inf")
    best_run_id = None

    for index, config in enumerate(
        model_config["configurations"], start=1
    ):
        pipeline = build_pipeline(
            X_train,
            model_config["random_state"],
            config["n_estimators"],
            config["max_depth"],
        )

        with mlflow.start_run(run_name=f"v1_config_{index}") as run:
            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_val)
            mae = mean_absolute_error(y_val, predictions)

            mlflow.set_tag("data_sha256", data_sha256)
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_param("n_estimators", config["n_estimators"])
            mlflow.log_param("max_depth", str(config["max_depth"]))
            mlflow.log_param("random_state", model_config["random_state"])
            mlflow.log_param("test_size", model_config["test_size"])

            mlflow.log_metric("mae", mae)

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format="cloudpickle",
            )

            print(
                f"Run {index}: MAE={mae:.4f}, "
                f"Run ID={run.info.run_id}"
            )

            if mae < best_mae:
                best_mae = mae
                best_run_id = run.info.run_id

    # Register the best v1 model
    registered_name = model_config["registered_model_name"]
    model_uri = f"runs:/{best_run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=registered_name,
    )

    client = mlflow.tracking.MlflowClient()

    client.set_registered_model_alias(
        registered_name,
        "champion",
        registered_model.version,
    )

    print("\n========== V1 TRAINING COMPLETE ==========")
    print("Best validation MAE:", round(best_mae, 4))
    print("Best run ID:", best_run_id)
    print("Registered model:", registered_name)
    print("Registered version:", registered_model.version)
    print("Champion alias: v", registered_model.version, sep="")


if __name__ == "__main__":
    main()
