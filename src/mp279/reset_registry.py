
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "MP279_Grain_Model_Final"

BEST_V1_RUN_ID = "f4632be75c174644b31adb5ad963c189"
BEST_V1_MODEL_ID = "m-cf91079863754eb1b4e8a7ca7bb732c2"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)

# Verify the corrected V1 run
run = client.get_run(BEST_V1_RUN_ID)

assert "mae" in run.data.metrics
assert "data_sha256" in run.data.tags

# Verify the logged model
df = mlflow.search_logged_models(
    experiment_ids=[run.info.experiment_id]
)

matches = df[df["model_id"] == BEST_V1_MODEL_ID]

if matches.empty:
    raise RuntimeError("V1 logged model not found.")

if matches.iloc[0]["status"] != "READY":
    raise RuntimeError("V1 logged model is not READY.")

# Do not overwrite an existing registered model
existing = client.search_registered_models(
    filter_string=f"name='{MODEL_NAME}'"
)

if existing:
    raise RuntimeError(
        f"{MODEL_NAME} already exists. Stop and inspect registry."
    )

# Register V1 using its MLflow 3 logged model ID
model_uri = f"models:/{BEST_V1_MODEL_ID}"

version = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME,
)

print("Registered model:", MODEL_NAME)
print("V1 version:", version.version)

if int(version.version) != 1:
    raise RuntimeError(
        "Expected V1 version 1. Stop before V2 training."
    )

client.set_registered_model_alias(
    MODEL_NAME, "champion", version.version
)

print("Champion alias:", client.get_model_version_by_alias(
    MODEL_NAME, "champion"
).version)

print("V1 registry correction successful!")
