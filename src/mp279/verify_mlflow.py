
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///mlflow.db")

# Check experiment
experiment = client.get_experiment_by_name(
    "MP279_Grain_Distribution"
)

if experiment is None:
    raise RuntimeError("Experiment not found")

# Check all runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id]
)

print("Total runs:", len(runs))

for run in runs:
    print("\nRun ID:", run.info.run_id)
    print("MAE:", run.data.metrics.get("mae"))
    print("SHA256:", run.data.tags.get("data_sha256"))

# Check registered model
model_name = "MP279_Grain_Model"

versions = client.search_model_versions(
    filter_string=f"name='{model_name}'"
)

print("\nModel versions:")

for version in versions:
    print("Version:", version.version)

# Check champion alias
champion = client.get_model_version_by_alias(
    model_name,
    "champion"
)

print("\nChampion alias version:", champion.version)


print("MAE:", run.data.metrics.get("mae"))
print("SHA256:", run.data.tags.get("data_sha256"))

print("All metrics:", run.data.metrics)
print("All tags:", run.data.tags)