# Ration Shop Distribution — Surat
### MP279 | SCSE3040 MLOps Mini Project

## Project Overview
This project develops a machine learning system to predict grain distribution quantity (`grain_kg`) for ration shops in Surat.

The project follows an end-to-end MLOps workflow, including data cleaning, model training, experiment tracking, model registry, monitoring, API deployment, containerization, Kubernetes configuration, and automated testing.

## Machine Learning
- **Problem type:** Regression
- **Target variable:** `grain_kg`
- **Model:** Registered champion model (V2)
- **Model artifact:** `models/model.joblib`
- **Experiment tracking:** MLflow
- **Model evaluation:** Mean Absolute Error (MAE)

## MLOps Workflow
1. Data validation and cleaning
2. Feature preparation
3. Model training and evaluation
4. MLflow experiment tracking
5. Model registry and champion model selection
6. Data and model monitoring
7. Holdout prediction generation
8. FastAPI model serving
9. Docker containerization
10. Kubernetes deployment configuration
11. Automated testing with GitHub Actions

## API
The FastAPI application is located at `app/main.py`.

### Available endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict` | Generate grain distribution prediction |
| GET | `/docs` | Interactive API documentation |

## Run Locally
Create and activate a Python virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Open the API documentation at:

`http://127.0.0.1:8000/docs`

## Docker
Build the Docker image:

```bash
docker build -t mp279-api:1.0 .
```

Run the container:

```bash
docker run -d --name mp279-container -p 8000:8000 mp279-api:1.0
```

## Testing
Run the test suite:

```bash
python -m pytest -v
```

The project test suite contains API and data-cleaning tests. All 7 tests passed in the GitHub Actions CI workflow.

## Kubernetes
Kubernetes deployment and service manifests are available in:

`k8s/deployment.yaml`

## Repository Structure
```text
.
├── app/
│   └── main.py
├── models/
│   └── model.joblib
├── predictions/
│   └── holdout_predictions.csv
├── reports/
│   └── findings.yaml
├── src/
│   └── mp279/
├── tests/
├── k8s/
│   └── deployment.yaml
├── Dockerfile
├── requirements.txt
└── .github/
    └── workflows/
        └── ci.yml
```

## Team
Group: G162  
Project Code: MP279  
Course: SCSE3040 — MLOps
