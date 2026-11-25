
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    payload = {
        "cards_attached": 500,
        "working_hours": 8,
        "biometric_failures": 5,
        "temp_c": 35,
        "stock_kg_morning": 1000,
        "days_to_month_end": 10,
        "shop_area": "rural",
        "block": "Block North",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], float)


def test_invalid_temperature():
    payload = {
        "cards_attached": 500,
        "working_hours": 8,
        "biometric_failures": 5,
        "temp_c": 81,
        "stock_kg_morning": 1000,
        "days_to_month_end": 10,
        "shop_area": "rural",
        "block": "Block North",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_invalid_working_hours():
    payload = {
        "cards_attached": 500,
        "working_hours": 1,
        "biometric_failures": 5,
        "temp_c": 35,
        "stock_kg_morning": 1000,
        "days_to_month_end": 10,
        "shop_area": "rural",
        "block": "Block North",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
