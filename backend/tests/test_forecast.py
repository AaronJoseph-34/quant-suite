import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.forecast_metrics import confidence_band
from app import forecast_ml_ensemble as ml

client = TestClient(app)


def sample_prices(n=100):
    return (100 * np.exp(np.cumsum(np.random.default_rng(10).normal(0.0003, 0.01, n)))).tolist()


class ConstantModel:
    def __init__(self, constant):
        self.constant = constant
        self.fit_sizes = []

    def fit(self, X, y):
        self.fit_sizes.append(len(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.constant)


def test_ensemble_reports_actual_residual_error_and_refits(monkeypatch):
    models = {"a": ConstantModel(0), "b": ConstantModel(0.01)}
    monkeypatch.setattr(ml, "build_models", lambda: models)
    X, y = np.zeros((20, 2)), np.linspace(-0.02, 0.02, 20)
    _, weights, error = ml.train_ensemble(X, y)
    expected = np.mean((y[16:] - weights["b"] * 0.01) ** 2)
    assert error == pytest.approx(expected)
    assert error < 0.001
    assert sum(weights.values()) == pytest.approx(1)
    assert models["a"].fit_sizes == [16, 20]


def test_band_log_width_scales_with_sqrt_horizon():
    forecast = [100] * 20
    band = confidence_band(sample_prices(), forecast, sigma=0.01)
    width = np.log(np.array(band["high"]) / forecast)
    assert width[-1] / width[0] == pytest.approx(np.sqrt(20))
    assert width[0] == pytest.approx(1.96 * 0.01)


def test_backtest_uses_only_training_prefix(monkeypatch):
    prices = np.asarray(sample_prices())
    X, y = ml.make_features(prices)
    captured = []

    def trainer(train_X, train_y):
        captured.append(train_y.copy())
        return {"zero": ConstantModel(0)}, {"zero": 1}, 0

    monkeypatch.setattr(ml, "train_ensemble", trainer)
    result = ml.ml_backtest(prices)
    split = int(0.8 * len(X))
    np.testing.assert_array_equal(captured[0], y[:split])
    assert result["model"] == result["random_walk"]
    assert result["observations"] == len(y) - split
    assert result["random_walk"]["rmse"] == pytest.approx(np.sqrt(np.mean(np.diff(prices)[10 + split:] ** 2)))
    changed = prices.copy()
    changed[10 + split + 1:] *= np.linspace(1.01, 1.2, len(y) - split)
    ml.ml_backtest(changed)
    np.testing.assert_array_equal(captured[0], captured[1])


def test_ml_endpoint_preserves_models_own_band(monkeypatch):
    monkeypatch.setattr("app.main.ml_forecast", lambda prices, horizon: {
        "forecast": [100] * horizon, "low": [97] * horizon, "high": [103] * horizon,
        "vol_est": 0.015, "last_price": prices[-1], "backtest": {},
    })
    response = client.post("/forecast", json={"prices": sample_prices(), "horizon": 5})
    assert response.json()["low"] == [97] * 5


@pytest.mark.parametrize("method,n", [("ml", 30), ("ml", 100), ("arima", 50)])
def test_real_forecast_and_backtest_are_finite(method, n):
    response = client.post("/forecast", json={"prices": sample_prices(n), "horizon": 20, "method": method})
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result["forecast"]) == 20
    assert result["vol_est"] < 0.1
    assert np.all(np.array(result["low"]) <= result["forecast"])
    assert np.all(np.array(result["high"]) >= result["forecast"])
    assert result["backtest"]["observations"] > 0
    widths = np.log(np.array(result["high"]) / result["forecast"])
    assert widths[-1] / widths[0] == pytest.approx(np.sqrt(20))


@pytest.mark.parametrize("prices", [[100] * 29, [100] * 29 + [0], [100] * 29 + [-1]])
def test_bad_prices_return_400(prices):
    assert client.post("/forecast", json={"prices": prices}).status_code == 400
