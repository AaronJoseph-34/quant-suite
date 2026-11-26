import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from app.forecast_metrics import backtest_metrics, confidence_band


def features(window):
    return [*window, np.std(window), np.sum(window), np.max(window), np.min(window)]


def make_features(prices, lags=10):
    returns = np.diff(np.log(np.asarray(prices, dtype=float)))
    X = [features(returns[i - lags:i]) for i in range(lags, len(returns))]
    return np.asarray(X), returns[lags:]


def build_models():
    return {
        "elastic": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.5)),
        ]),
        "rf": RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42,
        ),
    }


def train_ensemble(X, y):
    """Weight on a chronological validation suffix, then refit on X/y.

    The third result is the ensemble's validation log-return MSE, not a weight.
    """
    if len(X) < 5:
        raise ValueError("Need at least five feature observations")
    split = int(0.8 * len(X))
    models = build_models()
    validation = {}
    inverse_errors = {}
    for name, model in models.items():
        model.fit(X[:split], y[:split])
        prediction = model.predict(X[split:])
        validation[name] = prediction
        inverse_errors[name] = 1 / (np.mean((y[split:] - prediction) ** 2) + 1e-12)
    total = sum(inverse_errors.values())
    weights = {name: error / total for name, error in inverse_errors.items()}
    combined = sum(weights[name] * validation[name] for name in models)
    mse = float(np.mean((y[split:] - combined) ** 2))
    for model in models.values():
        model.fit(X, y)
    return models, weights, mse


def ml_backtest(prices, lags=10):
    """One-step predictions on the final 20%, with frozen models.

    Only earlier observations fit/weight models. Observed test-period returns
    can enter later lag windows; test targets never select model weights.
    """
    prices = np.asarray(prices, dtype=float)
    X, y = make_features(prices, lags)
    split = int(0.8 * len(X))
    models, weights, _ = train_ensemble(X[:split], y[:split])
    predicted_returns = sum(weights[n] * m.predict(X[split:]) for n, m in models.items())
    previous = prices[lags + split:-1]
    actual = prices[lags + split + 1:]
    return backtest_metrics(actual, previous * np.exp(predicted_returns), previous)


def ml_forecast(prices, horizon=10, lags=10):
    prices = np.asarray(prices, dtype=float)
    if (prices.ndim != 1 or len(prices) < max(30, lags + 10)
            or not np.all(np.isfinite(prices)) or np.any(prices <= 0)):
        raise ValueError("Need at least 30 finite, positive prices and enough lag observations")
    if horizon < 1 or horizon > 60:
        raise ValueError("Horizon must be between 1 and 60")
    backtest = ml_backtest(prices, lags)
    X, y = make_features(prices, lags)
    models, weights, mse = train_ensemble(X, y)
    returns = np.diff(np.log(prices))
    predictions = []
    for _ in range(horizon):
        row = np.asarray(features(returns[-lags:])).reshape(1, -1)
        prediction = sum(weights[n] * float(m.predict(row)[0]) for n, m in models.items())
        predictions.append(prediction)
        returns = np.append(returns, prediction)
    forecast = prices[-1] * np.exp(np.cumsum(predictions))
    return {
        "method": "ml_ensemble", "forecast": forecast.tolist(),
        **confidence_band(prices, forecast, sigma=np.sqrt(mse)),
        "last_price": float(prices[-1]), "model_weights": weights, "backtest": backtest,
    }
