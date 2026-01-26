import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error




# ---------------------------
# Feature Engineering
# ---------------------------

def make_features(prices, lags=10):
    prices = np.asarray(prices, dtype=float)
    returns = np.diff(np.log(prices))

    X, y = [], []

    for i in range(lags, len(returns)):
        window = returns[i - lags : i]

        X.append([
            *window,
            np.std(window),
            np.sum(window),
            np.max(window),
            np.min(window),
        ])
        y.append(returns[i])

    return np.array(X), np.array(y)


# ---------------------------
# Model Zoo
# ---------------------------

def build_models():
    return {
        "elastic": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.5)),
        ]),

        "rf": RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
        ),

        
    }


# ---------------------------
# Ensemble Training
# ---------------------------

def train_ensemble(X, y):
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    models = build_models()
    weights = {}
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mse = mean_squared_error(y_val, preds)

        trained[name] = model
        weights[name] = 1 / (mse + 1e-6)  # inverse-error weighting

    # Normalize weights
    total = sum(weights.values())
    for k in weights:
        weights[k] /= total

    return trained, weights, np.mean(list(weights.values()))


# ---------------------------
# Forecast Engine
# ---------------------------

def ml_forecast(prices, horizon=10, lags=10):
    prices = np.asarray(prices, dtype=float)

    X, y = make_features(prices, lags)
    models, weights, avg_error = train_ensemble(X, y)

def ml_forecast(prices, horizon=10, lags=10):
    import numpy as np

    prices = np.asarray(prices, dtype=float)

    # Build features
    X, y = make_features(prices, lags)
    models, weights, avg_error = train_ensemble(X, y)

    # Log returns (NumPy array ONLY)
    returns = np.diff(np.log(prices))

    preds = np.zeros(horizon, dtype=float)

    for i in range(horizon):
        window = returns[-lags:]

        feats = np.array([
            *window,
            np.std(window),
            np.sum(window),
            np.max(window),
            np.min(window),
        ], dtype=float).reshape(1, -1)

        r_hat = 0.0
        for name, model in models.items():
            r_hat += weights[name] * float(model.predict(feats)[0])

        preds[i] = r_hat
        returns = np.append(returns, r_hat)

    # Price forecast
    last_price = float(prices[-1])
    forecast = np.asarray(last_price * np.exp(np.cumsum(preds)), dtype=float)

    # Confidence bands
    sigma = float(np.sqrt(avg_error))
    z = 1.96

    low = forecast * np.exp(-z * sigma)
    high = forecast * np.exp(z * sigma)

    return {
        "method": "ml_ensemble",
        "forecast": forecast.tolist(),
        "low": low.tolist(),
        "high": high.tolist(),
        "vol_est": sigma,
        "last_price": last_price,
        "model_weights": weights,
    }

