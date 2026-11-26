"""Forecast diagnostics. Bands assume independent, constant-variance log errors."""
import numpy as np


def confidence_band(prices, forecast, sigma=None):
    values = np.asarray(forecast, dtype=float)
    if sigma is None:
        sigma = float(np.std(np.diff(np.log(prices)), ddof=1))
    spread = 1.96 * sigma * np.sqrt(np.arange(1, len(values) + 1))
    return {
        "low": (values * np.exp(-spread)).tolist(),
        "high": (values * np.exp(spread)).tolist(),
        "vol_est": float(sigma),
    }


def backtest_metrics(actual, predicted, previous):
    actual, predicted, previous = map(np.asarray, (actual, predicted, previous))
    model_error = predicted - actual
    baseline_error = previous - actual
    model_rmse = float(np.sqrt(np.mean(model_error ** 2)))
    baseline_rmse = float(np.sqrt(np.mean(baseline_error ** 2)))
    return {
        "observations": len(actual),
        "horizon": 1,
        "model": {"mae": float(np.mean(np.abs(model_error))), "rmse": model_rmse},
        "random_walk": {
            "mae": float(np.mean(np.abs(baseline_error))), "rmse": baseline_rmse,
        },
        "rmse_skill": 1 - model_rmse / baseline_rmse if baseline_rmse > 0 else None,
    }
