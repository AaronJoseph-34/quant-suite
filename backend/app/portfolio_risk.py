"""Historical risk using simple returns and loss-positive VaR/CVaR.

Price input is (assets, time); return matrices are (time, assets).
Alpha is the lower-tail probability (0.05 = 95% confidence).
"""
import numpy as np


def prices_to_returns(prices):
    try:
        prices = np.asarray(prices, dtype=float)
    except ValueError as exc:
        raise ValueError("Asset price series must have equal lengths and aligned dates") from exc
    if prices.ndim == 1:
        prices = prices.reshape(1, -1)
    if prices.ndim != 2 or prices.shape[0] < 1 or prices.shape[1] < 3:
        raise ValueError("Prices must have shape (assets, time), with at least 3 prices per asset")
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("Prices must be finite and positive")
    return (prices[:, 1:] / prices[:, :-1] - 1).T


def normalize_weights(weights, n_assets):
    if weights is None:
        return np.full(n_assets, 1 / n_assets)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n_assets,) or not np.all(np.isfinite(weights)):
        raise ValueError("Provide one finite weight per asset")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Long-only weights must be nonnegative with a positive sum")
    return weights / weights.sum()


def var_cvar(portfolio_rets, alpha=0.05):
    losses = -np.asarray(portfolio_rets)
    var = np.quantile(losses, 1 - alpha)
    cvar = losses[losses >= var].mean()
    return float(var), float(cvar)


def summary(port, alpha):
    var, cvar = var_cvar(port, alpha)
    return {
        "mean": float(np.mean(port)), "vol": float(np.std(port, ddof=1)),
        "VaR": var, "CVaR": cvar,
    }


def compute_portfolio_risk(returns, weights=None, alpha=0.05, window=252):
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 2 or returns.shape[0] < 2 or returns.shape[1] < 1:
        raise ValueError("Returns must have shape (time, assets), with at least 2 observations")
    if not np.all(np.isfinite(returns)) or np.any(returns <= -1):
        raise ValueError("Simple returns must be finite and greater than -1")
    if not 0 < alpha < 0.5 or window < 2:
        raise ValueError("Alpha must be in (0, 0.5) and rolling window at least 2")
    weights = normalize_weights(weights, returns.shape[1])
    port = returns @ weights
    rolling = [var_cvar(port[end - window:end], alpha)
               for end in range(window, len(port) + 1)]
    # A constant asset has undefined correlation: encode it as null.
    std = returns.std(axis=0, ddof=1)
    denom = np.outer(std, std)
    corr = np.full(denom.shape, np.nan)
    np.divide(np.atleast_2d(np.cov(returns, rowvar=False)), denom, out=corr, where=denom > 0)
    correlation = [[float(v) if np.isfinite(v) else None for v in row] for row in corr]
    stress = {}
    for shock in (-0.05, -0.10):
        stress[f"shock_{int(shock * 100)}pct"] = {
            "type": "one_time_price_shock", "shock": shock,
            "portfolio_return": float(np.full(len(weights), shock) @ weights),
            "loss": float(-shock),
        }
    stressed_port = (returns.mean(axis=0) + 2 * (returns - returns.mean(axis=0))) @ weights
    stress["volatility_2x"] = {"type": "volatility_scale", **summary(stressed_port, alpha)}
    return {
        "summary": summary(port, alpha),
        "rolling": {
            "VaR": [r[0] for r in rolling], "CVaR": [r[1] for r in rolling], "window": window,
        },
        "correlation": correlation, "stress": stress, "weights": weights.tolist(),
        "confidence": 1 - alpha, "return_type": "simple", "loss_convention": "positive_is_loss",
    }
