"""Legacy single-asset endpoint, sharing the portfolio module's risk math."""
import numpy as np
from app.portfolio_risk import compute_portfolio_risk, prices_to_returns


def monte_carlo(prices, days=252, sims=5000):
    prices_to_returns(prices)  # validate before taking logs
    returns = np.diff(np.log(prices))
    rng = np.random.default_rng(42)
    shocks = rng.normal(returns.mean(), returns.std(ddof=1), (sims, days))
    return (prices[-1] * np.exp(np.cumsum(shocks, axis=1))).tolist()


def single_asset_risk(prices):
    stats = compute_portfolio_risk(prices_to_returns(prices))["summary"]
    return {"volatility": stats["vol"], "var_95": stats["VaR"], "cvar_95": stats["CVaR"]}
