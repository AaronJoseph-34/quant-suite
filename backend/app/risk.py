import numpy as np
from typing import List, Dict

def compute_returns(prices: List[float]) -> np.ndarray:
    prices = np.array(prices)
    return np.diff(np.log(prices))


def var_cvar(returns: np.ndarray, alpha=0.95):
    sorted_r = np.sort(returns)
    index = int((1 - alpha) * len(sorted_r))
    var = sorted_r[index]
    cvar = sorted_r[:index].mean()
    return float(var), float(cvar)


def monte_carlo(prices: List[float], days=252, sims=5000):
    returns = compute_returns(prices)
    mu = returns.mean()
    sigma = returns.std()

    last_price = prices[-1]
    paths = []

    for _ in range(sims):
        shocks = np.random.normal(mu, sigma, days)
        path = last_price * np.exp(np.cumsum(shocks))
        paths.append(path.tolist())

    return paths


def portfolio_risk(prices: List[float]) -> Dict:
    returns = compute_returns(prices)
    var, cvar = var_cvar(returns)

    return {
        "volatility": float(np.std(returns)),
        "var_95": var,
        "cvar_95": cvar,
    }
