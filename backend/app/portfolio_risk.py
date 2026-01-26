import numpy as np

# ---------------------------
# Helpers
# ---------------------------

def portfolio_returns(returns: np.ndarray, weights: np.ndarray):
    """
    returns: shape (T, N)
    weights: shape (N,)
    """
    return returns @ weights


def var_cvar(portfolio_rets, alpha=0.95):
    """
    Historical VaR & CVaR
    """
    losses = -portfolio_rets
    var = np.quantile(losses, alpha)
    cvar = losses[losses >= var].mean()
    return float(var), float(cvar)


# ---------------------------
# Main API function
# ---------------------------

def compute_portfolio_risk(returns, weights, alpha=0.95):
    returns = np.asarray(returns, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Weights must sum to 1")

    port_rets = portfolio_returns(returns, weights)

    mean_ret = port_rets.mean()
    vol = port_rets.std()

    var, cvar = var_cvar(port_rets, alpha)

    return {
        "mean_return": float(mean_ret),
        "volatility": float(vol),
        "VaR": var,
        "CVaR": cvar,
        "confidence": alpha,
    }
