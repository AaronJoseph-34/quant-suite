from __future__ import annotations
from app.data_provider import fetch_market_data
from dotenv import load_dotenv
load_dotenv(".env")
from app.portfolio_risk import compute_portfolio_risk
from pydantic import BaseModel, Field
from typing import List, Optional
from typing import List, Literal, Optional, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yfinance as yf
from app.forecast_ml_ensemble import ml_forecast
from typing import List, Optional
from pydantic import BaseModel

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


app = FastAPI(title="Quant Suite API", version="1.0")

# Allow Next.js dev server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],  # CRITICAL: Must allow all methods including OPTIONS
    allow_headers=["*"],  # CRITICAL: Must allow all headers
)

# -------------------------
# Helpers: core quant math
# -------------------------

def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, opt_type: str) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float, opt_type: str) -> Dict[str, float]:
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    pdf_d1 = norm.pdf(d1)

    delta = norm.cdf(d1) if opt_type == "call" else norm.cdf(d1) - 1
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega  = S * pdf_d1 * np.sqrt(T)  # per 1.0 change in vol (not 1%)
    # Theta (per year); sign convention often varies
    if opt_type == "call":
        theta = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r*T) * norm.cdf(d2)
    else:
        theta = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r*T) * norm.cdf(-d2)

    rho = K * T * np.exp(-r*T) * (norm.cdf(d2) if opt_type == "call" else -norm.cdf(-d2))

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }

def implied_volatility(target_price: float, S: float, K: float, T: float, r: float, opt_type: str) -> float:
    # Solve for sigma in [1e-6, 5.0]
    def f(sig: float) -> float:
        return black_scholes_price(S, K, T, r, sig, opt_type) - target_price

    try:
        return float(brentq(f, 1e-6, 5.0))
    except Exception:
        return float("nan")

# -------------------------
# Heston (Monte Carlo, feasible)
# -------------------------

def heston_mc_call(
    S0: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    vol_of_vol: float,
    rho: float,
    steps: int = 252,
    sims: int = 20000,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    dt = T / steps

    # Simulate correlated normals
    z1 = rng.standard_normal((sims, steps))
    z2 = rng.standard_normal((sims, steps))
    z2 = rho * z1 + np.sqrt(1 - rho**2) * z2

    S = np.full(sims, S0, dtype=float)
    v = np.full(sims, v0, dtype=float)

    for t in range(steps):
        # variance process (full truncation Euler)
        v = np.maximum(v + kappa*(theta - v)*dt + vol_of_vol*np.sqrt(np.maximum(v, 0)*dt)*z2[:, t], 0.0)
        # price process
        S = S * np.exp((r - 0.5*v)*dt + np.sqrt(np.maximum(v, 0)*dt)*z1[:, t])

    payoff = np.maximum(S - K, 0.0)
    return float(np.exp(-r*T) * payoff.mean())

# -------------------------
# Monte Carlo (GBM paths)
# -------------------------

def gbm_paths(
    S0: float, mu: float, sigma: float, T: float, steps: int, sims: int, seed: int = 123
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = T / steps
    paths = np.zeros((sims, steps+1), dtype=float)
    paths[:, 0] = S0

    z = rng.standard_normal((sims, steps))
    for t in range(1, steps+1):
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z[:, t-1])

    return paths

# -------------------------
# Forecasting: ARIMA + ML (lag features)
# -------------------------

def forecast_arima(prices: List[float], horizon: int = 5) -> List[float]:
    s = pd.Series(prices).astype(float)
    # ARIMA on log prices is often more stable
    logp = np.log(s.replace(0, np.nan).dropna())
    model = ARIMA(logp, order=(1,1,1))
    fit = model.fit()
    fc = fit.forecast(steps=horizon)
    return list(np.exp(fc).astype(float))

def make_lag_features(prices: np.ndarray, lags: int = 10):
    X, y = [], []
    for i in range(lags, len(prices)):
        X.append(prices[i-lags:i])
        y.append(prices[i])
    return np.array(X, dtype=float), np.array(y, dtype=float)

def forecast_ml_lags(prices: List[float], horizon: int = 5, lags: int = 10) -> List[float]:
    arr = np.array(prices, dtype=float)
    X, y = make_lag_features(arr, lags=lags)
    if len(X) < 20:
        # not enough data; fallback: repeat last
        return [float(arr[-1])] * horizon

    # A simple but real ML pipeline: standardize + ridge regression
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])
    model.fit(X, y)

    preds = []
    window = arr[-lags:].copy()
    for _ in range(horizon):
        next_price = float(model.predict(window.reshape(1, -1))[0])
        preds.append(next_price)
        window = np.roll(window, -1)
        window[-1] = next_price

    return preds

def confidence_band(prices, forecast):
    import numpy as np

    prices = np.asarray(prices, dtype=float)

    # Handle both ARIMA (list) and ML (dict) outputs
    if isinstance(forecast, dict):
        forecast_vals = forecast["forecast"]
    else:
        forecast_vals = forecast

    forecast_vals = np.asarray(forecast_vals, dtype=float)

    # Volatility estimate from log returns
    log_returns = np.diff(np.log(prices))
    vol = float(np.std(log_returns))

    z = 1.96  # 95% confidence
    spread = z * vol

    low = forecast_vals * np.exp(-spread)
    high = forecast_vals * np.exp(spread)

    return {
        "low": low.tolist(),
        "high": high.tolist(),
        "vol_est": vol,
    }
# -------------------------
# Portfolio Risk Helpers
# -------------------------
def compute_log_returns(prices: list[float]) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    return np.diff(np.log(prices))
def rolling_VaR(returns, weights, alpha=0.05, window=30):
    returns = np.asarray(returns, dtype=float)
    weights = np.asarray(weights, dtype=float)

    portfolio_returns = returns @ weights

    if len(portfolio_returns) < window:
        return [], []

    vars_, cvars_ = [], []

    for i in range(window, len(portfolio_returns)):
        window_slice = portfolio_returns[i - window : i]

        var = np.quantile(window_slice, alpha)
        cvar = window_slice[window_slice <= var].mean()

        vars_.append(float(var))
        cvars_.append(float(cvar))

    return vars_, cvars_

def build_returns_matrix(prices: list[list[float]]) -> np.ndarray:
    returns = [compute_log_returns(p) for p in prices]
    min_len = min(len(r) for r in returns)
    returns = [r[-min_len:] for r in returns]
    return np.column_stack(returns)
def portfolio_var_cvar(returns: np.ndarray, weights: list[float], alpha=0.95):
    weights = np.asarray(weights, dtype=float)
    portfolio_returns = returns @ weights

    var = np.percentile(portfolio_returns, (1 - alpha) * 100)
    cvar = portfolio_returns[portfolio_returns <= var].mean()

    return {
        "VaR": float(var),
        "CVaR": float(cvar),
        "mean": float(portfolio_returns.mean()),
        "vol": float(portfolio_returns.std()),
    }
def prices_to_returns(prices):
    prices = np.asarray(prices, dtype=float)

    # Handle 1D list → (T, 1)
    if prices.ndim == 1:
        prices = prices.reshape(-1, 1)

    # Handle (1, T) → (T, 1)
    if prices.shape[0] == 1 and prices.shape[1] > 1:
        prices = prices.T

    if prices.shape[0] < 2:
        raise ValueError("Need at least 2 price observations to compute returns")

    returns = np.diff(np.log(prices), axis=0)

    return returns





def normalize_weights(weights, n_assets):
    if weights is None or len(weights) != n_assets:
        return np.ones(n_assets) / n_assets

    w = np.asarray(weights, dtype=float)
    return w / w.sum()
def compute_portfolio_risk(returns, weights, alpha=0.05):
    returns = np.asarray(returns, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if returns.ndim != 2:
        raise ValueError("Returns must be 2D: (time, assets)")

    if returns.shape[0] < 2:
        raise ValueError("Not enough return observations")

    # Normalize weights
    weights = weights / np.sum(weights)

    # Portfolio returns
    portfolio_returns = returns @ weights

    # ------------------
    # Summary statistics
    # ------------------
    mu = np.mean(portfolio_returns)
    sigma = np.std(portfolio_returns, ddof=1)

    VaR = np.quantile(portfolio_returns, alpha)
    CVaR = portfolio_returns[portfolio_returns <= VaR].mean()

    # ------------------
    # Rolling risk
    # ------------------
    rolling_var, rolling_cvar = rolling_VaR(
        returns=returns,
        weights=weights,
        alpha=alpha,
        window=30,
    )

    # ------------------
    # Correlation
    # ------------------
    corr = np.corrcoef(returns.T)

    # ------------------
    # Stress tests
    # ------------------
    stress_results = {
        "shock_-5pct": {
            "mean": float(np.mean(portfolio_returns - 0.05)),
            "vol": float(np.std(portfolio_returns - 0.05)),
            "VaR": float(np.quantile(portfolio_returns - 0.05, alpha)),
        },
        "shock_-10pct": {
            "mean": float(np.mean(portfolio_returns - 0.10)),
            "vol": float(np.std(portfolio_returns - 0.10)),
            "VaR": float(np.quantile(portfolio_returns - 0.10, alpha)),
        },
    }

    # ------------------
    # Final response
    # ------------------
    return {
        "summary": {
            "mean": float(mu),
            "vol": float(sigma),
            "VaR": float(VaR),
            "CVaR": float(CVaR),
        },
        "rolling": {
            "VaR": rolling_var,
            "CVaR": rolling_cvar,
        },
        "correlation": corr.tolist(),
        "stress": stress_results,
    }

def rolling_var_cvar(returns, weights, alpha, window):
    port = returns.T @ weights
    vars_, cvars_ = [], []

    for i in range(window, len(port)):
        slice_ = port[i - window : i]
        var = np.percentile(slice_, 100 * alpha)
        cvar = slice_[slice_ <= var].mean()
        vars_.append(var)
        cvars_.append(cvar)

    return vars_, cvars_
def correlation_matrix(returns):
    return np.corrcoef(returns)
def stress_test(returns, weights):
    shocks = {
        "market_crash_-20%": returns * 0.8,
        "vol_spike_2x": returns * 2.0,
        "mild_drawdown_-10%": returns * 0.9,
    }

    results = {}
    for name, shocked in shocks.items():
        port = shocked.T @ weights
        results[name] = {
            "mean": float(np.mean(port)),
            "vol": float(np.std(port)),
            "VaR": float(np.percentile(port, 5)),
        }

    return results

# -------------------------
# Request/Response models
# -------------------------

OptType = Literal["call", "put"]

class BSRequest(BaseModel):
    S: float
    K: float
    T: float
    r: float
    sigma: float
    type: OptType

class ImpliedVolRequest(BaseModel):
    price: float
    S: float
    K: float
    T: float
    r: float
    type: OptType

class HestonRequest(BaseModel):
    S: float
    K: float
    T: float
    r: float
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    vol_of_vol: float = 0.5
    rho: float = -0.7
    steps: int = 252
    sims: int = 20000

class MCRequest(BaseModel):
    S0: float
    mu: float
    sigma: float
    T: float = 1.0
    steps: int = 252
    sims: int = 20000
    K: Optional[float] = None  # for payoff simulation (call payoff)
    seed: int = 123

class ForecastRequest(BaseModel):
    prices: List[float] = Field(..., min_items=30)
    horizon: int = Field(5, ge=1, le=60)
    method: Literal["arima", "ml"] = "ml"

class PortfolioRiskRequest(BaseModel):
    prices: List[List[float]]
    weights: Optional[List[float]] = None
    alpha: float = 0.05

class PortfolioAutoRequest(BaseModel):
    prices: List[List[float]]  # [[asset1 prices], [asset2 prices], ...]
    weights: Optional[List[float]] = None
    alpha: float = Field(0.05, gt=0, lt=0.5)
    window: int = 252  # rolling window
# -------------------------
# API endpoints
# -------------------------

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/pricing/black_scholes")
def pricing_black_scholes(req: BSRequest):
    price = black_scholes_price(req.S, req.K, req.T, req.r, req.sigma, req.type)
    greeks = black_scholes_greeks(req.S, req.K, req.T, req.r, req.sigma, req.type)
    return {"price": price, "greeks": greeks}

@app.post("/pricing/implied_vol")
def pricing_implied_vol(req: ImpliedVolRequest):
    iv = implied_volatility(req.price, req.S, req.K, req.T, req.r, req.type)
    return {"implied_vol": iv}

@app.post("/pricing/heston_call")
def pricing_heston_call(req: HestonRequest):
    px = heston_mc_call(
        S0=req.S, K=req.K, T=req.T, r=req.r,
        v0=req.v0, kappa=req.kappa, theta=req.theta,
        vol_of_vol=req.vol_of_vol, rho=req.rho,
        steps=req.steps, sims=req.sims
    )
    return {"price": px, "method": "heston_mc", "steps": req.steps, "sims": req.sims}

@app.post("/sim/gbm_paths")
def sim_gbm_paths(req: MCRequest):
    paths = gbm_paths(req.S0, req.mu, req.sigma, req.T, req.steps, req.sims, seed=req.seed)
    # Optional: call payoff distribution at maturity
    payoff = None
    if req.K is not None:
        payoff = np.maximum(paths[:, -1] - req.K, 0.0)
    # Return summary (not the full matrix to keep API light)
    out = {
        "S0": req.S0,
        "terminal_mean": float(paths[:, -1].mean()),
        "terminal_p05": float(np.percentile(paths[:, -1], 5)),
        "terminal_p95": float(np.percentile(paths[:, -1], 95)),
    }
    if payoff is not None:
        out.update({
            "payoff_mean": float(payoff.mean()),
            "payoff_p95": float(np.percentile(payoff, 95)),
        })
    return out

@app.post("/forecast")
def forecast(req: ForecastRequest):
    if req.method == "arima":
        fc_vals = forecast_arima(req.prices, horizon=req.horizon)
    else:
        fc = ml_forecast(req.prices, horizon=req.horizon)
        fc_vals = fc["forecast"]

    band = confidence_band(req.prices, fc_vals)

    return {
        "method": req.method,
        "horizon": req.horizon,
        "forecast": fc_vals,          # <-- ALWAYS array
        "low": band["low"],            # <-- ALWAYS array
        "high": band["high"],          # <-- ALWAYS array
        "vol_est": band["vol_est"],
        "last_price": float(req.prices[-1]),
    }

class FetchRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    period: str = Field("1y")          # e.g. 1mo, 3mo, 6mo, 1y, 2y, 5y, max
    interval: str = Field("1d")        # e.g. 1m, 5m, 15m, 1h, 1d, 1wk


@app.post("/data/fetch")
def data_fetch(req: FetchRequest):
    return fetch_market_data(req.symbol)


    # --- Safety rules for Yahoo Finance ---
    interval_limits = {
        "1m": "7d",
        "2m": "60d",
        "5m": "60d",
        "15m": "60d",
        "30m": "60d",
        "60m": "730d",
        "90m": "60d",
        "1h": "730d",
    }

    period = req.period
    interval = req.interval

    # Auto-fix invalid intraday requests
    if interval in interval_limits:
        max_period = interval_limits[interval]
        if period not in ["1d", "5d", "7d", "1mo", "3mo"] and max_period == "60d":
            period = max_period

    try:
        df = yf.download(
            tickers=sym,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            return {
                "symbol": sym,
                "dates": [],
                "prices": [],
                "error": f"No data returned for {sym}. Try interval=1d with period=1y.",
            }

        df = df.reset_index()

        # Detect date column
        date_col = "Date" if "Date" in df.columns else "Datetime"
        close_col = "Close"

        dates = [str(x) for x in df[date_col].astype(str).tolist()]
        prices = [float(x) for x in df[close_col].astype(float).tolist()]

        return {
            "symbol": sym,
            "dates": dates,
            "prices": prices,
            "interval": interval,
            "period": period,
            "count": len(prices),
        }

    except Exception as e:
        return {
            "symbol": sym,
            "dates": [],
            "prices": [],
            "error": str(e),
        }
from app.risk import portfolio_risk, monte_carlo

@app.post("/risk")
def risk_endpoint(data: dict):
    prices = data["prices"]
    return {
        "risk": portfolio_risk(prices),
        "mc_paths": monte_carlo(prices),
    }
@app.post("/risk/portfolio")
def portfolio_risk(req: PortfolioRiskRequest):
    return compute_portfolio_risk(
        returns=req.returns,
        weights=req.weights,
        alpha=req.alpha,
    )
@app.post("/risk/portfolio/auto")
def portfolio_risk_auto(req: PortfolioAutoRequest):
    returns = prices_to_returns(req.prices)

    if returns.shape[0] < 10:
        raise ValueError("Need at least 10 return observations for risk metrics")

    weights = normalize_weights(req.weights, returns.shape[1])

    return compute_portfolio_risk(
        returns=returns,
        weights=weights,
        alpha=req.alpha,
    )
