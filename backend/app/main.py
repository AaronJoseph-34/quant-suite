from __future__ import annotations

import os
from typing import Annotated, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import brentq
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA

load_dotenv()
from app.data_provider import fetch_market_data
from app.forecast_metrics import backtest_metrics, confidence_band
from app.forecast_ml_ensemble import ml_forecast
from app.portfolio_risk import compute_portfolio_risk, prices_to_returns
from app.risk import single_asset_risk, monte_carlo

app = FastAPI(title="Quant Suite API", version="1.0")

origins = [origin.strip().rstrip("/") for origin in
           os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if origin.strip()]
if not origins or "*" in origins:
    raise RuntimeError("CORS_ORIGINS must list explicit frontend origins")
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=False,
    allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    # Exclude input/context, which can themselves contain non-JSON NaN values.
    errors = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=400, content={"detail": errors})


@app.exception_handler(ValueError)
async def invalid_value(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


def finite_result(value):
    if isinstance(value, dict):
        for child in value.values():
            finite_result(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            finite_result(child)
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError("Inputs produced a non-finite result; reduce their magnitude")
    return value


def validate_option(S, K, T, r, sigma, opt_type):
    if not all(np.isfinite(v) for v in (S, K, T, r, sigma)):
        raise ValueError("Option inputs must be finite")
    if min(S, K, T, sigma) <= 0:
        raise ValueError("S, K, T and sigma must be greater than zero")
    if opt_type not in ("call", "put"):
        raise ValueError("Option type must be call or put")

# -------------------------
# Helpers: core quant math
# -------------------------

def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, opt_type: str) -> float:
    validate_option(S, K, T, r, sigma, opt_type)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "call":
        return finite_result(float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)))
    else:
        return finite_result(float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)))

def black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float, opt_type: str) -> Dict[str, float]:
    validate_option(S, K, T, r, sigma, opt_type)
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
    validate_option(S, K, T, r, 1.0, opt_type)
    discounted_strike = finite_result(float(K * np.exp(-r * T)))
    lower = max(0.0, S - discounted_strike) if opt_type == "call" else max(0.0, discounted_strike - S)
    upper = S if opt_type == "call" else discounted_strike
    if not np.isfinite(target_price) or not lower < target_price < upper:
        raise ValueError(f"Option price must be strictly between {lower:g} and {upper:g}")

    def f(sig):
        return black_scholes_price(S, K, T, r, sig, opt_type) - target_price

    try:
        return float(brentq(f, 1e-6, 5.0))
    except ValueError as exc:
        raise ValueError("Target price is unreachable within volatility bounds [0.000001, 5]") from exc

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

def arima_backtest(prices):
    arr = np.asarray(prices, dtype=float)
    split = int(0.8 * len(arr))
    fitted = ARIMA(np.log(arr[:split]), order=(1, 1, 1)).fit()
    predictions = []
    for price in arr[split:]:
        predictions.append(float(np.exp(fitted.forecast(steps=1)[0])))
        fitted = fitted.append([np.log(price)], refit=False)
    return backtest_metrics(arr[split:], predictions, arr[split - 1:-1])


# -------------------------
# Request/Response models
# -------------------------

OptType = Literal["call", "put"]

Positive = Annotated[float, Field(gt=0)]
PriceSeries = Annotated[List[Positive], Field(min_length=3)]


class RequestModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")


class BSRequest(RequestModel):
    S: Positive
    K: Positive
    T: Positive
    r: float
    sigma: Positive
    type: OptType


class ImpliedVolRequest(RequestModel):
    price: Positive
    S: Positive
    K: Positive
    T: Positive
    r: float
    type: OptType


class HestonRequest(RequestModel):
    S: Positive
    K: Positive
    T: Positive
    r: float
    v0: float = Field(0.04, ge=0)
    kappa: Positive = 2.0
    theta: float = Field(0.04, ge=0)
    vol_of_vol: float = Field(0.5, ge=0)
    rho: float = Field(-0.7, ge=-1, le=1)
    steps: int = Field(252, ge=1, le=1000)
    sims: int = Field(20000, ge=1, le=50000)


class MCRequest(RequestModel):
    S0: Positive
    mu: float
    sigma: float = Field(..., ge=0)
    T: Positive = 1.0
    steps: int = Field(252, ge=1, le=1000)
    sims: int = Field(20000, ge=1, le=50000)
    K: Optional[Positive] = None
    seed: int = Field(123, ge=0)


class ForecastRequest(RequestModel):
    prices: List[Positive] = Field(..., min_length=30)
    horizon: int = Field(5, ge=1, le=60)
    method: Literal["arima", "ml"] = "ml"


class PortfolioRiskRequest(RequestModel):
    prices: List[PriceSeries] = Field(..., min_length=1)
    weights: Optional[List[float]] = None
    alpha: float = Field(0.05, gt=0, lt=0.5)
    window: int = Field(252, ge=2)


class SingleRiskRequest(RequestModel):
    prices: PriceSeries


PortfolioAutoRequest = PortfolioRiskRequest
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
    return finite_result({"price": price, "greeks": greeks})

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
    return finite_result({"price": px, "method": "heston_mc", "steps": req.steps, "sims": req.sims})

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
    return finite_result(out)

@app.post("/forecast")
def forecast(req: ForecastRequest):
    if req.method == "arima":
        values = forecast_arima(req.prices, req.horizon)
        result = {
            "forecast": values, **confidence_band(req.prices, values),
            "last_price": req.prices[-1], "backtest": arima_backtest(req.prices),
        }
    else:
        result = ml_forecast(req.prices, req.horizon)
    return finite_result({**result, "method": req.method, "horizon": req.horizon})


class FetchRequest(RequestModel):
    symbol: str = Field(..., min_length=1, max_length=32, pattern=r"^[A-Za-z0-9.^=\-]+$")
    period: str = "1y"
    interval: str = "1d"


@app.post("/data/fetch")
def data_fetch(req: FetchRequest):
    try:
        return fetch_market_data(req.symbol, req.period, req.interval)
    except RuntimeError:
        return JSONResponse(status_code=502, content={
            "detail": "Market data unavailable for this symbol/range. Check the symbol or retry later.",
        })


@app.post("/risk")
def risk_endpoint(req: SingleRiskRequest):
    return finite_result({"risk": single_asset_risk(req.prices), "mc_paths": monte_carlo(req.prices)})


@app.post("/risk/portfolio")
@app.post("/risk/portfolio/auto")
def portfolio_risk_auto(req: PortfolioRiskRequest):
    return finite_result(compute_portfolio_risk(
        prices_to_returns(req.prices), req.weights, req.alpha, req.window,
    ))
