# Quant Suite

An educational full-stack demo for market data, price forecasting, European options pricing, and portfolio risk analysis. Built with FastAPI, NumPy/SciPy, pandas, statsmodels, scikit-learn, Next.js, TypeScript, and Recharts.

## Features

- Adjusted price history from Alpha Vantage, with Yahoo Finance fallback; range-aware caching for five minutes.
- ARIMA(1,1,1) on log prices, or an ElasticNet/Random Forest ensemble on lagged log returns.
- Approximate 95% forecast bands and a chronological one-step backtest against a random walk (`tomorrow = today`).
- Black-Scholes prices and Greeks, bounded implied-volatility solving, Heston Monte Carlo calls, and GBM simulations.
- Single-asset or multi-asset historical VaR/CVaR, rolling estimates, correlations, and explicit stress scenarios.

These are demonstration analytics, not production trading APIs or evidence of profitable forecasts.

## Run locally

Use Python 3.12 and Node.js 24. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-dev.txt
cp backend/.env.example backend/.env
cd backend
uvicorn app.main:app --reload
```

In a second terminal:

```bash
cd frontend
npm ci
cp .env.example .env.local
npm run dev
```

Open [the dashboard](http://localhost:3000) or [API documentation](http://localhost:8000/docs).

## Configuration

| Variable | Where | Purpose |
| --- | --- | --- |
| `NEXT_PUBLIC_API_BASE` | Frontend `.env.local` / hosting build settings | Backend URL; defaults to `http://localhost:8000`. Set before building Next.js. |
| `CORS_ORIGINS` | Backend `.env` / hosting settings | Comma-separated exact frontend origins, e.g. `https://your-frontend.example`. Defaults to `http://localhost:3000`; wildcards are rejected and credentials are disabled. |
| `ALPHA_VANTAGE_KEY` | Backend `.env` / hosting settings | Optional Alpha Vantage key. Missing key, quota, or plan restrictions trigger Yahoo fallback. |

Restart the backend after configuration changes and rebuild the frontend after changing its public API URL. Environment files and generated build/cache output are ignored by Git.

## Market data behavior

`POST /data/fetch` accepts `symbol`, `period`, and `interval`. Supported periods are `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, and `max`; intervals are `1m`, `5m`, `15m`, `1h`, `1d`, `1wk`, and `1mo`.

Alpha Vantage requests full adjusted daily history, filters it to the requested calendar period, and aggregates weekly/monthly observations. Full adjusted data may require a paid plan; see [Alpha Vantage documentation](https://www.alphavantage.co/documentation/). Yahoo receives the requested period and interval directly, using adjusted close prices. Source, period, interval, count, and adjustment status are returned with the data.

Intraday requests use Yahoo: use `5d` for `1m`, at most `1mo` for `5m`/`15m`, and at most `2y` for `1h`. Unsupported combinations return HTTP 400 instead of silently changing the request. Provider failures return 502. Actual observations depend on trading calendars, listing history, and provider availability; a year of daily data usually contains about 252 prices, not 365. See [yfinance download documentation](https://github.com/ranaroussi/yfinance/blob/main/yfinance/multi.py).

## Forecast methodology and limits

The ensemble predicts log returns using ten lagged returns and window statistics. Inverse-validation-MSE weights combine ElasticNet and Random Forest. The final model is refit on all available observations after weights are selected. Its one-step log-error scale is the ensemble's measured chronological validation RMSE, not an average of model weights.

Both forecast methods use `forecast[h] * exp(±1.96 * sigma * sqrt(h))`, for `h = 1..horizon`. ML uses validation log-error RMSE; ARIMA uses historical log-return volatility. These bands assume independent errors and constant variance. They are heuristic prediction bands, not fitted ARIMA intervals or calibrated coverage guarantees; recursive ML uncertainty can behave differently.

Every forecast response includes a `backtest` with model and random-walk MAE/RMSE in price units, observation count, and RMSE skill (`1 - model_rmse / random_walk_rmse`; positive means better, null for a zero baseline error). For ML, the last 20% of feature targets are held out. Only the preceding 80% fit the evaluation model; an inner chronological split selects weights. Each held-out step uses previously observed returns, with parameters fixed. ARIMA holds out the last 20% of prices, then updates its state with each actual observation without refitting parameters. After evaluation, a separate model uses the full history for the future forecast.

This is a small one-step diagnostic on the current series. It does not establish outperformance across assets or regimes, evaluate transaction costs, or validate a 20-step recursive forecast. Lagged returns often fail to beat the random walk; the dashboard reports the measured errors even when the model loses. A horizon means observations at the input interval, not necessarily days.

## Portfolio risk contract

`POST /risk/portfolio` and `/risk/portfolio/auto` share the same implementation:

```json
{
  "prices": [[100, 110, 99], [200, 180, 198]],
  "weights": [0.75, 0.25],
  "alpha": 0.05,
  "window": 252
}
```

Price input is **assets × time**; internal return matrices are **time × assets**. Series must have equal lengths and shared chronological observation dates. The API has no dates, so callers must align observations before sending them. The dashboard's portfolio CSV path enforces shared dates, sorts them, and rejects missing prices or duplicate dates. Upload a wide CSV under Risk:

```csv
Date,AAPL,MSFT
2026-01-01,100,200
2026-01-02,110,180
2026-01-03,99,198
```

Blank weights mean equal weighting. Supplied weights must be finite, nonnegative, match the number of assets, and have a positive total; they are normalized. Risk uses weighted **simple returns** (`P[t]/P[t-1] - 1`), representing constant weights rebalanced each observation. This avoids treating a weighted sum of log returns as an exact portfolio return. Volatility is the sample standard deviation per observation, not annualized.

`alpha=0.05` is a 5% tail probability / 95% confidence. VaR is the 95th percentile of losses (`-portfolio_return`), and CVaR is the mean loss at or above that threshold. Positive values mean losses; negative values mean gains, and are not clipped. The legacy `/risk` endpoint follows the same convention. Constant-series correlations are undefined and returned as `null`.

Rolling risk uses exactly `window` returns per estimate, includes the final complete window, and returns empty arrays if there is insufficient history. The default 252-return window requires at least 253 prices.

The −5% and −10% scenarios are **one-time immediate price shocks** to all assets, returning portfolio return and loss. They do not subtract a shock from every historical daily return. A separate `volatility_2x` scenario doubles each asset's deviations around its historical mean, preserving mean returns and correlations. These illustrative scenarios do not model changing correlations, nonlinear option exposures, or path-dependent liquidity effects.

## Validation and tests

Invalid numeric inputs, nonpositive prices/expiry, invalid weights, and unreachable implied-volatility targets return HTTP 400 with a `detail` message. Implied volatility is solved within `[0.000001, 5]`; expiry must be strictly positive for prices and Greeks. Vega is per unit volatility and theta is per year.

```bash
# From repository root, with the virtual environment active
python -m pytest -c backend/pytest.ini backend/tests -q

cd frontend
npm test
npm run lint
npx tsc --noEmit
npm run build
```

Tests cover option failures and known values, true ensemble residual error, increasing band widths, time-ordered baseline evaluation, provider ranges/cache keys, multi-asset orientation, VaR sign, rolling windows, stress semantics, and shared-date CSV parsing. CI runs backend tests and frontend checks without live provider credentials.

Removing tracked build files reduces the current source tree; those files remain in existing Git history. This change does not rewrite published history.
