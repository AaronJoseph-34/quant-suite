"""Market data with explicit ranges, provider limits, and request-aware caching."""
import logging
import os

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from app.cache import get as cache_get, set as cache_set

logger = logging.getLogger(__name__)
PERIODS = {"1d": 1, "5d": 5, "1mo": 31, "3mo": 93, "6mo": 186,
           "1y": 366, "2y": 732, "5y": 1830, "max": None}
INTERVALS = {"1d", "1wk", "1mo", "1h", "15m", "5m", "1m"}


def validate_range(period, interval):
    if period not in PERIODS or interval not in INTERVALS:
        raise ValueError("Unsupported period or interval")
    days = PERIODS[period]
    limit = 7 if interval == "1m" else 60 if interval in {"5m", "15m"} else 732 if interval == "1h" else None
    if limit is not None and (days is None or days > limit):
        raise ValueError(f"Interval {interval} supports at most {limit} days; choose a shorter period")


def period_start(period):
    if period == "max":
        return None
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    if period.endswith("mo"):
        return now - pd.DateOffset(months=int(period[:-2]))
    if period.endswith("y"):
        return now - pd.DateOffset(years=int(period[:-1]))
    return now - pd.Timedelta(days=int(period[:-1]))


def market_result(symbol, period, interval, close, source):
    close = close.sort_index().dropna()
    if close.empty or not np.isfinite(close.to_numpy()).all() or (close <= 0).any():
        raise RuntimeError("Provider returned no usable prices")
    return {
        "symbol": symbol, "dates": close.index.astype(str).tolist(),
        "prices": close.astype(float).tolist(), "source": source,
        "period": period, "interval": interval, "count": len(close), "adjusted": True,
    }


def fetch_alpha(symbol, period="1y", interval="1d"):
    if interval not in {"1d", "1wk", "1mo"}:
        raise RuntimeError("Alpha daily provider does not serve intraday requests")
    key = os.getenv("ALPHA_VANTAGE_KEY")
    if not key:
        raise RuntimeError("Alpha Vantage key not set")
    response = requests.get("https://www.alphavantage.co/query", params={
        "function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": symbol,
        "outputsize": "full", "apikey": key,
    }, timeout=15)
    response.raise_for_status()
    series = response.json().get("Time Series (Daily)")
    if not series:
        raise RuntimeError("Alpha Vantage data unavailable (plan, quota, or symbol)")
    close = pd.Series({pd.Timestamp(date): float(row["5. adjusted close"])
                       for date, row in series.items()}).sort_index()
    # Aggregate before cutting the range; retain each bar's last observation date.
    if interval in {"1wk", "1mo"}:
        frame = close.to_frame("price")
        frame["date"] = frame.index
        frame = frame.resample("W-FRI" if interval == "1wk" else "ME").last().dropna()
        close = frame.set_index("date")["price"]
    start = period_start(period)
    if start is not None:
        close = close.loc[close.index >= start]
    return market_result(symbol, period, interval, close, "alpha_vantage")


def fetch_yahoo(symbol, period="1y", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True,
                     progress=False, threads=False, timeout=15)
    if df is None or df.empty:
        raise RuntimeError("Yahoo returned no data")
    close = df["Close"]
    # yfinance returns a ticker column level even for a single symbol.
    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise RuntimeError("Expected exactly one ticker")
        close = close.iloc[:, 0]
    return market_result(symbol, period, interval, close, "yahoo")


def fetch_market_data(symbol, period="1y", interval="1d"):
    validate_range(period, interval)
    symbol = symbol.strip().upper()
    cache_key = f"market:{symbol}:{period}:{interval}"
    cached = cache_get(cache_key)
    if cached is not None:
        return {**cached, "cached": True}
    data = None
    providers = (fetch_alpha, fetch_yahoo) if interval in {"1d", "1wk", "1mo"} else (fetch_yahoo,)
    for provider in providers:
        try:
            data = provider(symbol, period, interval)
            break
        except (RuntimeError, requests.RequestException, ValueError, KeyError, TypeError):
            # Never log response bodies, URLs containing API keys, or user data.
            logger.info("Market provider unavailable: %s", provider.__name__)
    if data is None:
        raise RuntimeError("All market providers unavailable")
    data["cached"] = False
    cache_set(cache_key, data)
    return data
