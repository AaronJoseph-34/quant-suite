import os
import time
import requests
import yfinance as yf
from app.cache import get as cache_get, set as cache_set

def fetch_alpha(symbol: str):
    ALPHA_KEY = os.getenv("ALPHA_VANTAGE_KEY")
    if not ALPHA_KEY:
        raise RuntimeError("Alpha Vantage key not set")
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": ALPHA_KEY,
    }
    
    r = requests.get(url, params=params, timeout=10)
    data = r.json()
    
    # DEBUG: Print what we got
    print(f"Alpha Vantage response: {data}")
    
    # Check for specific error messages
    if "Error Message" in data:
        raise RuntimeError(f"Alpha Vantage API error: {data['Error Message']}")
    
    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage rate limit: {data['Note']}")
    
    if "Information" in data:
        raise RuntimeError(f"Alpha Vantage info: {data['Information']}")
    
    if "Time Series (Daily)" not in data:
        raise RuntimeError(f"Alpha Vantage failed: {data}")
    
    series = data["Time Series (Daily)"]
    dates = list(series.keys())[::-1]
    prices = [float(series[d]["4. close"]) for d in dates]
    
    return {
        "symbol": symbol,
        "dates": dates,
        "prices": prices,
        "source": "alpha_vantage",
    }

def fetch_yahoo(symbol: str):
    print(f"Attempting to fetch {symbol} from Yahoo...")
    
    df = yf.download(
        symbol,
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    
    # DEBUG: Print what we got
    print(f"Yahoo returned dataframe: {df}")
    print(f"Is empty? {df.empty if df is not None else 'df is None'}")
    
    if df is None or df.empty:
        raise RuntimeError("Yahoo failed: No data returned")
    
    df = df.reset_index()
    return {
        "symbol": symbol,
        "dates": df["Date"].astype(str).tolist(),
        "prices": df["Close"].astype(float).tolist(),
        "source": "yahoo",
    }

def fetch_market_data(symbol: str):
    symbol = symbol.upper()
    cache_key = f"market:{symbol}"

    # 1️⃣ Check cache first
    cached = cache_get(cache_key)
    if cached:
        cached["cached"] = True
        return cached

    # 2️⃣ Fetch from providers
    try:
        data = fetch_alpha(symbol)
    except Exception as e:
        print(f"Alpha failed with: {e}")
        time.sleep(1)
        data = fetch_yahoo(symbol)

    # 3️⃣ Store in cache
    data["cached"] = False
    cache_set(cache_key, data)

    return data