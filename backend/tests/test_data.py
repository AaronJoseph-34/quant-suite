import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app import cache, data_provider as data

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_cache():
    cache._CACHE.clear()


def test_endpoint_passes_picker_values(monkeypatch):
    captured = []
    monkeypatch.setattr("app.main.fetch_market_data", lambda *args: captured.append(args) or {"prices": []})
    assert client.post("/data/fetch", json={"symbol": "AAPL", "period": "2y", "interval": "1wk"}).status_code == 200
    assert captured == [("AAPL", "2y", "1wk")]


def test_cache_varies_by_period_and_interval_without_mutation(monkeypatch):
    calls = []
    def provider(symbol, period, interval):
        calls.append((symbol, period, interval))
        return {"prices": [100, 101], "period": period, "interval": interval}
    monkeypatch.setattr(data, "fetch_alpha", provider)
    first = data.fetch_market_data("aapl", "1y", "1d")
    first["prices"][0] = 0
    assert data.fetch_market_data("aapl", "1y", "1d")["prices"][0] == 100
    assert data.fetch_market_data("aapl", "1y", "1d")["cached"] is True
    data.fetch_market_data("aapl", "2y", "1d")
    data.fetch_market_data("aapl", "1y", "1wk")
    assert len(calls) == 3


def test_alpha_requests_full_adjusted_history_and_respects_period_interval(monkeypatch):
    dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=800)
    series = {str(d.date()): {"5. adjusted close": str(100 + i / 10)} for i, d in enumerate(dates)}
    captured = []
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"Time Series (Daily)": series}
    def request(url, **kwargs):
        captured.append(kwargs["params"])
        return Response()
    monkeypatch.setenv("ALPHA_VANTAGE_KEY", "test-key")
    monkeypatch.setattr(data.requests, "get", request)
    year = data.fetch_alpha("AAPL", "1y", "1d")
    quarter = data.fetch_alpha("AAPL", "3mo", "1d")
    weekly = data.fetch_alpha("AAPL", "1y", "1wk")
    monthly = data.fetch_alpha("AAPL", "1y", "1mo")
    assert 250 <= year["count"] <= 265
    assert quarter["count"] < 70
    assert 50 <= weekly["count"] <= 54
    assert 11 <= monthly["count"] <= 13
    assert captured[0]["outputsize"] == "full"
    assert captured[0]["function"] == "TIME_SERIES_DAILY_ADJUSTED"


def test_yahoo_handles_multiindex_and_intraday_index(monkeypatch):
    captured = []
    dates = pd.date_range("2026-01-01", periods=3, freq="h", name="Datetime", tz="UTC")
    frame = pd.DataFrame(np.array([[100], [101], [102]]), index=dates,
                         columns=pd.MultiIndex.from_tuples([("Close", "AAPL")]))
    monkeypatch.setattr(data.yf, "download", lambda *a, **kw: captured.append(kw) or frame)
    result = data.fetch_yahoo("AAPL", "1mo", "1h")
    assert result["prices"] == [100, 101, 102]
    assert captured[0]["period"] == "1mo"
    assert captured[0]["interval"] == "1h"


@pytest.mark.parametrize("period,interval", [("1y", "1m"), ("3mo", "5m"), ("5y", "1h"), ("bad", "1d"), ("1y", "bad")])
def test_unsupported_ranges_return_400(period, interval):
    assert client.post("/data/fetch", json={"symbol": "AAPL", "period": period, "interval": interval}).status_code == 400


def test_provider_failure_returns_safe_502(monkeypatch):
    def fail(*args):
        raise RuntimeError("secret API key and raw response")
    monkeypatch.setattr(data, "fetch_alpha", fail)
    monkeypatch.setattr(data, "fetch_yahoo", fail)
    response = client.post("/data/fetch", json={"symbol": "AAPL"})
    assert response.status_code == 502
    assert "secret" not in response.text
