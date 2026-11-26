import pytest
from fastapi.testclient import TestClient
from app.main import app, black_scholes_price, black_scholes_greeks

client = TestClient(app)
BASE = {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2, "type": "call"}


def test_known_black_scholes_and_put_call_parity():
    call = client.post("/pricing/black_scholes", json=BASE)
    assert call.status_code == 200
    assert call.json()["price"] == pytest.approx(10.45058357)
    assert call.json()["greeks"]["delta"] == pytest.approx(0.63683065)
    put = black_scholes_price(100, 100, 1, 0.05, 0.2, "put")
    import math
    assert call.json()["price"] - put == pytest.approx(100 - 100 * math.exp(-0.05))


@pytest.mark.parametrize("field,value", [("T", 0), ("T", -1), ("sigma", 0), ("S", 0), ("K", -1), ("type", "bad")])
def test_invalid_black_scholes_returns_400(field, value):
    response = client.post("/pricing/black_scholes", json={**BASE, field: value})
    assert response.status_code == 400
    assert "detail" in response.json()


@pytest.mark.parametrize("helper", [black_scholes_price, black_scholes_greeks])
def test_direct_helpers_reject_zero_expiry(helper):
    with pytest.raises(ValueError, match="greater than zero"):
        helper(100, 100, 0, 0.05, 0.2, "call")


def test_non_finite_json_does_not_break_error_serialization():
    response = client.post("/pricing/black_scholes", content='{"S":NaN,"K":100,"T":1,"r":0.05,"sigma":0.2,"type":"call"}', headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert "finite" in response.text.lower()


def test_implied_vol_roundtrip():
    request = {k: v for k, v in BASE.items() if k != "sigma"}
    response = client.post("/pricing/implied_vol", json={**request, "price": 10.450583572185565})
    assert response.status_code == 200
    assert response.json()["implied_vol"] == pytest.approx(0.2)


@pytest.mark.parametrize("changes", [{"T": 0}, {"price": 101}, {"price": 1}, {"price": 99.9, "T": 0.001}])
def test_invalid_or_unreachable_iv_returns_400(changes):
    request = {k: v for k, v in BASE.items() if k != "sigma"}
    response = client.post("/pricing/implied_vol", json={**request, "price": 10, **changes})
    assert response.status_code == 400
    assert "detail" in response.json()


@pytest.mark.parametrize("path,body", [
    ("/pricing/heston_call", {"S": 100, "K": 100, "T": 1, "r": 0.03, "rho": 2}),
    ("/sim/gbm_paths", {"S0": 100, "mu": 0.1, "sigma": 0.2, "steps": 0}),
])
def test_simulation_validation(path, body):
    assert client.post(path, json=body).status_code == 400


def test_cors_restricts_origins():
    allowed = client.options("/forecast", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"})
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-credentials" not in allowed.headers
    denied = client.options("/forecast", headers={"Origin": "https://untrusted.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in denied.headers
