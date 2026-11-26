import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.portfolio_risk import compute_portfolio_risk, prices_to_returns

client = TestClient(app)


def test_multi_asset_axis_and_weighted_simple_returns():
    returns = prices_to_returns([[100, 110, 99], [200, 180, 198]])
    np.testing.assert_allclose(returns, [[0.1, -0.1], [-0.1, 0.1]], atol=1e-14)
    result = compute_portfolio_risk(returns, [3, 1], window=2)
    assert result["summary"]["mean"] == pytest.approx(0, abs=1e-14)
    assert result["summary"]["VaR"] == pytest.approx(0.045)
    assert result["summary"]["CVaR"] == pytest.approx(0.05)
    np.testing.assert_allclose(result["correlation"], [[1, -1], [-1, 1]])
    assert result["weights"] == [0.75, 0.25]


@pytest.mark.parametrize("path", ["/risk/portfolio", "/risk/portfolio/auto"])
def test_both_portfolio_routes_use_prices_and_same_math(path):
    response = client.post(path, json={"prices": [[100, 90, 81], [100, 110, 121]], "weights": [0.75, 0.25], "window": 2})
    assert response.status_code == 200
    assert response.json()["summary"]["VaR"] == pytest.approx(0.05)
    assert len(response.json()["rolling"]["VaR"]) == 1


def test_rolling_window_is_used_including_last_observation():
    returns = np.linspace(-0.1, 0.1, 260).reshape(-1, 1)
    result = compute_portfolio_risk(returns, window=252)
    assert len(result["rolling"]["VaR"]) == 9
    assert result["rolling"]["VaR"][-1] == pytest.approx(np.quantile(-returns[-252:, 0], 0.95))
    assert compute_portfolio_risk(returns, window=300)["rolling"]["VaR"] == []


def test_one_time_shock_and_volatility_scale():
    returns = np.array([[-0.02], [0.01], [-0.01], [0.03]])
    result = compute_portfolio_risk(returns)
    assert result["stress"]["shock_-5pct"]["portfolio_return"] == -0.05
    assert result["stress"]["shock_-10pct"]["loss"] == 0.1
    assert result["stress"]["volatility_2x"]["vol"] == pytest.approx(2 * result["summary"]["vol"])
    assert result["stress"]["volatility_2x"]["mean"] == pytest.approx(result["summary"]["mean"])


def test_constant_and_single_asset_correlations_are_finite_json():
    response = client.post("/risk/portfolio/auto", json={"prices": [[100, 100, 100]]})
    assert response.status_code == 200
    assert response.json()["correlation"] == [[None]]
    response = client.post("/risk/portfolio/auto", json={"prices": [[100, 110, 90, 95]]})
    assert response.json()["correlation"][0][0] == pytest.approx(1.0)


@pytest.mark.parametrize("changes", [
    {"weights": [0, 0]}, {"weights": [-1, 2]}, {"weights": [1]},
    {"prices": [[100, 101, 102], [100, 101]]}, {"prices": []},
    {"prices": [[100, 0, 102]]}, {"alpha": 0}, {"window": 1},
])
def test_invalid_risk_inputs_return_400(changes):
    response = client.post("/risk/portfolio/auto", json={"prices": [[100, 99, 101], [100, 101, 99]], **changes})
    assert response.status_code == 400


def test_loss_convention_all_gains_can_have_negative_var():
    result = compute_portfolio_risk(np.array([[0.01], [0.02], [0.03]]))
    assert result["summary"]["VaR"] < 0  # negative loss is a gain; do not clip it
