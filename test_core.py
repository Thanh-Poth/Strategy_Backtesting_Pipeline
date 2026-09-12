"""
Tests de non-régression rapides, sur données synthétiques (pas d'accès
réseau requis). Lancer avec : pytest tests/
"""
import numpy as np
import pandas as pd
import pytest

from sizing import risk_parity_weights, kelly_weights, drawdown_throttle
from risk import historical_var, cvar, max_drawdown, drawdown_series
from portfolio import PortfolioConfig, construct_weights
from backtest import run_walk_forward_backtest
from signals import combined_alpha, make_walk_forward_predictor


@pytest.fixture
def synthetic_prices():
    np.random.seed(0)
    dates = pd.bdate_range("2015-01-01", periods=500)
    tickers = ["A", "B", "C", "D"]
    rets = np.random.normal(0.0003, 0.01, size=(500, 4))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=dates, columns=tickers)


def test_risk_parity_weights_sum_to_one_and_are_positive(synthetic_prices):
    cov = synthetic_prices.pct_change().dropna().cov()
    w = risk_parity_weights(cov)
    assert np.isclose(w.sum(), 1.0, atol=1e-4)
    assert (w > 0).all()


def test_risk_parity_equalizes_risk_contribution(synthetic_prices):
    cov = synthetic_prices.pct_change().dropna().cov()
    w = risk_parity_weights(cov)
    sigma = cov.values
    marginal_contrib = sigma @ w
    risk_contrib = w * marginal_contrib
    # les contributions au risque doivent être proches les unes des autres
    assert np.std(risk_contrib) / np.mean(risk_contrib) < 0.05


def test_kelly_respects_leverage_cap(synthetic_prices):
    returns = synthetic_prices.pct_change().dropna()
    cov = returns.cov()
    mu = returns.mean()
    w = kelly_weights(mu, cov, kelly_fraction=1.0, gross_leverage_cap=1.5)
    assert np.abs(w).sum() <= 1.5 + 1e-6


def test_drawdown_throttle_bounds():
    assert drawdown_throttle(0.0, dd_limit=0.15, min_exposure=0.25) == 1.0
    assert drawdown_throttle(-0.30, dd_limit=0.15, min_exposure=0.25) == 0.25
    mid = drawdown_throttle(-0.075, dd_limit=0.15, min_exposure=0.25)
    assert 0.25 < mid < 1.0


def test_max_drawdown_and_var_are_non_negative():
    equity = pd.Series([1.0, 1.1, 0.9, 1.05, 0.8, 1.2])
    assert max_drawdown(equity) > 0
    returns = equity.pct_change().dropna()
    assert historical_var(returns, 0.95) >= 0
    assert cvar(returns, 0.95) >= historical_var(returns, 0.95)


def test_construct_weights_returns_valid_series(synthetic_prices):
    returns = synthetic_prices.pct_change().dropna()
    cov = returns.iloc[-60:].cov()
    alpha_row = combined_alpha(synthetic_prices).iloc[-1]
    cfg = PortfolioConfig(method="risk_parity_alpha")
    w = construct_weights(cov, alpha_row, current_drawdown=-0.05, cfg=cfg)
    assert list(w.index) == list(cov.columns)
    assert w.notna().all()


def test_walk_forward_backtest_runs_without_lookahead_nans(synthetic_prices):
    """Test de bout en bout : vérifie que le backtest tourne avec les 3
    méthodes de sizing et ne produit ni NaN ni erreur (couvre la régression
    liée à l'interface alpha_predict_func)."""
    alpha_predict_func = make_walk_forward_predictor(combined_alpha)
    for method in ["risk_parity", "kelly", "risk_parity_alpha"]:
        cfg = PortfolioConfig(method=method)
        result = run_walk_forward_backtest(
            synthetic_prices, alpha_predict_func, cfg,
            initial_train_window=200, cov_lookback=60, rebalance_every=20,
        )
        assert not result["weights_history"].isna().any().any()
        assert result["equity_curve"].iloc[-1] > 0
        assert np.isfinite(result["metrics"]["max_drawdown"])