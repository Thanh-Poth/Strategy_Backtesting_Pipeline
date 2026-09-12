import numpy as np
import pandas as pd
from scipy.stats import norm


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    return -np.percentile(returns.dropna(), 100 * (1 - confidence))


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    mu, sigma = returns.mean(), returns.std()
    z = norm.ppf(1 - confidence)
    return -(mu + z * sigma)


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    var = historical_var(returns, confidence)
    tail = returns[returns <= -var]
    return -tail.mean() if len(tail) > 0 else var


def portfolio_var(weights: np.ndarray, cov: pd.DataFrame,
                   confidence: float = 0.95, horizon_days: int = 1) -> float:
    port_var_daily = weights @ cov.values @ weights.T
    port_vol_daily = np.sqrt(max(port_var_daily, 0))
    z = norm.ppf(1 - confidence)
    return -z * port_vol_daily * np.sqrt(horizon_days)


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1.0


def max_drawdown(equity_curve: pd.Series) -> float:
    return -drawdown_series(equity_curve).min()