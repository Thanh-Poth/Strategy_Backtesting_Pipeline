import numpy as np
import pandas as pd


def risk_parity_weights(cov: pd.DataFrame, tol: float = 1e-8,
                         max_iter: int = 1000) -> np.ndarray:
   
    n = cov.shape[0]
    sigma = cov.values
    w = np.ones(n) / n

    for _ in range(max_iter):
        port_var = w @ sigma @ w
        marginal_contrib = sigma @ w
        risk_contrib = w * marginal_contrib
        target = port_var / n

        safe_risk_contrib = np.clip(risk_contrib, 1e-8, None)
        w_new = w * (target / safe_risk_contrib) ** 0.5
        w_new = np.clip(w_new, 1e-6, None)
        w_new /= w_new.sum()

        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    return w


def kelly_weights(mu: pd.Series, cov: pd.DataFrame,
                   kelly_fraction: float = 0.5,
                   long_only: bool = True,
                   gross_leverage_cap: float = 1.5) -> np.ndarray:
    sigma_inv = np.linalg.pinv(cov.values)
    f_star = kelly_fraction * (sigma_inv @ mu.values)

    if long_only:
        f_star = np.clip(f_star, 0, None)

    gross = np.abs(f_star).sum()
    if gross > gross_leverage_cap and gross > 0:
        f_star = f_star * (gross_leverage_cap / gross)

    return f_star


def drawdown_throttle(current_drawdown: float, dd_limit: float = 0.15,
                       min_exposure: float = 0.2) -> float:
    dd = abs(current_drawdown)
    if dd <= 0:
        return 1.0
    if dd >= dd_limit:
        return min_exposure
    # interpolation linéaire entre 1.0 et min_exposure
    return 1.0 - (1.0 - min_exposure) * (dd / dd_limit)