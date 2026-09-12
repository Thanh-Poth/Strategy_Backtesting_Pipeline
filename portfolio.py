import numpy as np
import pandas as pd

from risk import portfolio_var
from sizing import risk_parity_weights, kelly_weights, drawdown_throttle

from pydantic import BaseModel, Field

class PortfolioConfig(BaseModel):
    method: str = Field(default="risk_parity_alpha", pattern="^(risk_parity|kelly|risk_parity_alpha)$")
    kelly_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    var_confidence: float = Field(default=0.95, ge=0.9, le=0.999)
    var_budget: float = Field(default=0.02, gt=0.0)
    dd_limit: float = Field(default=0.15, gt=0.0)
    dd_min_exposure: float = Field(default=0.2, ge=0.0)
    gross_leverage_cap: float = Field(default=1.5, ge=1.0)

def base_weights(cov: pd.DataFrame, alpha_row: pd.Series,
                  cfg: PortfolioConfig) -> np.ndarray:
    if cfg.method == "risk_parity":
        return risk_parity_weights(cov)

    if cfg.method == "kelly":
        return kelly_weights(alpha_row, cov, kelly_fraction=cfg.kelly_fraction,
                              gross_leverage_cap=cfg.gross_leverage_cap)

    if cfg.method == "risk_parity_alpha":
        rp = risk_parity_weights(cov)
        tilt = 1.0 + alpha_row.clip(-1, 1).values
        w = rp * tilt
        return w / np.abs(w).sum() if np.abs(w).sum() > 0 else rp

    raise ValueError(f"méthode inconnue : {cfg.method}")


def apply_var_constraint(weights: np.ndarray, cov: pd.DataFrame,
                          cfg: PortfolioConfig) -> np.ndarray:
    current_var = portfolio_var(weights, cov, confidence=cfg.var_confidence)
    if current_var <= cfg.var_budget or current_var == 0:
        return weights
    scale = cfg.var_budget / current_var
    return weights * scale


def apply_drawdown_throttle(weights: np.ndarray, current_drawdown: float,
                             cfg: PortfolioConfig) -> np.ndarray:
    factor = drawdown_throttle(current_drawdown, dd_limit=cfg.dd_limit,
                                min_exposure=cfg.dd_min_exposure)
    return weights * factor


def construct_weights(cov: pd.DataFrame, alpha_row: pd.Series,
                       current_drawdown: float,
                       cfg: PortfolioConfig) -> pd.Series:
    w = base_weights(cov, alpha_row, cfg)
    w = apply_var_constraint(w, cov, cfg)
    w = apply_drawdown_throttle(w, current_drawdown, cfg)
    return pd.Series(w, index=cov.columns)