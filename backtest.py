import numpy as np
import pandas as pd
import logging

from portfolio import PortfolioConfig, construct_weights
from risk import drawdown_series, max_drawdown, historical_var, cvar

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_walk_forward_backtest(
    prices: pd.DataFrame,
    alpha_predict_func: callable,
    cfg: PortfolioConfig,
    initial_train_window: int = 252,
    cov_lookback: int = 60,
    rebalance_every: int = 20,
    cost_bps: float = 2.0
) -> dict:

    returns = prices.pct_change().dropna()
    dates = returns.index

    weights_history = pd.DataFrame(0.0, index=dates, columns=prices.columns)
    equity = pd.Series(1.0, index=dates)
    current_weights = pd.Series(0.0, index=prices.columns)

    running_equity = 1.0
    running_peak = 1.0

    for i, t in enumerate(dates):
        if i % rebalance_every == 0 and i >= initial_train_window:
            expanding_train_data = prices.iloc[:i]
            alpha_row = alpha_predict_func(expanding_train_data)

            rolling_cov_window = returns.iloc[i - cov_lookback:i]
            cov = rolling_cov_window.cov()

            current_dd = (running_equity / running_peak) - 1.0
            current_weights = construct_weights(cov, alpha_row, current_dd, cfg)

        weights_history.loc[t] = current_weights

        if i == 0:
            day_return = 0.0
        else:
            w_prev = weights_history.iloc[i - 1]
            day_return = float((w_prev * returns.loc[t]).sum())

            if i % rebalance_every == 0 and i >= initial_train_window:
                turnover = (weights_history.iloc[i] - w_prev).abs().sum()
            else:
                turnover = 0.0

            day_return -= turnover * (cost_bps / 10_000.0)

        running_equity *= (1 + day_return)
        running_peak = max(running_peak, running_equity)
        equity.loc[t] = running_equity

    port_returns = equity.pct_change().fillna(0.0)

    benchmark_returns = returns.mean(axis=1)
    active_returns = port_returns - benchmark_returns

    if active_returns.std() > 0:
        information_ratio = (active_returns.mean() / active_returns.std()) * np.sqrt(252)
    else:
        information_ratio = np.nan

    metrics = {
        "total_return": equity.iloc[-1] - 1.0,
        "annualized_return": (equity.iloc[-1]) ** (252 / len(equity)) - 1.0,
        "annualized_vol": port_returns.std() * np.sqrt(252),
        "sharpe": (port_returns.mean() / port_returns.std()) * np.sqrt(252) if port_returns.std() > 0 else np.nan,
        "information_ratio": information_ratio,
        "max_drawdown": max_drawdown(equity),
        "var_95_daily": historical_var(port_returns, 0.95),
        "cvar_95_daily": cvar(port_returns, 0.95),
    }

    return {
        "equity_curve": equity,
        "daily_returns": port_returns,
        "weights_history": weights_history,
        "drawdown_curve": drawdown_series(equity),
        "metrics": metrics,
    }


def print_report(result: dict) -> None:
    m = result["metrics"]
    logger.info("=== Rapport de backtest ===")
    logger.info(f"Rendement total        : {m['total_return']*100:.2f}%")
    logger.info(f"Rendement annualisé    : {m['annualized_return']*100:.2f}%")
    logger.info(f"Volatilité annualisée  : {m['annualized_vol']*100:.2f}%")
    logger.info(f"Sharpe                 : {m['sharpe']:.2f}")
    logger.info(f"Information Ratio      : {m['information_ratio']:.2f}")
    logger.info(f"Max drawdown           : {m['max_drawdown']*100:.2f}%")
    logger.info(f"VaR 95% (1j)           : {m['var_95_daily']*100:.2f}%")
    logger.info(f"CVaR 95% (1j)          : {m['cvar_95_daily']*100:.2f}%")