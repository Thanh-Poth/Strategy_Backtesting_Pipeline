import numpy as np
import pandas as pd


def momentum_alpha(prices: pd.DataFrame, lookback: int = 126) -> pd.DataFrame:
    mom = prices.pct_change(lookback)
    z = mom.sub(mom.mean(axis=1), axis=0).div(mom.std(axis=1), axis=0)
    return np.tanh(z / 2)


def mean_reversion_alpha(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    ma = prices.rolling(window).mean()
    sd = prices.rolling(window).std()
    z = (prices - ma) / sd
    return np.tanh(-z / 2)


def combined_alpha(prices: pd.DataFrame,
                    w_momentum: float = 0.6,
                    w_meanrev: float = 0.4) -> pd.DataFrame:
    mom = momentum_alpha(prices)
    mr = mean_reversion_alpha(prices)
    combo = w_momentum * mom + w_meanrev * mr
    return combo.clip(-1, 1).dropna(how="all")


def make_walk_forward_predictor(alpha_fn=combined_alpha) -> callable:
    def _predict(expanding_prices: pd.DataFrame) -> pd.Series:
        alpha_df = alpha_fn(expanding_prices)
        return alpha_df.iloc[-1]
    return _predict