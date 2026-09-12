import pandas as pd
import yfinance as yf


def download_prices(tickers: list[str], start: str, end: str | None = None,
                     interval: str = "1d") -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, interval=interval,
                       auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    prices = prices.dropna(how="all").ffill().dropna()
    return prices


def compute_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    if log:
        import numpy as np
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


if __name__ == "__main__":
    tickers = ["SPY", "TLT", "GLD", "QQQ", "EEM"]
    prices = download_prices(tickers, start="2018-01-01")
    print(prices.tail())
    print(compute_returns(prices).tail())