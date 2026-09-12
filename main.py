import matplotlib.pyplot as plt

from data import download_prices, compute_returns
from signals import combined_alpha, make_walk_forward_predictor
from portfolio import PortfolioConfig
from backtest import run_walk_forward_backtest, print_report


def main():
    tickers = ["SPY", "QQQ", "TLT", "GLD", "EEM"]
    prices = download_prices(tickers, start="2015-01-01")

    alpha_predict_func = make_walk_forward_predictor(combined_alpha)

    cfg = PortfolioConfig(
        method="risk_parity_alpha",   
        kelly_fraction=0.5,
        var_confidence=0.95,
        var_budget=0.015,             
        dd_limit=0.15,              
        dd_min_exposure=0.25,
        gross_leverage_cap=1.5,
    )

    result = run_walk_forward_backtest(
        prices, alpha_predict_func, cfg,
        initial_train_window=252,
        cov_lookback=60,
        rebalance_every=5,
        cost_bps=2.0,
    )

    print_report(result)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    result["equity_curve"].plot(ax=axes[0], title="Courbe d'équité")
    result["drawdown_curve"].plot(ax=axes[1], title="Drawdown", color="crimson")
    result["weights_history"].plot(ax=axes[2], title="Poids par actif", linewidth=1)

    plt.tight_layout()
    plt.savefig("backtest_report.png", dpi=130)
    print("\nGraphiques sauvegardés dans backtest_report.png")


if __name__ == "__main__":
    main()