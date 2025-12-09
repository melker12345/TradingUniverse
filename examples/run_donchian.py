from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from trading_universe.metrics import compute_all_metrics
from trading_universe.strategies import BuyAndHold, DonchianBreakout

DATA_DIR = Path("data/sweden_top10_daily")
TICKERS = [
    "INVE-B",
    "ATCO-A",
    "ATCO-B",
    "VOLV-B",
    "ERIC-B",
    "NDA-SE",
    "SEB-A",
    "SAND",
    "ESSITY-B",
    "HEXA-B",
]


def load_closes() -> Dict[str, pd.Series]:
    closes: Dict[str, pd.Series] = {}
    for ticker in TICKERS:
        path = DATA_DIR / f"{ticker}.csv"
        df = pd.read_csv(path, index_col=0, parse_dates=[0])
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        close.name = ticker
        closes[ticker] = close
    return closes


def run_strategy_across_tickers(strategy, closes: Dict[str, pd.Series]) -> pd.Series:
    rets: List[pd.Series] = []
    for _, prices in closes.items():
        rets.append(strategy.run(prices))
    avg_returns = pd.concat(rets, axis=1).mean(axis=1)
    return avg_returns


def main() -> None:
    closes = load_closes()

    bh = BuyAndHold()
    donchian = DonchianBreakout(breakout_window=55, exit_window=20, vol_window=40, vol_percentile=0.8)

    bh_rets = run_strategy_across_tickers(bh, closes)
    donchian_rets = run_strategy_across_tickers(donchian, closes)

    metrics_bh = compute_all_metrics(bh_rets)
    metrics_donchian = compute_all_metrics(donchian_rets, benchmark_returns=bh_rets)

    print("=== Buy & Hold (equal-weight) ===")
    print(metrics_bh)
    print(f"Final equity: {(1 + bh_rets).cumprod().iloc[-1]:.2f}\n")

    print("=== Donchian breakout (55/20) with vol filter 40d @ 80th pct ===")
    print(metrics_donchian)
    print(f"Final equity: {(1 + donchian_rets).cumprod().iloc[-1]:.2f}")


if __name__ == "__main__":
    main()

