from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from trading_universe.metrics import compute_all_metrics
from trading_universe.strategies import BuyAndHold, LeverageDip

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
    lev = LeverageDip(dip_pct=0.08, base_weight=0.9, leverage=10.0)

    bh_rets = run_strategy_across_tickers(bh, closes)
    lev_rets = run_strategy_across_tickers(lev, closes)

    metrics_bh = compute_all_metrics(bh_rets)
    metrics_lev = compute_all_metrics(lev_rets, benchmark_returns=bh_rets)

    print("=== Buy & Hold (90/10 benchmark reference is pure BH here) ===")
    print(metrics_bh)
    print(f"Final equity: {(1 + bh_rets).cumprod().iloc[-1]:.2f}\n")

    print("=== LeverageDip: 90% BH + 10x sleeve on 8% dip ===")
    print(metrics_lev)
    print(f"Final equity: {(1 + lev_rets).cumprod().iloc[-1]:.2f}")


if __name__ == "__main__":
    main()

