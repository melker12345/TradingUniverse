from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from trading_universe.metrics import compute_all_metrics
from trading_universe.strategies import BuyAndHold, LeverageDip, LeverageDipSplit

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
    # Reference single-sleeve (from prior sweep best-ish but with 90% core)
    lev_single = LeverageDip(dip_pct=0.04, base_weight=0.9, leverage=10.0)

    # Split sleeve: 80% core, 5% at 10x on dip, if that sleeve is down 50%, add 5% at 20x
    lev_split = LeverageDipSplit(
        dip_pct=0.04,
        base_weight=0.8,
        weight_a=0.05,
        lev_a=10.0,
        weight_b=0.05,
        lev_b=20.0,
        dd_trigger=0.5,
    )

    bh_rets = run_strategy_across_tickers(bh, closes)
    single_rets = run_strategy_across_tickers(lev_single, closes)
    split_rets = run_strategy_across_tickers(lev_split, closes)

    print("=== Buy & Hold baseline ===")
    print(compute_all_metrics(bh_rets))
    print(f"Final equity: {(1 + bh_rets).cumprod().iloc[-1]:.2f}\n")

    print("=== Single-sleeve leverage dip (90% core, 4% trigger, 10x) ===")
    print(compute_all_metrics(single_rets, benchmark_returns=bh_rets))
    print(f"Final equity: {(1 + single_rets).cumprod().iloc[-1]:.2f}\n")

    print("=== Split-sleeve leverage dip (80% core, 5% @10x, DD 50% then 5% @20x) ===")
    print(compute_all_metrics(split_rets, benchmark_returns=bh_rets))
    print(f"Final equity: {(1 + split_rets).cumprod().iloc[-1]:.2f}\n")


if __name__ == "__main__":
    main()

