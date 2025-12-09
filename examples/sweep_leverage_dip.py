from __future__ import annotations

from itertools import product
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
    bh_rets = run_strategy_across_tickers(bh, closes)
    bh_metrics = compute_all_metrics(bh_rets)
    bh_final = float((1 + bh_rets).cumprod().iloc[-1])

    print("=== Buy & Hold baseline ===")
    print(bh_metrics)
    print(f"Final equity: {bh_final:.2f}\n")

    dip_grid = [0.04, 0.06, 0.08, 0.10]
    leverage_grid = [5.0, 10.0]
    base_weight_grid = [0.8, 0.9]

    results = []
    for dip_pct, lev, bw in product(dip_grid, leverage_grid, base_weight_grid):
        strat = LeverageDip(dip_pct=dip_pct, base_weight=bw, leverage=lev)
        rets = run_strategy_across_tickers(strat, closes)
        eq = float((1 + rets).cumprod().iloc[-1])
        metrics = compute_all_metrics(rets, benchmark_returns=bh_rets)
        results.append((dip_pct, lev, bw, eq, metrics))

    results.sort(key=lambda x: x[3], reverse=True)

    print("Top 5 leverage-dip configs by final equity:")
    for dip_pct, lev, bw, eq, metrics in results[:5]:
        print(
            f"dip={dip_pct:.0%} lev={lev:.1f} base={bw:.0%} "
            f"final_eq={eq:.2f} ann_ret={metrics['annualized_return']:.2%} "
            f"maxDD={metrics['max_drawdown']:.2%} sharpe={metrics['sharpe']:.2f}"
        )

    best = results[0]
    dip_pct, lev, bw, eq, metrics = best
    print("\nBest config detail:")
    print(
        f"dip={dip_pct:.0%} lev={lev:.1f} base={bw:.0%} "
        f"final_eq={eq:.2f} vs BH {bh_final:.2f}"
    )
    print(metrics)


if __name__ == "__main__":
    main()

