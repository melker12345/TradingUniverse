from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from trading_universe.strategies import BuyAndHold, LeverageDipSplit

DATA_DIR = Path("data/sweden_top10_daily")

# Best config from sweep: dip=4%, base=90%, A:5%@10x, DD 40% then B:5%@20x, exit at 95% recovery
BEST_CONFIG = {
    "dip_pct": 0.04,
    "base_weight": 0.9,
    "weight_a": 0.05,
    "lev_a": 10.0,
    "weight_b": 0.05,
    "lev_b": 20.0,
    "dd_trigger": 0.4,
    "exit_recover_frac": 0.95,
}


def load_closes() -> Dict[str, pd.Series]:
    closes: Dict[str, pd.Series] = {}
    for path in DATA_DIR.glob("*.csv"):
        label = path.stem
        df = pd.read_csv(path, index_col=0, parse_dates=[0])
        # tolerate different column casing; if missing, skip file
        close_cols = [c for c in df.columns if c.lower() == "close"]
        if not close_cols:
            continue
        close = pd.to_numeric(df[close_cols[0]], errors="coerce").dropna()
        close.name = label
        closes[label] = close
    return closes


def run_strategy_across_tickers(strategy, closes: Dict[str, pd.Series]) -> pd.Series:
    rets: List[pd.Series] = []
    for _, prices in closes.items():
        rets.append(strategy.run(prices))
    avg_returns = pd.concat(rets, axis=1).mean(axis=1)
    return avg_returns


def main() -> None:
    closes = load_closes()
    if not closes:
        raise SystemExit("No data found in sweden_top10_daily")

    bh = BuyAndHold()
    best = LeverageDipSplit(**BEST_CONFIG)

    bh_rets = run_strategy_across_tickers(bh, closes)
    best_rets = run_strategy_across_tickers(best, closes)

    bh_equity = (1 + bh_rets).cumprod()
    best_equity = (1 + best_rets).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(bh_equity.index, bh_equity.values, label="Buy & Hold")
    plt.plot(best_equity.index, best_equity.values, label="Leverage Split Best")
    plt.title("Equity Curves: Buy & Hold vs Leverage Split (best config)")
    plt.ylabel("Growth of 1")
    plt.legend()
    outdir = Path("artifacts")
    outdir.mkdir(exist_ok=True)
    outfile = outdir / "equity_leverage_split.png"
    plt.savefig(outfile, bbox_inches="tight")
    print(f"Saved plot to {outfile}")


if __name__ == "__main__":
    main()

