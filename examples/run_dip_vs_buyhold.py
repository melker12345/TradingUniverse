from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from trading_universe.metrics import compute_all_metrics
from trading_universe.monte_carlo import monte_carlo_pnl
from trading_universe.strategies import BuyAndHold, DipBuy

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


def to_returns(series: pd.Series) -> pd.Series:
    return series.pct_change().fillna(0.0)


def aggregate_strategy_returns(
    closes: Dict[str, pd.Series], dip_pct: float
) -> Tuple[pd.Series, pd.Series, int]:
    """
    Run buy-hold and dip-buy for each ticker, average returns across tickers
    (equal capital), and return the aggregated series plus total trades (entries).
    """
    bh_returns_list: List[pd.Series] = []
    dip_returns_list: List[pd.Series] = []
    trade_count = 0

    dip_strategy = DipBuy(dip_pct=dip_pct)
    bh_strategy = BuyAndHold()

    for _, prices in closes.items():
        bh_ret = bh_strategy.run(prices)
        dip_ret = dip_strategy.run(prices)

        # Count entries: position goes from 0 to 1
        positions = dip_strategy.generate_positions(prices)
        trade_count += int((positions.diff() == 1).sum())

        bh_returns_list.append(bh_ret)
        dip_returns_list.append(dip_ret)

    bh_returns = pd.concat(bh_returns_list, axis=1).mean(axis=1)
    dip_returns = pd.concat(dip_returns_list, axis=1).mean(axis=1)
    return bh_returns, dip_returns, trade_count


def sweep_dip_pcts(
    closes: Dict[str, pd.Series], dip_pcts: Sequence[float]
) -> List[Tuple[float, float, Dict[str, float], int]]:
    """
    Evaluate a list of dip percentages; return list of (dip_pct, final_equity, metrics, trades)
    sorted by final equity descending. Trades counts total entries across tickers.
    """
    results = []
    for pct in dip_pcts:
        bh_returns, dip_returns, trades = aggregate_strategy_returns(closes, pct)
        metrics = compute_all_metrics(dip_returns, benchmark_returns=bh_returns)
        final_equity = float((1 + dip_returns).cumprod().iloc[-1])
        results.append((pct, final_equity, metrics, trades))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def main() -> None:
    closes = load_closes()
    if not closes:
        raise SystemExit("No data loaded")

    bh_returns, dip_returns, trades_8 = aggregate_strategy_returns(closes, dip_pct=0.08)

    metrics_bh = compute_all_metrics(bh_returns)
    metrics_dip = compute_all_metrics(dip_returns, benchmark_returns=bh_returns)

    print("=== Equal-weight Buy & Hold across 10 stocks ===")
    print(metrics_bh)
    print("\n=== Dip-buy strategy (8% from peak, exit at prior high) averaged across 10 stocks ===")
    print({**metrics_dip, "trades": trades_8})
    print(f"\nFinal equity BH: {(1 + bh_returns).cumprod().iloc[-1]:.2f}")
    print(f"Final equity DIP: {(1 + dip_returns).cumprod().iloc[-1]:.2f}")

    # Simple sweep to search for better dip percentages
    dip_grid = [x / 100 for x in range(4, 21, 2)]  # 4%, 6%, ..., 20%
    sweep = sweep_dip_pcts(closes, dip_grid)
    print("\n=== Sweep results (sorted by final equity) ===")
    for pct, final_eq, metrics, trades in sweep:
        print(
            f"dip={pct:.0%} final_equity={final_eq:.2f} "
            f"annualized_return={metrics['annualized_return']:.2%} trades={trades}"
        )

    # Take the best dip % by final equity and run Monte Carlo to visualize dispersion
    best_pct, best_final_eq, best_metrics, best_trades = sweep[0]
    _, best_dip_returns, _ = aggregate_strategy_returns(closes, dip_pct=best_pct)
    mc = monte_carlo_pnl(best_dip_returns, n_paths=1000, seed=123)
    print(f"\n=== Monte Carlo on best dip ({best_pct:.0%}) ===")
    print(mc.summary.to_string(index=False))


if __name__ == "__main__":
    main()

