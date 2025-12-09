from __future__ import annotations

from pathlib import Path

from trading_universe.backtester import Backtester
from trading_universe.data import generate_random_walk
from trading_universe.monte_carlo import monte_carlo_pnl
from trading_universe.strategies import BuyAndHold, SmaCross


def main() -> None:
    prices = generate_random_walk(seed=42)
    benchmark = prices  # simple buy-and-hold on the same asset

    strategies = [
        BuyAndHold(),
        SmaCross(short_window=10, long_window=50),
        SmaCross(short_window=20, long_window=100),
    ]

    bt = Backtester(prices=prices, benchmark_prices=benchmark, risk_free_rate=0.0)
    results = bt.compare_strategies(strategies)

    print("=== Strategy leaderboard (sorted by final equity) ===")
    for res in results:
        print(f"{res.name}: final_equity={res.equity.iloc[-1]:.2f} annualized={res.metrics['annualized_return']:.2%}")
        print(res.metrics)
        print("-" * 60)

    output_dir = Path("artifacts")
    bt.plot(results, output_dir=str(output_dir))
    print(f"Saved plots to: {output_dir.resolve()}")

    best = results[0]
    mc = monte_carlo_pnl(best.returns, n_paths=500, seed=123)
    print("Monte Carlo summary for best strategy:")
    print(mc.summary.to_string(index=False))


if __name__ == "__main__":
    main()

