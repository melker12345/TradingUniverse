from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .metrics import compute_all_metrics
from .plotting import plot_drawdowns, plot_equity_curves
from .strategies.base import BaseStrategy


@dataclass
class StrategyResult:
    name: str
    equity: pd.Series
    returns: pd.Series
    metrics: Dict[str, float]


class Backtester:
    def __init__(
        self,
        prices: pd.Series,
        benchmark_prices: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ):
        self.prices = prices.sort_index()
        self.returns = self.prices.pct_change().fillna(0.0)
        self.benchmark_prices = benchmark_prices.sort_index() if benchmark_prices is not None else prices
        self.benchmark_returns = self.benchmark_prices.pct_change().fillna(0.0)
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def run_strategy(self, strategy: BaseStrategy) -> StrategyResult:
        strat_returns = strategy.run(self.prices)
        equity = (1 + strat_returns).cumprod()
        metrics = compute_all_metrics(
            strat_returns,
            benchmark_returns=self.benchmark_returns,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year,
        )
        return StrategyResult(
            name=strategy.name,
            equity=equity,
            returns=strat_returns,
            metrics=metrics,
        )

    def compare_strategies(self, strategies: List[BaseStrategy]) -> List[StrategyResult]:
        results = [self.run_strategy(strategy) for strategy in strategies]
        results.sort(key=lambda r: r.equity.iloc[-1], reverse=True)
        return results

    def plot(self, results: List[StrategyResult], output_dir: Optional[str] = None) -> None:
        equity_curves = {result.name: result.equity for result in results}
        benchmark_equity = (1 + self.benchmark_returns).cumprod()
        equity_curves["buy_and_hold"] = benchmark_equity
        plot_equity_curves(equity_curves, output_dir=output_dir)
        plot_drawdowns(equity_curves, output_dir=output_dir)

