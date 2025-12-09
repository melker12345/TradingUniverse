from __future__ import annotations

import pandas as pd


class BaseStrategy:
    name: str = "base"

    def generate_positions(self, prices: pd.Series) -> pd.Series:
        raise NotImplementedError

    def run(self, prices: pd.Series) -> pd.Series:
        """
        Convert price series to strategy returns using positions from generate_positions.
        Positions are lagged by one period to avoid look-ahead bias.
        """
        positions = self.generate_positions(prices).fillna(0.0)
        returns = prices.pct_change().fillna(0.0)
        strategy_returns = positions.shift(1).fillna(0.0) * returns
        strategy_returns.name = self.name
        return strategy_returns

