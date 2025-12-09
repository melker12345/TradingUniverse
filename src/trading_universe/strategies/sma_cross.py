from __future__ import annotations

import pandas as pd

from .base import BaseStrategy


class SmaCross(BaseStrategy):
    name = "sma_cross"

    def __init__(self, short_window: int = 20, long_window: int = 50):
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")
        self.short_window = short_window
        self.long_window = long_window
        self.name = f"sma_{short_window}_{long_window}"

    def generate_positions(self, prices: pd.Series) -> pd.Series:
        short_ma = prices.rolling(window=self.short_window).mean()
        long_ma = prices.rolling(window=self.long_window).mean()
        signal = (short_ma > long_ma).astype(float)
        return signal.rename("position")

