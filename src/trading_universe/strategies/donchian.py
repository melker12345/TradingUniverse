from __future__ import annotations

import pandas as pd

from .base import BaseStrategy


class DonchianBreakout(BaseStrategy):
    """
    Long-only Donchian breakout:
    - Enter long when price breaks above prior 'breakout_window' high.
    - Exit to flat when price falls below prior 'exit_window' low.
    Optional volatility filter: stay flat when rolling vol exceeds a percentile threshold.
    """

    name = "donchian_breakout"

    def __init__(
        self,
        breakout_window: int = 55,
        exit_window: int = 20,
        vol_window: int | None = None,
        vol_percentile: float | None = None,
    ):
        if exit_window >= breakout_window:
            raise ValueError("exit_window should be smaller than breakout_window")
        self.breakout_window = breakout_window
        self.exit_window = exit_window
        self.vol_window = vol_window
        self.vol_percentile = vol_percentile
        self.name = f"donchian_{breakout_window}_{exit_window}"

    def generate_positions(self, prices: pd.Series) -> pd.Series:
        prices = prices.sort_index()
        upper = prices.rolling(self.breakout_window).max().shift(1)
        lower = prices.rolling(self.exit_window).min().shift(1)

        pos = pd.Series(0.0, index=prices.index)
        long_entries = prices > upper
        exits = prices < lower

        pos = long_entries.astype(float)
        pos = pos.where(~exits, other=0.0)
        pos = pos.ffill().fillna(0.0)

        if self.vol_window and self.vol_percentile:
            vol = prices.pct_change().rolling(self.vol_window).std(ddof=0)
            thresh = vol.quantile(self.vol_percentile)
            high_vol = vol > thresh
            pos = pos.where(~high_vol, other=0.0)
            pos = pos.ffill().fillna(0.0)

        return pos.rename("position")

