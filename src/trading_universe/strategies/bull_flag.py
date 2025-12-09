from __future__ import annotations

import pandas as pd

from .base import BaseStrategy


class BullFlag(BaseStrategy):
    """
    Simple bull-flag breakout heuristic:
    - Require price near recent highs (within max_pullback of 20-day high).
    - Require alignment with trend (price > 20SMA and 50SMA).
    - Enter when price breaks above prior breakout_window high.
    - Exit on break below exit_window low.
    """

    name = "bull_flag"

    def __init__(
        self,
        breakout_window: int = 5,
        trend_fast: int = 20,
        trend_slow: int = 50,
        high_lookback: int = 20,
        max_pullback: float = 0.05,
        exit_window: int = 10,
    ):
        self.breakout_window = breakout_window
        self.trend_fast = trend_fast
        self.trend_slow = trend_slow
        self.high_lookback = high_lookback
        self.max_pullback = max_pullback
        self.exit_window = exit_window
        self.name = f"bull_flag_b{breakout_window}_pb{int(max_pullback*100)}"

    def generate_positions(self, prices: pd.Series) -> pd.Series:
        prices = prices.sort_index()
        ma_fast = prices.rolling(self.trend_fast).mean()
        ma_slow = prices.rolling(self.trend_slow).mean()
        recent_high = prices.rolling(self.high_lookback).max()
        pullback = (recent_high - prices) / recent_high

        breakout_level = prices.rolling(self.breakout_window).max().shift(1)
        breakout = prices > breakout_level

        in_trend = (prices > ma_fast) & (prices > ma_slow)
        within_flag = pullback <= self.max_pullback

        pos = (breakout & in_trend & within_flag).astype(float)

        exit_level = prices.rolling(self.exit_window).min().shift(1)
        exits = prices < exit_level
        pos = pos.where(~exits, other=0.0)
        pos = pos.ffill().fillna(0.0)
        return pos.rename("position")

