from __future__ import annotations

import pandas as pd

from .base import BaseStrategy


class DipBuy(BaseStrategy):
    """
    Go long when price is down at least `dip_pct` from the prior peak,
    exit when price reclaims that prior peak.
    Positions: 0 or 1 (no leverage, no shorting).
    """

    name = "dip_buy"

    def __init__(self, dip_pct: float = 0.08):
        if dip_pct <= 0 or dip_pct >= 1:
            raise ValueError("dip_pct must be in (0, 1)")
        self.dip_pct = dip_pct
        self.name = f"dip_buy_{int(dip_pct*100)}pct"

    def generate_positions(self, prices: pd.Series) -> pd.Series:
        positions = pd.Series(0.0, index=prices.index)
        if prices.empty:
            return positions

        peak = prices.iloc[0]
        in_trade = False
        entry_peak = peak

        for ts, price in prices.items():
            if not in_trade:
                peak = max(peak, price)
                if price <= peak * (1 - self.dip_pct):
                    in_trade = True
                    entry_peak = peak
                    positions.loc[ts] = 1.0
            else:
                positions.loc[ts] = 1.0
                if price >= entry_peak:
                    in_trade = False
                    peak = price  # reset peak after recovery

        return positions

