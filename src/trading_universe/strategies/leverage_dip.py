from __future__ import annotations

import pandas as pd


class LeverageDip:
    """
    Hybrid: hold base_weight long at all times; when price is at least dip_pct below
    its prior peak, allocate the remaining weight into a leveraged long (leverage factor).
    Exit leveraged sleeve when price recovers to prior peak.
    """

    def __init__(self, dip_pct: float = 0.08, base_weight: float = 0.9, leverage: float = 10.0):
        if not (0 < base_weight < 1):
            raise ValueError("base_weight must be in (0,1)")
        if dip_pct <= 0 or dip_pct >= 1:
            raise ValueError("dip_pct must be in (0,1)")
        self.dip_pct = dip_pct
        self.base_weight = base_weight
        self.leverage = leverage
        self.name = f"leverage_dip_{int(dip_pct*100)}pct_{int(leverage)}x"

    def generate_positions(self, prices: pd.Series) -> pd.Series:
        """Binary sleeve position for the leveraged sleeve."""
        prices = prices.sort_index()
        peak = prices.expanding().max()
        sleeve_on = prices <= peak * (1 - self.dip_pct)
        # turn off sleeve once price recovers to peak
        recover = prices >= peak
        pos = pd.Series(0.0, index=prices.index)
        active = False
        entry_peak = peak.iloc[0]
        for ts, price in prices.items():
            if not active and price <= peak.loc[ts] * (1 - self.dip_pct):
                active = True
                entry_peak = peak.loc[ts]
            if active:
                pos.loc[ts] = 1.0
                if price >= entry_peak:
                    active = False
        return pos

    def run(self, prices: pd.Series) -> pd.Series:
        r = prices.pct_change().fillna(0.0)
        sleeve_pos = self.generate_positions(prices).shift(1).fillna(0.0)
        # base always long
        blended = self.base_weight * r + (1 - self.base_weight) * sleeve_pos * self.leverage * r
        blended.name = self.name
        return blended

