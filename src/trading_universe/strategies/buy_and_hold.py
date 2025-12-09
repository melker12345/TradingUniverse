from __future__ import annotations

import pandas as pd

from .base import BaseStrategy


class BuyAndHold(BaseStrategy):
    name = "buy_and_hold"

    def generate_positions(self, prices: pd.Series) -> pd.Series:
        return pd.Series(1.0, index=prices.index, name="position")

