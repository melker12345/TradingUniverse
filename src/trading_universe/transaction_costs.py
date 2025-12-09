from __future__ import annotations

import pandas as pd


def apply_transaction_costs(returns: pd.Series, positions: pd.Series, bps: float) -> pd.Series:
    """
    Apply per-entry transaction cost in basis points.
    Cost is deducted when position increases (0->1) and when decreases (1->0),
    modeling a round-trip per change.
    """
    if bps <= 0:
        return returns
    changes = positions.diff().abs().fillna(0.0)
    # cost per unit notional changed
    cost = (bps / 10000.0) * changes
    # align cost to returns index
    cost = cost.reindex(returns.index, fill_value=0.0)
    return returns - cost

