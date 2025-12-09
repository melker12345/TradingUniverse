from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    paths: pd.DataFrame
    summary: pd.DataFrame


def monte_carlo_pnl(
    returns: pd.Series,
    n_paths: int = 1000,
    horizon: Optional[int] = None,
    seed: Optional[int] = None,
) -> MonteCarloResult:
    """
    Bootstrap return stream to visualize payout dispersion.
    """
    if returns.empty:
        empty = pd.DataFrame()
        return MonteCarloResult(empty, empty)

    rng = np.random.default_rng(seed)
    base = returns.values
    horizon = horizon or len(base)
    draws = rng.choice(base, size=(n_paths, horizon), replace=True)
    compounded = np.cumprod(1 + draws, axis=1)

    paths = pd.DataFrame(
        compounded,
        index=[f"path_{i}" for i in range(n_paths)],
        columns=[f"step_{j}" for j in range(horizon)],
    )

    final = paths.iloc[:, -1]
    summary = pd.DataFrame(
        {
            "median_final": [float(np.median(final))],
            "p10_final": [float(np.percentile(final, 10))],
            "p90_final": [float(np.percentile(final, 90))],
            "mean_final": [float(final.mean())],
        }
    )

    return MonteCarloResult(paths=paths, summary=summary)

