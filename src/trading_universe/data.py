from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_price_csv(
    path: str | Path,
    date_col: str = "date",
    price_col: str = "close",
    parse_dates: bool = True,
) -> pd.Series:
    """
    Load a CSV with at least a date column and a price column.
    Returns a pandas Series indexed by datetime.
    """
    df = pd.read_csv(path)
    if parse_dates:
        df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    series = df[price_col].astype(float)
    series.name = price_col
    return series


def generate_random_walk(
    periods: int = 252 * 2,
    start_price: float = 100.0,
    drift: float = 0.05,
    volatility: float = 0.20,
    seed: Optional[int] = None,
) -> pd.Series:
    """
    Utility for demos/tests: geometric Brownian motion style random walk.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    shock = rng.normal(loc=(drift - 0.5 * volatility**2) * dt, scale=volatility * np.sqrt(dt), size=periods)
    log_path = np.cumsum(shock)
    prices = start_price * np.exp(log_path)
    index = pd.date_range(freq="B", periods=periods, start="2020-01-01")
    return pd.Series(prices, index=index, name="price")

