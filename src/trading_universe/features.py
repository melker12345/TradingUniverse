from __future__ import annotations

from typing import Dict, Iterable, Tuple

import pandas as pd


def build_features(prices: pd.Series, horizons: Iterable[int] = (1, 5, 20, 60)) -> pd.DataFrame:
    """
    Build lagged, rolling features from a price series using only past data.
    Features:
    - lagged returns for each horizon
    - rolling volatility (std) of 1-day returns over each horizon
    - drawdown from rolling peak
    """
    prices = prices.sort_index()
    rets = prices.pct_change()

    feats: Dict[str, pd.Series] = {}
    for h in horizons:
        feats[f"ret_{h}"] = rets.rolling(h).sum().shift(1)  # past h-day return
    for h in horizons:
        feats[f"vol_{h}"] = rets.rolling(h).std(ddof=0).shift(1)

    rolling_peak = prices.cummax()
    drawdown = (prices / rolling_peak - 1).shift(1)
    feats["drawdown"] = drawdown

    df = pd.DataFrame(feats)
    df = df.dropna()
    return df


def build_features_and_labels(
    prices: pd.Series, horizons: Iterable[int] = (1, 5, 20)
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Returns (features, label) where label is next-day return.
    Drop rows with NA to keep alignment.
    """
    features = build_features(prices, horizons=horizons)
    future_ret = prices.pct_change().shift(-1)
    labels = future_ret.loc[features.index]
    labels.name = "future_return"
    features = features.loc[labels.index]
    return features, labels

