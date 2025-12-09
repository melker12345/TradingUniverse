from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .base import BaseStrategy
from ..features import build_features


class MlThresholdStrategy(BaseStrategy):
    """
    Long-only: predict next-period return; go long when prediction > threshold.
    Uses only lagged features; BaseStrategy.run handles position lag to avoid lookahead.
    """

    def __init__(
        self,
        model,
        threshold: float,
        horizons: Iterable[int] = (1, 5, 20),
        name: str | None = None,
    ):
        self.model = model
        self.threshold = threshold
        self.horizons = tuple(horizons)
        self.name = name or f"ml_thr_{threshold:.4f}"

    def generate_positions(self, prices: pd.Series) -> pd.Series:
        features = build_features(prices, horizons=self.horizons)
        # Align predictions to feature index
        preds = pd.Series(self.model.predict(features), index=features.index)
        positions = (preds > self.threshold).astype(float)
        # Reindex to prices index; forward-fill zeros where no features yet
        positions = positions.reindex(prices.index, fill_value=0.0)
        return positions

