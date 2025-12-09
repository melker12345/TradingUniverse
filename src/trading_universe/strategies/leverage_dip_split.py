from __future__ import annotations

import pandas as pd


class LeverageDipSplit:
    """
    Split-sleeve levered dip overlay:
    - Core base_weight stays long 1x at all times.
    - Sleeve A: on dip >= dip_pct from prior peak, deploy weight_a at lev_a.
    - Sleeve B: if Sleeve A MTM drawdown reaches dd_trigger (e.g., -50%), deploy weight_b at lev_b.
    - Exit both sleeves when price recovers to the prior peak at which Sleeve A was opened.

    All weights are fractions of total capital; ensure base_weight + weight_a + weight_b <= 1.
    Positions are decided on close t and applied to return t->t+1 via shift(1) in run().
    """

    def __init__(
        self,
        dip_pct: float = 0.08,
        base_weight: float = 0.8,
        weight_a: float = 0.05,
        lev_a: float = 10.0,
        weight_b: float = 0.05,
        lev_b: float = 20.0,
        dd_trigger: float = 0.5,
        exit_recover_frac: float = 1.0,
    ):
        if not (0 < dip_pct < 1):
            raise ValueError("dip_pct must be in (0,1)")
        if not (0 < base_weight < 1):
            raise ValueError("base_weight must be in (0,1)")
        if weight_a < 0 or weight_b < 0:
            raise ValueError("weights must be non-negative")
        if base_weight + weight_a + weight_b > 1.0:
            raise ValueError("sum of weights exceeds 1")
        if not (0 < dd_trigger < 1):
            raise ValueError("dd_trigger must be in (0,1), e.g. 0.5 for -50%")
        if not (0 < exit_recover_frac <= 1.0):
            raise ValueError("exit_recover_frac must be in (0,1]")
        self.dip_pct = dip_pct
        self.base_weight = base_weight
        self.weight_a = weight_a
        self.lev_a = lev_a
        self.weight_b = weight_b
        self.lev_b = lev_b
        self.dd_trigger = dd_trigger
        self.exit_recover_frac = exit_recover_frac
        self.name = f"levsplit_{int(dip_pct*100)}pct_bw{int(base_weight*100)}_a{int(weight_a*100)}@{int(lev_a)}x_b{int(weight_b*100)}@{int(lev_b)}x"

    def generate_positions(self, prices: pd.Series) -> pd.DataFrame:
        prices = prices.sort_index()
        peak = prices.expanding().max()

        pos_a = pd.Series(0.0, index=prices.index)
        pos_b = pd.Series(0.0, index=prices.index)

        active_a = False
        active_b = False
        entry_peak = prices.iloc[0]
        entry_a = prices.iloc[0]

        for ts, price in prices.items():
            # open A on dip
            if (not active_a) and price <= peak.loc[ts] * (1 - self.dip_pct):
                active_a = True
                entry_peak = peak.loc[ts]
                entry_a = price

            # check B trigger if A active and B not
            if active_a and (not active_b):
                dd = (price / entry_a) - 1.0
                if dd <= -self.dd_trigger:
                    active_b = True

            # set positions
            if active_a:
                pos_a.loc[ts] = 1.0
            if active_b:
                pos_b.loc[ts] = 1.0

            # exit both on recovery to entry_peak * exit_recover_frac
            if active_a and price >= entry_peak * self.exit_recover_frac:
                active_a = False
                active_b = False

        return pd.DataFrame({"pos_a": pos_a, "pos_b": pos_b})

    def run(self, prices: pd.Series) -> pd.Series:
        r = prices.pct_change().fillna(0.0)
        positions = self.generate_positions(prices)
        pos_a = positions["pos_a"].shift(1).fillna(0.0)
        pos_b = positions["pos_b"].shift(1).fillna(0.0)

        exposure = (
            self.base_weight * 1.0
            + self.weight_a * self.lev_a * pos_a
            + self.weight_b * self.lev_b * pos_b
        )
        strat_ret = exposure * r
        strat_ret.name = self.name
        return strat_ret

