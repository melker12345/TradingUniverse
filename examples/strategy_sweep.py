from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from trading_universe.metrics import compute_all_metrics
from trading_universe.strategies import BuyAndHold, BullFlag, DonchianBreakout, SmaCross

DATA_DIR = Path("data/sweden_top10_daily")
TICKERS = [
    "INVE-B",
    "ATCO-A",
    "ATCO-B",
    "VOLV-B",
    "ERIC-B",
    "NDA-SE",
    "SEB-A",
    "SAND",
    "ESSITY-B",
    "HEXA-B",
]

# Transaction cost per trade (entry/exit) in bps
TX_BPS = 0.0


def load_closes() -> Dict[str, pd.Series]:
    closes: Dict[str, pd.Series] = {}
    for ticker in TICKERS:
        path = DATA_DIR / f"{ticker}.csv"
        df = pd.read_csv(path, index_col=0, parse_dates=[0])
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        close.name = ticker
        closes[ticker] = close
    return closes


def apply_costs(returns: pd.Series, positions: pd.Series, bps: float) -> pd.Series:
    if bps <= 0:
        return returns
    changes = positions.diff().abs().fillna(0.0)
    cost = (bps / 10000.0) * changes
    cost = cost.reindex(returns.index, fill_value=0.0)
    return returns - cost


def run_strategy_across_tickers(strategy, closes: Dict[str, pd.Series]) -> pd.Series:
    rets: List[pd.Series] = []
    for _, prices in closes.items():
        strat_ret = strategy.run(prices)
        # recover positions to apply costs
        positions = strategy.generate_positions(prices).shift(1).fillna(0.0)
        strat_ret = apply_costs(strat_ret, positions, bps=TX_BPS)
        rets.append(strat_ret)
    avg_returns = pd.concat(rets, axis=1).mean(axis=1)
    return avg_returns


def donchian_returns(closes: Dict[str, pd.Series], up: int, down: int) -> pd.Series:
    rets: List[pd.Series] = []
    for _, prices in closes.items():
        high = prices.rolling(up).max()
        low = prices.rolling(down).min()
        pos = ((prices > high.shift(1)) | (pos := pd.Series(0.0, index=prices.index)))  # type: ignore
        pos = (prices > high.shift(1)).astype(float)
        exit_mask = prices < low.shift(1)
        pos = pos.where(~exit_mask, other=0.0)
        pos = pos.ffill().fillna(0.0)
        returns = prices.pct_change().fillna(0.0)
        strat_ret = pos.shift(1).fillna(0.0) * returns
        strat_ret = apply_costs(strat_ret, pos.shift(1).fillna(0.0), bps=TX_BPS)
        rets.append(strat_ret)
    return pd.concat(rets, axis=1).mean(axis=1)


def momentum_returns(closes: Dict[str, pd.Series], lookback: int, thresh: float = 0.0) -> pd.Series:
    rets: List[pd.Series] = []
    for _, prices in closes.items():
        mom = prices.pct_change(lookback).shift(1)
        pos = (mom > thresh).astype(float)
        returns = prices.pct_change().fillna(0.0)
        strat_ret = pos.shift(1).fillna(0.0) * returns
        strat_ret = apply_costs(strat_ret, pos.shift(1).fillna(0.0), bps=TX_BPS)
        rets.append(strat_ret)
    return pd.concat(rets, axis=1).mean(axis=1)


def vol_filter_returns(closes: Dict[str, pd.Series], vol_win: int, vol_pct: float) -> pd.Series:
    rets: List[pd.Series] = []
    for _, prices in closes.items():
        daily = prices.pct_change()
        vol = daily.rolling(vol_win).std(ddof=0)
        thresh = vol.quantile(vol_pct)
        pos = (vol < thresh).astype(float)
        strat_ret = pos.shift(1).fillna(0.0) * daily.fillna(0.0)
        strat_ret = apply_costs(strat_ret, pos.shift(1).fillna(0.0), bps=TX_BPS)
        rets.append(strat_ret)
    return pd.concat(rets, axis=1).mean(axis=1)


def dual_momentum_returns(closes: Dict[str, pd.Series], short_lb: int, long_lb: int) -> pd.Series:
    rets: List[pd.Series] = []
    for _, prices in closes.items():
        short_ret = prices.pct_change(short_lb).shift(1)
        long_ret = prices.pct_change(long_lb).shift(1)
        pos = (short_ret > long_ret).astype(float)
        returns = prices.pct_change().fillna(0.0)
        strat_ret = pos.shift(1).fillna(0.0) * returns
        strat_ret = apply_costs(strat_ret, pos.shift(1).fillna(0.0), bps=TX_BPS)
        rets.append(strat_ret)
    return pd.concat(rets, axis=1).mean(axis=1)


def mean_revert_band_returns(closes: Dict[str, pd.Series], ma_win: int = 20, band: float = 0.02) -> pd.Series:
    rets: List[pd.Series] = []
    for _, prices in closes.items():
        ma = prices.rolling(ma_win).mean()
        pos = (prices < ma * (1 - band)).astype(float)
        exits = prices >= ma
        pos = pos.where(~exits, other=0.0).ffill().fillna(0.0)
        returns = prices.pct_change().fillna(0.0)
        strat_ret = pos.shift(1).fillna(0.0) * returns
        strat_ret = apply_costs(strat_ret, pos.shift(1).fillna(0.0), bps=TX_BPS)
        rets.append(strat_ret)
    return pd.concat(rets, axis=1).mean(axis=1)


def day_of_week_returns(closes: Dict[str, pd.Series], allowed_days: List[int]) -> pd.Series:
    rets: List[pd.Series] = []
    for _, prices in closes.items():
        idx = pd.to_datetime(prices.index)
        pos = pd.Series(0.0, index=idx)
        pos.loc[idx.dayofweek.isin(allowed_days)] = 1.0
        returns = prices.pct_change().fillna(0.0)
        strat_ret = pos.shift(1).fillna(0.0) * returns
        strat_ret = apply_costs(strat_ret, pos.shift(1).fillna(0.0), bps=TX_BPS)
        rets.append(strat_ret)
    return pd.concat(rets, axis=1).mean(axis=1)


def trend_vol_combo_returns(closes: Dict[str, pd.Series], ret_lb: int, vol_win: int, vol_pct: float) -> pd.Series:
    rets: List[pd.Series] = []
    for _, prices in closes.items():
        ret = prices.pct_change(ret_lb).shift(1)
        vol = prices.pct_change().rolling(vol_win).std(ddof=0)
        thresh = vol.quantile(vol_pct)
        pos = ((ret > 0) & (vol < thresh)).astype(float)
        returns = prices.pct_change().fillna(0.0)
        strat_ret = pos.shift(1).fillna(0.0) * returns
        strat_ret = apply_costs(strat_ret, pos.shift(1).fillna(0.0), bps=TX_BPS)
        rets.append(strat_ret)
    return pd.concat(rets, axis=1).mean(axis=1)


def breadth_returns(closes: Dict[str, pd.Series], lb: int) -> pd.Series:
    # average 50d return across tickers; long all when breadth > 0
    rets_all = []
    positions = []
    for _, prices in closes.items():
        rets_all.append(prices.pct_change(lb).shift(1))
    breadth = pd.concat(rets_all, axis=1).mean(axis=1)
    pos_flag = (breadth > 0).astype(float)
    avg_daily = pd.concat([p.pct_change() for p in closes.values()], axis=1).mean(axis=1).fillna(0.0)
    strat_ret = pos_flag.shift(1).fillna(0.0) * avg_daily
    strat_ret = apply_costs(strat_ret, pos_flag.shift(1).fillna(0.0), bps=TX_BPS)
    return strat_ret


def bull_flag_returns(closes: Dict[str, pd.Series]) -> pd.Series:
    rets: List[pd.Series] = []
    strat = BullFlag(
        breakout_window=5,
        trend_fast=20,
        trend_slow=50,
        high_lookback=20,
        max_pullback=0.05,
        exit_window=10,
    )
    for _, prices in closes.items():
        strat_ret = strat.run(prices)
        rets.append(strat_ret)
    return pd.concat(rets, axis=1).mean(axis=1)


def main() -> None:
    closes = load_closes()
    if not closes:
        raise SystemExit("No data")

    bh = BuyAndHold()
    bh_rets = run_strategy_across_tickers(bh, closes)
    bh_metrics = compute_all_metrics(bh_rets)
    bh_final_eq = float((1 + bh_rets).cumprod().iloc[-1])

    print("Baseline Buy & Hold (equal-weight, with costs):")
    print(bh_metrics)
    print(f"Final equity: {bh_final_eq:.2f}\n")

    candidates: List[Tuple[str, float, Dict[str, float]]] = []

    # 1) Simple momentum 60d > 0
    rets = momentum_returns(closes, lookback=60, thresh=0.0)
    eq = float((1 + rets).cumprod().iloc[-1])
    metrics = compute_all_metrics(rets, benchmark_returns=bh_rets)
    candidates.append(("mom_60_gt0", eq, metrics))

    # 2) Dual momentum 60d > 200d
    rets = dual_momentum_returns(closes, short_lb=60, long_lb=200)
    eq = float((1 + rets).cumprod().iloc[-1])
    metrics = compute_all_metrics(rets, benchmark_returns=bh_rets)
    candidates.append(("dual_mom_60_200", eq, metrics))

    # 3) SMA cross 20/100
    strat = SmaCross(short_window=20, long_window=100)
    rets = run_strategy_across_tickers(strat, closes)
    eq = float((1 + rets).cumprod().iloc[-1])
    metrics = compute_all_metrics(rets, benchmark_returns=bh_rets)
    candidates.append(("sma_20_100", eq, metrics))

    # 4) Donchian 55/20
    rets = donchian_returns(closes, up=55, down=20)
    eq = float((1 + rets).cumprod().iloc[-1])
    metrics = compute_all_metrics(rets, benchmark_returns=bh_rets)
    candidates.append(("donchian_55_20", eq, metrics))

    # 5) Donchian 20/10
    rets = donchian_returns(closes, up=20, down=10)
    eq = float((1 + rets).cumprod().iloc[-1])
    metrics = compute_all_metrics(rets, benchmark_returns=bh_rets)
    candidates.append(("donchian_20_10", eq, metrics))

    # 6) Vol filter 20d vol < 60th pct
    rets = vol_filter_returns(closes, vol_win=20, vol_pct=0.6)
    eq = float((1 + rets).cumprod().iloc[-1])
    metrics = compute_all_metrics(rets, benchmark_returns=bh_rets)
    candidates.append(("volf_20_p60", eq, metrics))

    # 7) Trend+vol: 50d return >0 AND 20d vol < 70th pct
    rets = trend_vol_combo_returns(closes, ret_lb=50, vol_win=20, vol_pct=0.7)
    eq = float((1 + rets).cumprod().iloc[-1])
    metrics = compute_all_metrics(rets, benchmark_returns=bh_rets)
    candidates.append(("trendvol_50_20_p70", eq, metrics))

    # 8) Breadth: avg 50d return across tickers >0
    rets = breadth_returns(closes, lb=50)
    eq = float((1 + rets).cumprod().iloc[-1])
    metrics = compute_all_metrics(rets, benchmark_returns=bh_rets)
    candidates.append(("breadth_50", eq, metrics))

    # 9) Mean reversion band: price < 20d SMA -2%, exit at SMA
    rets = mean_revert_band_returns(closes, ma_win=20, band=0.02)
    eq = float((1 + rets).cumprod().iloc[-1])
    metrics = compute_all_metrics(rets, benchmark_returns=bh_rets)
    candidates.append(("mr_band_20_2pct", eq, metrics))

    # 9) Seasonality: long Tue-Thu
    rets = day_of_week_returns(closes, allowed_days=[1, 2, 3])  # Tue, Wed, Thu
    eq = float((1 + rets).cumprod().iloc[-1])
    metrics = compute_all_metrics(rets, benchmark_returns=bh_rets)
    candidates.append(("dow_tue_thu", eq, metrics))

    # 10) Bull flag breakout
    rets = bull_flag_returns(closes)
    eq = float((1 + rets).cumprod().iloc[-1])
    metrics = compute_all_metrics(rets, benchmark_returns=bh_rets)
    candidates.append(("bull_flag", eq, metrics))

    candidates.sort(key=lambda x: x[1], reverse=True)

    print("Top 10 strategies by final equity (with 5 bps costs):")
    for name, eq, metrics in candidates[:10]:
        print(f"{name}: final_eq={eq:.2f} ann_return={metrics['annualized_return']:.2%} sharpe={metrics['sharpe']:.2f}")

    best = candidates[0]
    print("\nBest strategy detail:")
    print(best[0], best[2])
    print(f"Final equity vs BH: {best[1]:.2f} vs BH {bh_final_eq:.2f}")


if __name__ == "__main__":
    main()

