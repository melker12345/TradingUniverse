from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingRegressor

from trading_universe.features import build_features_and_labels
from trading_universe.metrics import compute_all_metrics
from trading_universe.transaction_costs import apply_transaction_costs

DATA_DIR = Path("data/sweden_top10_daily")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

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


def load_closes() -> Dict[str, pd.Series]:
    closes: Dict[str, pd.Series] = {}
    for ticker in TICKERS:
        path = DATA_DIR / f"{ticker}.csv"
        df = pd.read_csv(path, index_col=0, parse_dates=[0])
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        close.name = ticker
        closes[ticker] = close
    return closes


def prepare_dataset(
    closes: Dict[str, pd.Series], horizons: Iterable[int] = (1, 5, 20, 60)
) -> pd.DataFrame:
    rows = []
    for ticker, prices in closes.items():
        feats, labels = build_features_and_labels(prices, horizons=horizons)
        feats = feats.copy()
        feats.index = pd.to_datetime(feats.index)
        labels = labels.copy()
        labels.index = pd.to_datetime(labels.index)
        df = feats.copy()
        df["label"] = labels
        df["ticker"] = ticker
        df["date"] = df.index
        df = df.set_index(["date", "ticker"])
        rows.append(df)
    data = pd.concat(rows, axis=0).sort_index()
    return data


def split_time(data: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2):
    dates = pd.to_datetime(data.index.get_level_values(0).unique()).sort_values()
    train_end = dates[int(len(dates) * train_frac)]
    val_end = dates[int(len(dates) * (train_frac + val_frac))]
    train = data.loc[:train_end]
    val = data.loc[train_end:val_end]
    test = data.loc[val_end:]
    return train, val, test


def pick_threshold(preds: pd.Series, labels: pd.Series) -> Tuple[float, float]:
    """
    Search thresholds to maximize final equity on provided set.
    Returns (best_threshold, best_final_equity).
    """
    q_low = preds.quantile(0.5)
    q_high = preds.quantile(0.99)
    if not np.isfinite(q_low) or not np.isfinite(q_high) or q_low == q_high:
        q_low, q_high = 0.0, preds.max() if np.isfinite(preds.max()) else 0.01
    thr_candidates = np.linspace(q_low, q_high, 25)
    best_thr, best_eq = thr_candidates[0], -np.inf
    for thr in thr_candidates:
        pos = (preds > thr).astype(float)
        strat_ret = pos * labels
        eq = float((1 + strat_ret).prod())
        if eq > best_eq:
            best_eq = eq
            best_thr = thr
    return best_thr, best_eq


def agg_bh_returns(closes: Dict[str, pd.Series]) -> pd.Series:
    rets = []
    for _, prices in closes.items():
        rets.append(prices.pct_change())
    df = pd.concat(rets, axis=1).mean(axis=1)
    return df.dropna()


def evaluate_strategy(
    preds: pd.Series, labels: pd.Series, threshold: float, tx_bps: float
) -> pd.Series:
    pos = (preds > threshold).astype(float)
    strat_ret = pos * labels
    strat_ret.name = "ml_strategy"
    # apply transaction costs per ticker series before aggregation
    per_ticker = []
    for ticker in preds.index.get_level_values(1).unique():
        p = pos.xs(ticker, level=1)
        r = labels.xs(ticker, level=1)
        r_cost = apply_transaction_costs(r, p, bps=tx_bps)
        per_ticker.append(r_cost)
    strat_daily = pd.concat(per_ticker, axis=1).mean(axis=1)
    return strat_daily


def walk_forward_splits(
    data: pd.DataFrame, n_folds: int = 3, train_years: int = 4, val_years: int = 2, test_years: int = 2
):
    dates = pd.Index(pd.to_datetime(data.index.get_level_values(0).unique())).sort_values()
    folds = []
    window_years = train_years + val_years + test_years
    step_years = max(1, window_years // n_folds)
    start_date = dates[0]
    last_date = dates[-1]

    while len(folds) < n_folds and start_date < last_date:
        train_end = start_date + pd.DateOffset(years=train_years)
        val_end = train_end + pd.DateOffset(years=val_years)
        test_end = val_end + pd.DateOffset(years=test_years)
        if test_end > last_date:
            break

        train_mask = (data.index.get_level_values(0) >= start_date) & (data.index.get_level_values(0) < train_end)
        val_mask = (data.index.get_level_values(0) >= train_end) & (data.index.get_level_values(0) < val_end)
        test_mask = (data.index.get_level_values(0) >= val_end) & (data.index.get_level_values(0) < test_end)

        train = data.loc[train_mask]
        val = data.loc[val_mask]
        test = data.loc[test_mask]
        if train.empty or val.empty or test.empty:
            break

        folds.append((train, val, test))
        start_date = start_date + pd.DateOffset(years=step_years)

    return folds


def main() -> None:
    closes = load_closes()
    data = prepare_dataset(closes)

    folds = walk_forward_splits(data, n_folds=4, train_years=3, val_years=1, test_years=1)
    if not folds:
        raise SystemExit("Not enough data for requested folds.")

    feature_cols = [c for c in data.columns if c.startswith(("ret_", "vol_", "drawdown"))]

    tx_bps = 2.0  # per trade (entry/exit) in basis points
    test_equities = []
    best_model = None
    best_thr = None

    for i, (train, val, test) in enumerate(folds, 1):
        model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=400,
            min_samples_leaf=5,
            random_state=123 + i,
        )
        model.fit(train[feature_cols], train["label"])

        preds_val = pd.Series(model.predict(val[feature_cols]), index=val.index)
        thr, val_eq = pick_threshold(preds_val, val["label"])

        preds_test = pd.Series(model.predict(test[feature_cols]), index=test.index)
        strat_test = evaluate_strategy(preds_test, test["label"], thr, tx_bps=tx_bps)

        bh_rets_full = agg_bh_returns(closes)
        bh_rets_test = bh_rets_full.reindex(strat_test.index).dropna()
        strat_test = strat_test.loc[bh_rets_test.index]

        metrics = compute_all_metrics(strat_test, benchmark_returns=bh_rets_test)
        final_equity = float((1 + strat_test).prod())
        test_equities.append(final_equity)
        print(f"\nFold {i}: thr={thr:.6f} val_final_equity={val_eq:.2f} test_final_equity={final_equity:.2f}")
        print(metrics)

        # keep last fold model/threshold
        best_model = model
        best_thr = thr

    mean_test_eq = float(np.mean(test_equities)) if test_equities else 0.0
    print(f"\nMean test final equity across folds: {mean_test_eq:.2f}")

    if best_model is not None and best_thr is not None:
        dump({"model": best_model, "threshold": best_thr, "feature_cols": feature_cols}, ARTIFACTS_DIR / "ml_threshold.joblib")
        print(f"Saved last-fold model to {ARTIFACTS_DIR / 'ml_threshold.joblib'}")


if __name__ == "__main__":
    main()

