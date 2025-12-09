from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
import yfinance as yf

DATA_DIR = Path("data/sweden_top10_daily")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Top Swedish large caps excluding AstraZeneca (ticker AZN.ST)
TICKERS: Dict[str, str] = {
    "INVE-B": "INVE-B.ST",   # Investor B
    "ATCO-A": "ATCO-A.ST",   # Atlas Copco A
    "ATCO-B": "ATCO-B.ST",   # Atlas Copco B
    "VOLV-B": "VOLV-B.ST",   # Volvo B
    "ERIC-B": "ERIC-B.ST",   # Ericsson B
    "NDA-SE": "NDA-SE.ST",   # Nordea
    "SEB-A": "SEB-A.ST",     # SEB A
    "SAND": "SAND.ST",       # Sandvik
    "ESSITY-B": "ESSITY-B.ST",  # Essity B
    "HEXA-B": "HEXA-B.ST",   # Hexagon B
}


def download_prices(tickers: Dict[str, str], period: str = "10y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV for the given tickers.
    Default: daily bars for 10y (works with Yahoo). For intraday, Yahoo only
    allows ~730 days.
    """
    data: Dict[str, pd.DataFrame] = {}
    for label, symbol in tickers.items():
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            print(f"[warn] no data for {symbol} at {interval}")
            continue
        df.index.name = "datetime"
        out_path = DATA_DIR / f"{label}.csv"
        df.to_csv(out_path)
        data[label] = df
        print(f"[ok] {label} -> {out_path} ({len(df)} rows)")
    return data


def build_equal_weight_benchmark(data: Dict[str, pd.DataFrame]) -> pd.Series:
    closes = []
    for label, df in data.items():
        if "Close" in df:
            s = df["Close"].copy()
            s.name = label  # set series name explicitly
            closes.append(s)
    if not closes:
        raise RuntimeError("No close data to build benchmark")
    closes_df = pd.concat(closes, axis=1).dropna(how="all")
    benchmark = closes_df.mean(axis=1)
    benchmark.name = "equal_weight_close"
    out_path = DATA_DIR / "benchmark_equal_weight.csv"
    benchmark.to_csv(out_path, header=True)
    print(f"[ok] benchmark -> {out_path} ({len(benchmark)} rows)")
    return benchmark


def main() -> None:
    # Daily bars for 10y; adjust if you want a different span/granularity.
    data = download_prices(TICKERS, period="10y", interval="1d")
    if not data:
        print("No data downloaded; aborting.")
        return
    benchmark = build_equal_weight_benchmark(data)
    meta = {"tickers": TICKERS, "rows_per_ticker": {k: len(v) for k, v in data.items()}}
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] saved meta.json with ticker info")


if __name__ == "__main__":
    main()

