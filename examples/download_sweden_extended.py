from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import yfinance as yf

DATA_DIR = Path("data/sweden_extended_daily")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Large/mid-cap Swedish names + OMX index
TICKERS: Dict[str, str] = {
    "OMX": "^OMX",            # OMX Stockholm 30 index
    "INVE-B": "INVE-B.ST",    # Investor B
    "ATCO-A": "ATCO-A.ST",    # Atlas Copco A
    "ATCO-B": "ATCO-B.ST",    # Atlas Copco B
    "VOLV-B": "VOLV-B.ST",    # Volvo B
    "ERIC-B": "ERIC-B.ST",    # Ericsson B
    "NDA-SE": "NDA-SE.ST",    # Nordea
    "SEB-A": "SEB-A.ST",      # SEB A
    "SAND": "SAND.ST",        # Sandvik
    "ESSITY-B": "ESSITY-B.ST",# Essity B
    "HEXA-B": "HEXA-B.ST",    # Hexagon B
    "SWED-A": "SWED-A.ST",    # Swedbank A
    "SHB-A": "SHB-A.ST",      # Handelsbanken A
    "ALFA": "ALFA.ST",        # Alfa Laval
    "ASSA-B": "ASSA-B.ST",    # ASSA ABLOY B
    "SKF-B": "SKF-B.ST",      # SKF B
    "BOL": "BOL.ST",          # Boliden
    "TELIA": "TELIA.ST",      # Telia
    "HUSQ-B": "HUSQ-B.ST",    # Husqvarna B
    "KIND-SDB": "KIND-SDB.ST",# Kindred Group SDB (mid-cap)
}


def download_prices(period: str = "10y", interval: str = "1d") -> None:
    for label, symbol in TICKERS.items():
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            print(f"[warn] no data for {symbol}")
            continue
        df.index.name = "Date"
        out = DATA_DIR / f"{label}.csv"
        df.to_csv(out)
        print(f"[ok] {label} -> {out} ({len(df)} rows)")


def main() -> None:
    download_prices()


if __name__ == "__main__":
    main()

