from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from trading_universe.metrics import compute_all_metrics


TICKERS = {
    "gold": "GLD",     # SPDR Gold Shares ETF
    "silver": "SLV",   # iShares Silver Trust ETF
    "omx": "^OMX",     # OMX Stockholm 30 Index
}

PERIOD = "10y"
INTERVAL = "1d"


def download_prices() -> pd.DataFrame:
    data = {}
    for name, ticker in TICKERS.items():
        df = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=True)
        if df.empty:
            raise RuntimeError(f"No data for {ticker}")
        s = df["Close"].copy()
        s.name = name
        data[name] = s
    prices = pd.concat(data.values(), axis=1).dropna()
    prices.columns = list(data.keys())
    return prices


def main() -> None:
    prices = download_prices()
    rets = prices.pct_change().dropna()

    gold = rets["gold"]
    silver = rets["silver"]
    omx = rets["omx"]

    print("Rolling 90-day correlations (last value):")
    print(f"corr(gold, omx): {gold.rolling(90).corr(omx).iloc[-1]:.3f}")
    print(f"corr(silver, omx): {silver.rolling(90).corr(omx).iloc[-1]:.3f}")
    print(f"corr(gold, silver): {gold.rolling(90).corr(silver).iloc[-1]:.3f}")

    # Static correlations
    print("\nFull-sample correlations:")
    print(rets.corr())

    # Betas of OMX to gold/silver via linear regression
    def beta(target: pd.Series, factor: pd.Series) -> float:
        aligned = pd.concat([target, factor], axis=1).dropna()
        y = aligned.iloc[:, 0].values
        x = aligned.iloc[:, 1].values
        x = pd.DataFrame({"const": 1.0, "factor": x})
        coeffs = np.linalg.lstsq(x.values, y, rcond=None)[0]
        return float(coeffs[1])

    beta_omx_gold = beta(omx, gold)
    beta_omx_silver = beta(omx, silver)
    print(f"\nBeta(OMX to gold): {beta_omx_gold:.3f}")
    print(f"Beta(OMX to silver): {beta_omx_silver:.3f}")

    # Simple long-gold short-omx hedge PnL (equal notional)
    hedge_rets = gold - omx
    hedge_metrics = compute_all_metrics(hedge_rets, benchmark_returns=omx)
    print("\nLong gold / short omx (equal notional) metrics:")
    print(hedge_metrics)

    # Save to artifacts
    outdir = Path("artifacts")
    outdir.mkdir(exist_ok=True)
    prices.to_csv(outdir / "gold_silver_omx_prices.csv")
    print(f"\nSaved prices to {outdir/'gold_silver_omx_prices.csv'}")


if __name__ == "__main__":
    main()

