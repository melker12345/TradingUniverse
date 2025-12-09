from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# GBM Monte Carlo strategy tester
# - Intended to run offline on a faster machine (can be slow with many paths).
# - Simulates geometric Brownian motion across a grid of mu (drift) and sigma (vol).
# - Dynamically loads a strategy class/module and applies it to each price path.
# - Outputs percentile stats for final equity and drawdown to a CSV.
#
# Defaults are set to our best split-sleeve config; you can run with NO flags:
#   python -m examples.sim_gbm_strategy
#
# To override, pass only what you need, e.g. higher path count:
#   python -m examples.sim_gbm_strategy --paths 10000
#
# Or swap strategy:
#   python -m examples.sim_gbm_strategy \
#     --strategy-module trading_universe.strategies.leverage_dip \
#     --strategy-class LeverageDip \
#     --strategy-kwargs '{"dip_pct":0.04,"base_weight":0.9,"leverage":10}'
# -----------------------------------------------------------------------------


def load_strategy(module_path: str, class_name: str, kwargs: Dict[str, Any]):
    """Dynamically import and instantiate the strategy."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**kwargs)


def simulate_gbm(mu: float, sigma: float, steps: int, paths: int, dt: float, s0: float, seed: int) -> np.ndarray:
    """Generate GBM price paths with drift mu and vol sigma."""
    rng = np.random.default_rng(seed)
    drift = (mu - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)
    shocks = rng.normal(drift, vol, size=(paths, steps))
    log_paths = np.cumsum(shocks, axis=1)
    prices = s0 * np.exp(log_paths)
    return prices


def max_drawdown_from_equity(equity: pd.Series) -> float:
    peaks = equity.cummax()
    dd = (equity - peaks) / peaks
    return float(dd.min())


def evaluate_strategy_on_paths(
    strategy,
    price_paths: np.ndarray,
    dt: float,
    periods_per_year: int = 252,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the strategy to each path; return arrays of final equity and max drawdown."""
    final_equities = []
    max_drawdowns = []
    for path in price_paths:
        prices = pd.Series(path)
        rets = strategy.run(prices)
        equity = (1 + rets).cumprod()
        final_equities.append(equity.iloc[-1])
        max_drawdowns.append(max_drawdown_from_equity(equity))
    return np.array(final_equities), np.array(max_drawdowns)


def summarize(vals: np.ndarray) -> Dict[str, float]:
    return {
        "p10": float(np.percentile(vals, 10)),
        "p50": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
        "mean": float(np.mean(vals)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GBM Monte Carlo strategy tester (run offline on a fast machine).")
    parser.add_argument(
        "--strategy-module",
        default="trading_universe.strategies.leverage_dip_split",
        help="Strategy module to import.",
    )
    parser.add_argument(
        "--strategy-class",
        default="LeverageDipSplit",
        help="Strategy class name within the module.",
    )
    parser.add_argument(
        "--strategy-kwargs",
        default='{"dip_pct":0.04,"base_weight":0.9,"weight_a":0.05,"lev_a":10,"weight_b":0.05,"lev_b":20,"dd_trigger":0.4,"exit_recover_frac":0.95}',
        help='JSON dict of init kwargs.',
    )
    parser.add_argument("--mu-grid", default="0.05,0.10,0.15", help="Comma sep annualized drift values")
    parser.add_argument("--sigma-grid", default="0.10,0.20,0.30", help="Comma sep annualized vol values")
    parser.add_argument("--paths", type=int, default=2000, help="Monte Carlo paths per grid point")
    parser.add_argument("--steps", type=int, default=2520, help="Time steps (e.g., 10y daily ~2520)")
    parser.add_argument("--dt", type=float, default=1 / 252, help="Time step size in years")
    parser.add_argument("--s0", type=float, default=100.0, help="Starting price")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--out", default="artifacts/gbm_sim_results.csv", help="Output CSV path")
    args = parser.parse_args()

    strategy_kwargs = json.loads(args.strategy_kwargs)
    strategy = load_strategy(args.strategy_module, args.strategy_class, strategy_kwargs)

    mu_list = [float(x) for x in args.mu_grid.split(",")]
    sigma_list = [float(x) for x in args.sigma_grid.split(",")]

    records = []
    for mu in mu_list:
        for sigma in sigma_list:
            prices = simulate_gbm(mu, sigma, steps=args.steps, paths=args.paths, dt=args.dt, s0=args.s0, seed=args.seed)
            finals, dds = evaluate_strategy_on_paths(strategy, prices, dt=args.dt)
            rec = {
                "mu": mu,
                "sigma": sigma,
                "paths": args.paths,
                "steps": args.steps,
                "final_p10": float(np.percentile(finals, 10)),
                "final_p50": float(np.percentile(finals, 50)),
                "final_p90": float(np.percentile(finals, 90)),
                "final_mean": float(np.mean(finals)),
                "dd_p10": float(np.percentile(dds, 10)),
                "dd_p50": float(np.percentile(dds, 50)),
                "dd_p90": float(np.percentile(dds, 90)),
            }
            records.append(rec)
            print(
                f"mu={mu:.2%} sigma={sigma:.2%} "
                f"final_p50={rec['final_p50']:.2f} dd_p50={rec['dd_p50']:.2%}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()

