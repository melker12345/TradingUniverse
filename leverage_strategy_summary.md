# Leveraged Overlay Strategy Summary

## Strategy
- Core: long-only buy-and-hold across the basket (equal-weight).
- Overlay (best config from sweep):
  - Base/core: 90% capital always long 1x.
  - Sleeve A: deploy 5% at 10x when price is ≥4% below prior peak.
  - Sleeve B: if Sleeve A mark-to-market drawdown reaches -40%, deploy another 5% at 20x.
  - Exit both sleeves when price recovers to 95% of the peak at which Sleeve A opened.

## Performance (Sweden top-10 daily, 10y, no costs)
- Buy & Hold final equity: 4.03
- Best split-sleeve final equity: 7.27
  - Annualized return: ~21.98%
  - Max drawdown: ~-45.0%
  - Sharpe: ~0.87
  - Beta: ~1.36

## Tweakable parameters
- dip_pct (overlay trigger): 4–10%
- base_weight: 0.7–0.9
- Sleeve A: weight (e.g., 5%), leverage (5x, 10x)
- Sleeve B: weight (e.g., 5%), leverage (10x, 15x, 20x)
- dd_trigger (A drawdown to add B): 40–60%
- exit_recover_frac: 95–100% of prior peak to exit sleeves

## Validation / next steps
- Re-run on extended Swedish large/mid-cap universe (script in examples/run_leverage_on_extended.py).
- Apply the split-sleeve sweep on the extended set (examples/sweep_leverage_dip_split.py) to confirm best params hold out-of-sample.
- Plot equity vs buy-and-hold: run examples/plot_leverage_best.py (saves artifacts/equity_leverage_split.png).
- Consider adding modest costs/slippage and assessing robustness; also test shorter periods to check regime sensitivity.

