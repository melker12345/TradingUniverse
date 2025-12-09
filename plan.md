# Levered Overlay Plan

Goal: keep a core buy-and-hold sleeve while using a small overlay that deploys leverage on dips. We will compare two overlays and optimize their parameters for payout, acknowledging drawdown/leverage decay.

## Overlay variants
1) Single-sleeve levered dip (already implemented as `LeverageDip`):
   - Base weight stays long 100% notional at `base_weight` (e.g., 0.8–0.9).
   - On dip >= `dip_pct` from peak, deploy remaining capital in a leveraged long (`leverage`).
   - Exit overlay when price recovers to prior peak.

2) Split-sleeve with double-down:
   - Core base weight stays long (optimize in 0.7–0.9).
   - Sleeve A: on dip >= `dip_pct`, deploy `sleeve_a_weight` (e.g., 5% of total) at `lev_a` (e.g., 5x–10x).
   - If Sleeve A falls `dd_trigger` (e.g., 50%) mark-to-market loss, then deploy Sleeve B: another `sleeve_b_weight` (e.g., 5% of total) at higher leverage `lev_b` (e.g., 10x–20x).
   - Both sleeves exit when price recovers to prior peak (or when their own entry peak is reclaimed).

## Parameters to sweep
- Base/core weight: 0.7, 0.8, 0.9
- Dip trigger: 4%, 6%, 8%, 10%
- Sleeve A: weight {2.5%, 5%, 10%}, leverage {5x, 10x}
- Sleeve B (double-down): weight {2.5%, 5%, 10%}, leverage {10x, 15x, 20x}
- Double-down trigger: MTM drawdown on Sleeve A of {40%, 50%, 60%}
- Exit: recover to entry peak (current), optionally test trailing stop on overlay (e.g., -30% from overlay peak)

## Metrics / selection
- Primary: final equity / annualized return (payout focus)
- Risk notes: max drawdown, beta, Sharpe/Sortino, overlay time-on-risk
- Report top configs and compare to buy-and-hold baseline

## Risks / considerations
- Leverage decay and volatility drag are significant; higher leverage + double-down increases tail risk and max drawdown.
- Liquidity/financing/slippage not modeled here; results are optimistic without costs.

## Next steps
- Implement split-sleeve overlay with double-down logic and run the above grid.
- Surface the top 5 configs by final equity and their risk stats vs buy-and-hold.

