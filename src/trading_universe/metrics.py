from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.empty:
        return 0.0
    compounded = float((1 + returns).prod())
    n_periods = returns.shape[0]
    return compounded ** (periods_per_year / n_periods) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(returns.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    if returns.empty:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    vol = annualized_volatility(excess, periods_per_year)
    return float(excess.mean() * periods_per_year / vol) if vol != 0 else 0.0


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    if returns.empty:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    downside_vol = float(downside.std(ddof=0) * np.sqrt(periods_per_year))
    return float(excess.mean() * periods_per_year / downside_vol) if downside_vol != 0 else 0.0


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity = (1 + returns).cumprod()
    peaks = equity.cummax()
    drawdowns = (equity - peaks) / peaks
    return float(drawdowns.min())


def hit_rate(returns: pd.Series) -> float:
    return float((returns > 0).mean()) if not returns.empty else 0.0


def profit_factor(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    return float(gains / abs(losses)) if losses != 0 else float("inf")


def average_gain_loss(returns: pd.Series) -> Dict[str, float]:
    return {
        "avg_gain": float(returns[returns > 0].mean() or 0.0),
        "avg_loss": float(returns[returns < 0].mean() or 0.0),
    }


def alpha_beta(
    returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = 252
) -> Dict[str, float]:
    if returns.empty or benchmark_returns.empty:
        return {"alpha": 0.0, "beta": 0.0}
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return {"alpha": 0.0, "beta": 0.0}
    y = aligned.iloc[:, 0].values
    x = aligned.iloc[:, 1].values
    x = np.vstack([np.ones_like(x), x]).T
    coeffs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    alpha_daily, beta = coeffs
    alpha = float((1 + alpha_daily) ** periods_per_year - 1)
    return {"alpha": alpha, "beta": float(beta)}


def compute_all_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "annualized_return": annualized_return(returns, periods_per_year),
        "volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "hit_rate": hit_rate(returns),
        "profit_factor": profit_factor(returns),
    }
    metrics.update(average_gain_loss(returns))

    if benchmark_returns is not None:
        metrics.update(alpha_beta(returns, benchmark_returns, periods_per_year))

    return metrics

