from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("ggplot")


def _maybe_save(fig: plt.Figure, output_dir: Optional[str], name: str) -> None:
    if output_dir is None:
        fig.show()
    else:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / name, bbox_inches="tight")
    plt.close(fig)


def plot_equity_curves(equity_curves: Dict[str, pd.Series], output_dir: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, series in equity_curves.items():
        series.plot(ax=ax, label=label)
    ax.set_title("Equity Curves")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    _maybe_save(fig, output_dir, "equity_curves.png")


def plot_drawdowns(equity_curves: Dict[str, pd.Series], output_dir: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, series in equity_curves.items():
        peaks = series.cummax()
        drawdown = (series - peaks) / peaks
        drawdown.plot(ax=ax, label=label)
    ax.set_title("Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.legend()
    _maybe_save(fig, output_dir, "drawdowns.png")

