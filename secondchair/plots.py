"""Shared look for the project's diagnostic charts (PNG via matplotlib, no display needed)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834")  # fixed order: first series blue, second orange


def chart(title: str, xlabel: str, ylabel: str, size: tuple[float, float] = (10, 4)) -> tuple[Figure, Axes]:
    """A figure with one recessive-gridded axis, left-aligned title and muted labels."""
    fig, ax = plt.subplots(figsize=size, dpi=120, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED)
    return fig, ax


def save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
