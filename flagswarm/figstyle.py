"""Publication style following figures4papers (vendor/figures4papers/scientific-figure-making/references/api.md
and design-theory.md): semantic palette, top/right spines off, frameless legends, tight_layout(pad=2), dpi 300."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700", "teal": "#42949E", "violet": "#9A4D8E",
    "gray_dark": "#4D4D4D",
}
DEFAULT_COLORS = [PALETTE[k] for k in ("blue_main", "green_3", "red_strong", "teal", "violet", "neutral")]


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2.0
    font_family: tuple[str, ...] = ("Arial", "Helvetica", "DejaVu Sans", "sans-serif")


def apply_publication_style(style: FigureStyle | None = None) -> None:
    s = style or FigureStyle()
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = [f for f in s.font_family if f in installed or f == "sans-serif"]   # Arial/Helvetica when present
    plt.rcParams.update({
        "font.family": family, "font.size": s.font_size,
        "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": s.axes_linewidth,
        "legend.frameon": False, "svg.fonttype": "none", "lines.linewidth": 2.5,
    })


def create_subplots(nrows=1, ncols=1, figsize=None, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, **kw)
    return fig, axes.ravel()


def finalize_figure(fig, out_path, formats=("png", "pdf"), dpi=300, pad=2.0, close=True) -> list[Path]:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=pad)
    paths = []
    for fmt in formats:
        p = out.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=dpi, facecolor="white")
        paths.append(p)
    if close:
        plt.close(fig)
    return paths


def make_trend(ax, x, y_series, labels, colors=None, lo=None, hi=None, ylabel=None, xlabel=None, styles=None):
    colors = colors or DEFAULT_COLORS
    for k, y in enumerate(y_series):
        ax.plot(x, y, color=colors[k % len(colors)], label=labels[k], ls=(styles[k] if styles else "-"), alpha=0.95)
        if lo is not None and lo[k] is not None:
            ax.fill_between(x, lo[k], hi[k], color=colors[k % len(colors)], alpha=0.2, lw=0)
    if ylabel: ax.set_ylabel(ylabel)
    if xlabel: ax.set_xlabel(xlabel)
