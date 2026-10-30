"""Panel (b) of the test-set overview figure: positives against passages per project.

Each project is one dot on log axes, with a box plot of the passage counts above. ``overview.py`` draws this panel
below the per-catalog panel of ``sectors.py``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import SubFigure
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from style import DOT_COLOR, MEAN_COLOR

LOG_TICKS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000)
DECADE_TICKS = (1, 10, 100, 1_000, 10_000)
# Padding factor around the data on the log axes.
LOG_PAD = 1.35


@dataclass(frozen=True)
class ProjectCounts:
    n_queries: int
    n_documents: int
    relevant_per_query: tuple[int, ...]

    @property
    def n_positive(self) -> int:
        return sum(self.relevant_per_query)


def load_project(project_dir: Path) -> ProjectCounts:
    queries = pd.read_parquet(project_dir / "queries.parquet", columns=["id"])
    corpus = pd.read_parquet(project_dir / "corpus.parquet", columns=["id"])
    qrels = pd.read_parquet(project_dir / "qrels.parquet", columns=["query-id", "score"])
    # Count score > 0 judgments per query.
    positives = Counter(qrels.loc[qrels["score"] > 0, "query-id"].astype(str))
    return ProjectCounts(
        n_queries=len(queries),
        n_documents=len(corpus),
        relevant_per_query=tuple(positives[query_id] for query_id in queries["id"].astype(str)),
    )


def set_log_axis(
    axis: matplotlib.axis.Axis, low: float, high: float, ticks: Sequence[int] = LOG_TICKS
) -> None:
    axis.set_major_locator(FixedLocator([tick for tick in ticks if low <= tick <= high]))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(round(value)):,}"))


def draw_margin_box(ax: plt.Axes, values: list[int], *, vertical: bool) -> None:
    ax.boxplot(
        values,
        orientation="vertical" if vertical else "horizontal",
        widths=0.5,
        patch_artist=True,
        boxprops={"facecolor": "white", "edgecolor": DOT_COLOR, "linewidth": 0.8},
        medianprops={"color": DOT_COLOR, "linewidth": 1.4},
        whiskerprops={"color": DOT_COLOR, "linewidth": 0.8},
        capprops={"color": DOT_COLOR, "linewidth": 0.8},
        flierprops={
            "marker": "o",
            "markersize": 2.5,
            "markerfacecolor": DOT_COLOR,
            "markeredgewidth": 0,
        },
    )
    ax.axis("off")


def draw_passages_vs_positives(fig: plt.Figure | SubFigure, projects: list[ProjectCounts]) -> None:
    """Draw the scatter with a box plot of the passage counts above it."""
    passages = [project.n_documents for project in projects]
    positives = [project.n_positive for project in projects]
    queries = [project.n_queries for project in projects]
    x_range = (min(passages) / LOG_PAD, max(passages) * LOG_PAD)
    y_range = (max(min(positives), 1) / LOG_PAD, max(positives) * LOG_PAD)

    grid = fig.add_gridspec(2, 1, height_ratios=(1, 7))
    ax = fig.add_subplot(grid[1, 0])
    top = fig.add_subplot(grid[0, 0], sharex=ax)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(
        passages,
        positives,
        s=14,
        color=DOT_COLOR,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    set_log_axis(ax.xaxis, *x_range)
    # A short panel only has room for one tick per decade.
    set_log_axis(ax.yaxis, *y_range, DECADE_TICKS)
    ax.grid(color="0.92", linewidth=0.5, zorder=0)
    ax.set(xlabel="Passages per project", ylabel="Positives per project")
    ax.text(
        0.97,
        0.03,
        f"$n = {len(projects)}$ projects\n"
        "Median per project:\n"
        + ", ".join(
            (
                f"{median(queries):,.0f} queries",
                f"{median(passages):,.0f} passages",
                f"{median(positives):,.0f} positives",
            )
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=MEAN_COLOR,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
        zorder=4,
    )
    sns.despine(ax=ax)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.6)

    draw_margin_box(top, passages, vertical=False)
