"""Per-catalog statistics of the TENDRIL datasets as dot-plot panels.

For the test set, each project belongs to one catalog (its sector); ``overview.py``
draws these statistics as panel (a) of the paper's overview figure. For the
training set, a project can be judged against several catalogs, so the rows group
the judgments by the catalog of the query; projects, documents, and passages then
repeat across catalogs, and the panel titles give the unique totals. Run as a
script, this module renders the training-set figure of the TENDRIL-Train card.

Example::

    python scripts/render/figures/sectors.py --output figures
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.figure import SubFigure
from matplotlib.ticker import NullLocator
from style import DOT_COLOR, FIGURE_WIDTH, LNCS_RC, MEAN_COLOR, save_figure

from tendril.hub import TRAIN_REPO, download_train_set

ROW_HEIGHT = 0.15


@dataclass(frozen=True)
class ProjectStats:
    sector: str
    query_ids: frozenset[str]
    n_queries: int
    n_documents: int
    n_passages: int
    n_judged: int
    n_positive: int


@dataclass(frozen=True)
class SectorStats:
    sector: str
    n_projects: int
    n_queries: int
    n_unique_queries: int
    n_documents: int
    n_passages: int
    n_judged: int
    n_positive: int

    @property
    def positives_per_query(self) -> float:
        return self.n_positive / self.n_queries if self.n_queries else 0.0

    @property
    def positive_rate(self) -> float:
        return self.n_positive / self.n_judged if self.n_judged else 0.0


def sector_of(project_name: str) -> str:
    sector, separator, suffix = project_name.rpartition("-")
    if separator and suffix.isalnum():
        return sector
    return project_name


def count_documents(corpus_ids: pd.Series) -> int:
    """Count source documents via the hashes in ``positive::<doc_hash>::<span>`` IDs.

    Negative passages keep their pipeline UUIDs, but the builder drops every
    document without a positive span, so each document has a positive ID.
    """
    positives = corpus_ids[corpus_ids.str.startswith("positive::")]
    return positives.str.split("::").str[1].nunique()


def load_project(project_dir: Path) -> ProjectStats:
    queries = pd.read_parquet(project_dir / "queries.parquet", columns=["id"])
    corpus = pd.read_parquet(project_dir / "corpus.parquet", columns=["id"])
    qrels = pd.read_parquet(project_dir / "qrels.parquet", columns=["score"])
    return ProjectStats(
        sector=sector_of(project_dir.name),
        query_ids=frozenset(queries["id"].astype(str)),
        n_queries=len(queries),
        n_documents=count_documents(corpus["id"].astype(str)),
        n_passages=len(corpus),
        n_judged=len(qrels),
        n_positive=int((qrels["score"] > 0).sum()),
    )


def aggregate(sector: str, projects: Sequence[ProjectStats]) -> SectorStats:
    return SectorStats(
        sector=sector,
        n_projects=len(projects),
        n_queries=sum(project.n_queries for project in projects),
        n_unique_queries=len(frozenset().union(*(p.query_ids for p in projects))),
        n_documents=sum(project.n_documents for project in projects),
        n_passages=sum(project.n_passages for project in projects),
        n_judged=sum(project.n_judged for project in projects),
        n_positive=sum(project.n_positive for project in projects),
    )


def sector_stats(projects: Sequence[ProjectStats]) -> list[SectorStats]:
    by_sector: dict[str, list[ProjectStats]] = defaultdict(list)
    for project in projects:
        by_sector[project.sector].append(project)
    stats = [aggregate(sector, members) for sector, members in by_sector.items()]
    return sorted(stats, key=lambda row: (-row.n_positive, row.sector))


def train_stats(data: Path) -> tuple[list[SectorStats], SectorStats]:
    """Group the training judgments by query catalog and return them with the totals."""
    corpus = pd.read_parquet(data / "corpus.parquet", columns=["id", "project", "document"])
    queries = pd.read_parquet(data / "queries.parquet", columns=["id", "catalog"])
    qrels = pd.read_parquet(data / "qrels.parquet")
    judged = qrels.merge(corpus, left_on="corpus-id", right_on="id").merge(
        queries, left_on="query-id", right_on="id", suffixes=("", "_query")
    )
    if len(judged) != len(qrels):
        raise SystemExit(f"{len(qrels) - len(judged)} qrels reference unknown queries or passages")
    judged["positive"] = judged["score"] > 0

    def summarise(sector: str, rows: pd.DataFrame) -> SectorStats:
        n_queries = rows["query-id"].nunique()
        return SectorStats(
            sector=sector,
            n_projects=rows["project"].nunique(),
            n_queries=n_queries,
            n_unique_queries=n_queries,
            n_documents=rows["document"].nunique(),
            n_passages=rows["corpus-id"].nunique(),
            n_judged=len(rows),
            n_positive=int(rows["positive"].sum()),
        )

    stats = [summarise(str(catalog), rows) for catalog, rows in judged.groupby("catalog")]
    stats.sort(key=lambda row: (-row.n_positive, row.sector))
    return stats, summarise("Overall", judged)


@dataclass(frozen=True)
class Panel:
    title: str
    value: Callable[[SectorStats], float]
    label: Callable[[float], str]
    reference: float | None = None


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def count_label(value: float) -> str:
    return f"{int(value):,}"


def sector_label(row: SectorStats, labels: Mapping[str, str] | None = None) -> str:
    name = (labels or {}).get(row.sector, row.sector)
    return tex_escape(f"{name} ({row.n_projects})")


def draw_panel(ax: plt.Axes, positions: list[int], values: list[float], panel: Panel) -> None:
    high = max(values) if values else 1.0
    ax.hlines(positions, 0, values, color=DOT_COLOR, linewidth=1.0, zorder=2)
    ax.scatter(
        values, positions, s=9, color=DOT_COLOR, linewidths=0, zorder=3, clip_on=False
    )
    for position, value in zip(positions, values):
        ax.text(
            value + high * 0.06,
            position,
            tex_escape(panel.label(value)),
            va="center",
            ha="left",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3},
            zorder=4,
        )
    if panel.reference is not None:
        ax.axvline(
            panel.reference, color=MEAN_COLOR, linewidth=0.6, linestyle=":", zorder=1
        )
    # Leave a small margin left of zero so dots of tiny values are not cut off.
    ax.set_xlim(-high * 0.04, high * 1.4)
    ax.xaxis.set_major_locator(NullLocator())
    ax.grid(axis="y", color="0.92", linewidth=0.5, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_title(tex_escape(panel.title), loc="left", pad=3)
    sns.despine(ax=ax, left=True, bottom=True)
    ax.tick_params(axis="y", length=0)


def draw_sectors(
    fig: plt.Figure | SubFigure,
    stats: list[SectorStats],
    total: SectorStats,
    labels: Mapping[str, str] | None = None,
) -> None:
    """Draw one dot-plot panel per metric; ``labels`` shortens sector names."""
    sigma = "$\\Sigma$"
    panels = (
        Panel(f"Queries\n({sigma} {total.n_queries:,})", lambda r: r.n_queries, count_label),
        Panel(
            f"Docs\n({sigma} {total.n_documents:,})", lambda r: r.n_documents, count_label
        ),
        Panel(
            f"Passages\n({sigma} {total.n_passages:,})", lambda r: r.n_passages, count_label
        ),
        Panel(
            f"Positives\n({sigma} {total.n_positive:,})",
            lambda r: r.n_positive,
            count_label,
        ),
        Panel(
            f"Positive rate\n({total.positive_rate:.1%})",
            lambda r: r.positive_rate * 100,
            lambda v: f"{v:.1f}%",
            reference=total.positive_rate * 100,
        ),
    )
    positions = list(range(len(stats)))
    axes = fig.subplots(1, len(panels), sharey=True)
    for ax, panel in zip(axes, panels):
        draw_panel(ax, positions, [panel.value(row) for row in stats], panel)
    axes[0].set_yticks(positions, [sector_label(row, labels) for row in stats])
    axes[0].set_ylim(len(stats) - 0.5, -0.5)


def sectors_height(stats: Sequence[SectorStats], row_height: float = ROW_HEIGHT) -> float:
    return row_height * len(stats) + 0.45


def plot_sectors(
    stats: list[SectorStats], total: SectorStats, output_dir: Path, stem: str
) -> None:
    if shutil.which("latex") is None:
        raise SystemExit("LaTeX is required to render the figure in the LNCS font")
    sns.set_theme(style="white", context="paper", palette="colorblind", rc=LNCS_RC)
    fig = plt.figure(figsize=(FIGURE_WIDTH, sectors_height(stats)), layout="constrained")
    draw_sectors(fig, stats, total)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    save_figure(fig, output_dir / f"{stem}.png")


def log_table(rows: Sequence[SectorStats]) -> None:
    width = max(len(row.sector) for row in rows)
    header = (
        f"{'Sector':<{width}}  Proj  Queries  Unique  Docs  Passages  Judged   Pos  Pos/q   Rate"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"{row.sector:<{width}}  {row.n_projects:>4}  {row.n_queries:>7}  "
            f"{row.n_unique_queries:>6}  {row.n_documents:>4}  {row.n_passages:>8}  {row.n_judged:>6}  "
            f"{row.n_positive:>4}  {row.positives_per_query:>5.1f}  "
            f"{row.positive_rate:>5.1%}"
        )
    logger.info("Per-sector statistics:\n" + "\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot per-catalog query, document, passage, and qrel counts of TENDRIL-Train"
    )
    parser.add_argument("--repo", default=TRAIN_REPO, help=f"Hub dataset repo (default: {TRAIN_REPO})")
    parser.add_argument(
        "--revision", default="main", help="Hub revision to read (default: main)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures"),
        help="Directory for the plots (default: figures)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    stats, total = train_stats(download_train_set(args.repo, args.revision))
    stem = "sectors_train"
    plot_sectors(stats, total, args.output, stem)
    log_table([*stats, total])
    logger.info(f"Wrote {stem}.png and {stem}.pdf to {args.output}")


if __name__ == "__main__":
    main()
