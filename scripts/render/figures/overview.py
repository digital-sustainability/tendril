"""Overview figure of the TENDRIL test set (Fig. 2 of the paper).

Panel (a) shows per-catalog statistics (``sectors.py``), panel (b) positives
against passages per project (``project_scatter.py``). Both panels are drawn at
the LNCS text width, so the figure is included at 100% scale and the 8 pt text
prints at 8 pt.

Example::

    python scripts/render/figures/overview.py --output figures
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import project_scatter
import seaborn as sns
import sectors
from loguru import logger
from style import LNCS_RC, save_figure

from tendril.hub import TEST_REPO, download_test_set

# LNCS text width: 12.2 cm.
TEXT_WIDTH = 4.8
SCATTER_HEIGHT = 1.6
ROW_HEIGHT = 0.13
# Blank band between the panels; the subfigure hspace has no visible effect here.
PANEL_GAP = 0.12
# Shorter sector names so the label column leaves room for the panels.
SHORT_LABELS = {
    "Construction_Civil_Engineering_Infrastructure": "Constr_Civil_Eng_Infra",
    "Wastepaper_Cleaning_Refuse": "Wastepaper_Cleaning",
}


def plot_overview(
    sector_rows: list[sectors.SectorStats],
    total: sectors.SectorStats,
    projects: list[project_scatter.ProjectCounts],
    output_dir: Path,
) -> None:
    if shutil.which("latex") is None:
        raise SystemExit("LaTeX is required to render the figure in the LNCS font")
    sns.set_theme(style="white", context="paper", palette="colorblind", rc=LNCS_RC)
    top_height = sectors.sectors_height(sector_rows, ROW_HEIGHT)
    fig = plt.figure(
        figsize=(TEXT_WIDTH, top_height + PANEL_GAP + SCATTER_HEIGHT), layout="constrained"
    )
    fig.get_layout_engine().set(h_pad=0.01, hspace=0.0)
    top, _gap, bottom = fig.subfigures(
        3, 1, height_ratios=(top_height, PANEL_GAP, SCATTER_HEIGHT), hspace=0.0
    )
    sectors.draw_sectors(top, sector_rows, total, SHORT_LABELS)
    project_scatter.draw_passages_vs_positives(bottom, projects)
    for subfigure, letter in ((top, "a"), (bottom, "b")):
        subfigure.text(0.0, 1.0, f"({letter})", ha="left", va="top", fontweight="bold")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "overview.pdf", bbox_inches="tight")
    save_figure(fig, output_dir / "overview.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot per-sector counts and per-project passages and positives of the TENDRIL test set"
    )
    parser.add_argument(
        "--repo", default=TEST_REPO, help=f"Hub dataset repo (default: {TEST_REPO})"
    )
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
    project_dirs = download_test_set(args.repo, args.revision)
    sector_projects = [sectors.load_project(path) for path in project_dirs]
    projects = [project_scatter.load_project(path) for path in project_dirs]
    plot_overview(
        sectors.sector_stats(sector_projects),
        sectors.aggregate("Overall", sector_projects),
        projects,
        args.output,
    )
    logger.info(f"Wrote overview.png and overview.pdf to {args.output}")


if __name__ == "__main__":
    main()
