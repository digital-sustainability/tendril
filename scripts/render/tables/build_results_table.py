"""Build a hierarchical results table (LaTeX + terminal) from evaluation JSON.

Scans a directory of per-model score files written by ``evaluate_models.py``,
maps each file back to a reviewed TENDRIL model via ``tendril.models.registry``,
and renders a table grouped as:

    Baseline                   -- BM25
    Open-Weight                -- BGE-M3 and the other local dense embedding models, as released
    Open-Weight (fine-tuned)   -- those same models after task fine-tuning
    Proprietary                -- API-based embedding models
    Oracle                     -- the reference ranking

Each group is read from a fixed subdirectory of the results directory
(``baseline/``, ``open-weight/``, ``fine-tuned/``, ``proprietary/``,
``oracle/``). A file is matched to a registry model by its
``model`` field or, failing that, by its file name. A fine-tuned checkpoint
is recorded as the open-weight id plus ``__TENDRIL``
(``Snowflake/snowflake-arctic-embed-l-v2.0__TENDRIL``), and that suffix is
removed before the lookup.

Within each group the best value per column is underlined; the best value
per column across all competing systems (Oracle excluded, since it is
a ceiling by construction, not a competitor) is bold.

Example::

    python scripts/render/tables/build_results_table.py
    python scripts/render/tables/build_results_table.py results --latex-output results/table.tex
    python scripts/render/tables/build_results_table.py results --ci-size footnotesize --ci-color black
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from latex_escape import latex_escape
from rich import box
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from tendril.models.registry import (
    KNOWN_MODELS,
    ORACLE_NAME,
    KnownModel,
    known_models_by_name,
    output_stem,
)

console = Console()

DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_LATEX_STEM = "results_table.tex"
DEFAULT_CI_SIZE = "scriptsize"
DEFAULT_CI_COLOR = "gray"

# Column order requested for the table (key must match the "scores"/"intervals"
# keys written by evaluate_models.py, e.g. "NDCG@10").
TABLE_METRICS = ("NDCG@10", "Precision@10", "Recall@10", "MAP@10", "MRR@10")
METRIC_HEADERS = {
    "NDCG@10": "NDCG@10",
    "Precision@10": "Precision@10",
    "Recall@10": "Recall@10",
    "MAP@10": "MAP@10",
    "MRR@10": "MRR@10",
}

ORACLE_GROUP = "Oracle"

# Subdirectory of the results root -> table heading, in table order.
# "Fine-tuned" alone would hide that these checkpoints are the open-weight
# models after adaptation. A parenthetical keeps the family, and stays readable
# in a paper table (unlike a marker symbol).
RESULT_DIRECTORIES = {
    "baseline": "Baseline",
    "open-weight": "Open-Weight",
    "fine-tuned": "Open-Weight (fine-tuned)",
    "proprietary": "Proprietary",
    "oracle": ORACLE_GROUP,
}
GROUP_ORDER = tuple(RESULT_DIRECTORIES.values())
# Fine-tuned checkpoints are the open-weight model after adaptation on TENDRIL.
FINE_TUNED_SUFFIX = "__TENDRIL"

DISPLAY_NAMES = {
    "mteb/baseline-bm25s": "BM25",
    "BAAI/bge-m3": "BGE-M3",
    "malteos/most-embed-de": "most-embed-de",
    "nvidia/Nemotron-3-Embed-1B-BF16": "Nemotron-3-Embed-1B",
    "Snowflake/snowflake-arctic-embed-l-v2.0": "Arctic-Embed-L-v2.0",
    "openai/text-embedding-3-large": "text-embedding-3-large",
    "voyageai/voyage-4-large": "voyage-4-large",
    "google/gemini-embedding-2": "gemini-embedding-2",
    ORACLE_NAME: "Oracle",
}


def resolve_model(file: Path, raw_name: str, by_name: dict[str, KnownModel]) -> KnownModel | None:
    """Registry model for a result file, by its ``model`` field or its file name.

    A fine-tuned result uses the open-weight id plus ``__TENDRIL``. That suffix
    is removed before the lookup.
    """
    known = by_name.get(raw_name) or by_name.get(raw_name.removesuffix(FINE_TUNED_SUFFIX))
    if known is not None:
        return known
    by_stem = {output_stem(model.name): model for model in KNOWN_MODELS}
    return by_stem.get(file.stem.removesuffix(FINE_TUNED_SUFFIX))


def groups_in_order(results: Sequence[ModelResult]) -> list[str]:
    present = {result.group for result in results}
    return [group for group in GROUP_ORDER if group in present]


@dataclass(frozen=True)
class ModelResult:
    canonical_name: str
    display_name: str
    group: str
    n: int
    scores: dict[str, float]
    intervals: dict[str, tuple[float, float]] | None
    source_file: Path


def row_id(result: ModelResult) -> tuple[str, str]:
    return (result.group, result.canonical_name)


def overall_row(payload: Mapping[str, object]) -> Mapping[str, object] | None:
    for row in payload.get("rows", []):  # type: ignore[union-attr]
        if row.get("sector") == "Overall":
            return row
    return None


def candidate_key(file: Path, payload: Mapping[str, object], canonical: str) -> tuple:
    row = overall_row(payload)
    has_overall = row is not None
    has_intervals = bool(row and "intervals" in row)
    stem_matches = file.stem.removesuffix(FINE_TUNED_SUFFIX) == output_stem(canonical)
    mtime = file.stat().st_mtime
    return (has_overall, has_intervals, stem_matches, mtime)


def load_results(directory: Path) -> list[ModelResult]:
    if not directory.is_dir():
        raise SystemExit(f"ERROR: {directory} is not a directory")

    by_name = known_models_by_name()
    candidates: dict[tuple[str, str], list[tuple[Path, dict]]] = {}

    for subdirectory, group in RESULT_DIRECTORIES.items():
        for file in sorted((directory / subdirectory).glob("*.json")):
            relative = file.relative_to(directory)
            try:
                payload = json.loads(file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                console.print(f"[yellow]Skipping {relative}: invalid JSON ({error})[/yellow]")
                continue

            raw_name = payload.get("model")
            if not isinstance(raw_name, str):
                console.print(f"[yellow]Skipping {relative}: no 'model' field[/yellow]")
                continue

            known = resolve_model(file, raw_name, by_name)
            if known is None:
                console.print(
                    f"[yellow]Skipping {relative}: unrecognized model '{raw_name}' "
                    "(not in tendril.models registry)[/yellow]"
                )
                continue

            candidates.setdefault((group, known.name), []).append((file, payload))

    results: list[ModelResult] = []
    for (group, canonical), files in candidates.items():
        if len(files) > 1:
            files.sort(key=lambda item: candidate_key(item[0], item[1], canonical), reverse=True)
            chosen_file, chosen_payload = files[0]
            skipped = ", ".join(f.name for f, _ in files[1:])
            console.print(
                f"[yellow]Multiple result files for {canonical} in {group}: "
                f"using {chosen_file.name}, ignoring {skipped}[/yellow]"
            )
        else:
            chosen_file, chosen_payload = files[0]

        row = overall_row(chosen_payload)
        if row is None:
            console.print(f"[yellow]Skipping {chosen_file.name}: no 'Overall' row[/yellow]")
            continue

        scores = row.get("scores", {})
        missing = [metric for metric in TABLE_METRICS if metric not in scores]
        if missing:
            console.print(f"[yellow]Skipping {chosen_file.name}: missing metrics {missing}[/yellow]")
            continue

        raw_intervals = row.get("intervals")
        intervals = None
        if raw_intervals:
            intervals = {
                metric: (float(raw_intervals[metric][0]), float(raw_intervals[metric][1]))
                for metric in TABLE_METRICS
                if metric in raw_intervals
            }

        results.append(
            ModelResult(
                canonical_name=canonical,
                display_name=DISPLAY_NAMES.get(canonical, canonical),
                group=group,
                n=int(row.get("n", 0)),
                scores={metric: float(scores[metric]) for metric in TABLE_METRICS},
                intervals=intervals,
                source_file=chosen_file,
            )
        )

    order = {model.name: index for index, model in enumerate(KNOWN_MODELS)}
    group_order = {name: index for index, name in enumerate(GROUP_ORDER)}
    results.sort(
        key=lambda result: (
            order.get(result.canonical_name, len(order)),
            group_order.get(result.group, len(group_order)),
        )
    )
    return results


def group_best(results: Sequence[ModelResult]) -> dict[str, dict[str, set[tuple[str, str]]]]:
    """Per-group, per-metric set of row ids tied for the best score."""
    best: dict[str, dict[str, set[tuple[str, str]]]] = {}
    by_group: dict[str, list[ModelResult]] = {}
    for result in results:
        by_group.setdefault(result.group, []).append(result)

    for group, members in by_group.items():
        if len(members) < 2:
            continue
        metric_best: dict[str, set[tuple[str, str]]] = {}
        for metric in TABLE_METRICS:
            top = max(member.scores[metric] for member in members)
            metric_best[metric] = {
                row_id(member) for member in members if round(member.scores[metric], 5) == round(top, 5)
            }
        best[group] = metric_best
    return best


def overall_best(results: Sequence[ModelResult]) -> dict[str, set[tuple[str, str]]]:
    """Per-metric set of row ids tied for the best score, excluding Oracle."""
    competitors = [result for result in results if result.group != ORACLE_GROUP]
    best: dict[str, set[tuple[str, str]]] = {}
    for metric in TABLE_METRICS:
        if not competitors:
            best[metric] = set()
            continue
        top = max(member.scores[metric] for member in competitors)
        best[metric] = {row_id(member) for member in competitors if round(member.scores[metric], 5) == round(top, 5)}
    return best


def format_parts(result: ModelResult, metric: str) -> tuple[str, str | None]:
    """Point estimate and CI text, kept separate so styling can target the estimate only."""
    value = f"{result.scores[metric]:.3f}"
    if result.intervals is None:
        return value, None
    lo, hi = result.intervals[metric]
    return value, f"[{lo:.3f}, {hi:.3f}]"


def format_ci_latex(ci: str, *, size: str, color: str) -> str:
    """Wrap a CI in a LaTeX group with the requested font size and color."""
    if not size and not color:
        return ci
    body = rf"\{size} {ci}" if size else ci
    if color:
        body = rf"\color{{{color}}}{body}"
    return f"{{{body}}}"


def render_latex(
    results: Sequence[ModelResult],
    *,
    group_bests: dict[str, dict[str, set[tuple[str, str]]]],
    overall_bests: dict[str, set[tuple[str, str]]],
    ci_size: str = DEFAULT_CI_SIZE,
    ci_color: str = DEFAULT_CI_COLOR,
) -> str:
    by_group: dict[str, list[ModelResult]] = {}
    for result in results:
        by_group.setdefault(result.group, []).append(result)

    num_columns = 1 + len(TABLE_METRICS)
    lines: list[str] = []
    lines.append(r"\begin{tabular}{l" + "l" * len(TABLE_METRICS) + "}")
    lines.append(r"\toprule")
    header = "System & " + " & ".join(METRIC_HEADERS[m] for m in TABLE_METRICS) + r" \\"
    lines.append(header)

    present_groups = groups_in_order(results)
    for group in present_groups:
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{num_columns}}}{{l}}{{\textit{{{latex_escape(group)}}}}} \\")
        for result in by_group[group]:
            cells = [latex_escape(result.display_name)]
            for metric in TABLE_METRICS:
                value, ci = format_parts(result, metric)
                is_group_best = row_id(result) in group_bests.get(group, {}).get(metric, set())
                is_overall_best = row_id(result) in overall_bests.get(metric, set())
                if is_group_best:
                    value = rf"\underline{{{value}}}"
                if is_overall_best:
                    value = rf"\textbf{{{value}}}"
                if ci is None:
                    cells.append(value)
                else:
                    cells.append(f"{value} {format_ci_latex(ci, size=ci_size, color=ci_color)}")
            lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def render_rich_table(
    results: Sequence[ModelResult],
    *,
    group_bests: dict[str, dict[str, set[tuple[str, str]]]],
    overall_bests: dict[str, set[tuple[str, str]]],
) -> Table:
    by_group: dict[str, list[ModelResult]] = {}
    for result in results:
        by_group.setdefault(result.group, []).append(result)

    table = Table(
        title="TENDRIL retrieval results (Overall, 95% cluster bootstrap CI)",
        box=box.SIMPLE_HEAVY,
        header_style="bold",
        title_style="bold",
    )
    table.add_column("System", no_wrap=True)
    for metric in TABLE_METRICS:
        table.add_column(METRIC_HEADERS[metric], justify="right")

    present_groups = groups_in_order(results)
    for group_index, group in enumerate(present_groups):
        table.add_row(f"[italic dim]{group}[/italic dim]", *([""] * len(TABLE_METRICS)))
        members = by_group[group]
        for member_index, result in enumerate(members):
            cells = []
            for metric in TABLE_METRICS:
                value, ci = format_parts(result, metric)
                is_group_best = row_id(result) in group_bests.get(group, {}).get(metric, set())
                is_overall_best = row_id(result) in overall_bests.get(metric, set())
                if is_group_best:
                    value = f"[underline]{value}[/underline]"
                if is_overall_best:
                    value = f"[bold cyan]{value}[/bold cyan]"
                cells.append(value if ci is None else f"{value} {ci}")
            end_section = member_index == len(members) - 1 and group_index != len(present_groups) - 1
            table.add_row(result.display_name, *cells, end_section=end_section)

    return table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a hierarchical results table (LaTeX + rich) from evaluation JSON"
    )
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=DEFAULT_RESULTS_DIR,
        help=(
            f"Results directory holding one subdirectory of per-model score JSON files "
            f"per table group: {', '.join(RESULT_DIRECTORIES)} (default: {DEFAULT_RESULTS_DIR})."
        ),
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=None,
        help=(f"Path to write the LaTeX table to (default: <directory>/{DEFAULT_LATEX_STEM})"),
    )
    parser.add_argument(
        "--no-latex-file",
        action="store_true",
        help="Do not write the LaTeX table to disk, only print it",
    )
    parser.add_argument(
        "--ci-size",
        default=DEFAULT_CI_SIZE,
        help=(
            "LaTeX font-size command for confidence intervals, without the backslash "
            f"(default: {DEFAULT_CI_SIZE}). Pass an empty string to leave the size unchanged."
        ),
    )
    parser.add_argument(
        "--ci-color",
        default=DEFAULT_CI_COLOR,
        help=(
            "LaTeX color name for confidence intervals "
            f"(default: {DEFAULT_CI_COLOR}). Pass an empty string to leave the color unchanged."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    results = load_results(args.directory)
    if not results:
        raise SystemExit(f"ERROR: no usable result files found in {args.directory}")

    group_bests = group_best(results)
    overall_bests = overall_best(results)

    console.print(render_rich_table(results, group_bests=group_bests, overall_bests=overall_bests))

    latex_source = render_latex(
        results,
        group_bests=group_bests,
        overall_bests=overall_bests,
        ci_size=args.ci_size,
        ci_color=args.ci_color,
    )
    console.print()
    console.print(Syntax(latex_source, "latex", theme="ansi_dark", word_wrap=True))

    if not args.no_latex_file:
        output_path = args.latex_output or (args.directory / DEFAULT_LATEX_STEM)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex_source + "\n", encoding="utf-8")
        console.print(f"\nWrote LaTeX table to [bold]{output_path}[/bold]")


if __name__ == "__main__":
    main()
