"""Build judge-agreement tables (LaTeX + terminal) from labeling runs.

The paired table reads ``model_metrics.csv`` and ``manifest.json`` from the
original-wording run and the LLM-summary run. Each metric is two columns:
the original area wording, then the LLM-summary area wording. The best value
in each column is bold (highest kappa, lowest cost).

The prompt-length table is GLM-5.3-Flash at low reasoning effort on the
LLM-summary area wording: the full guidelines prompt from the summary run,
then the shorter and shortest prompts. The best value in each column is bold.

Means match ``analysis.best_model_ranking``: the mean of ``kappa_binary`` and
``kappa_graded`` across area and criterion, and ``total_cost`` summed across
those targets. Rows with ``n == 0`` are excluded.

Example::

    python scripts/render/tables/build_judge_results_table.py
    python scripts/render/tables/build_judge_results_table.py \\
        --latex-output data/judge/model_reasoning/judge_agreement_table.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from rich import box
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from latex_escape import latex_escape

console = Console()

DEFAULT_ORIGINAL = Path("data/judge/model_reasoning/simple_aoa")
DEFAULT_SUMMARY = Path("data/judge/model_reasoning/summary_aoa")
DEFAULT_SHORTER = Path("data/judge/prompt_length/shorter")
DEFAULT_SHORTEST = Path("data/judge/prompt_length/shortest")
DEFAULT_LATEX_NAME = "judge_agreement_table.tex"
DEFAULT_PROMPT_LATEX_NAME = "judge_prompt_length_table.tex"

PROMPT_MODEL = "z-ai/glm-5.3-flash"
PROMPT_EFFORT = "low"
PROMPT_VARIANTS = ("original", "shorter", "shortest")

TITLE = (
    "LLM-as-a-judge agreement. "
    "Original is the original area wording; Summary is the LLM-summary wording. "
    "Bold is the best value in each column (highest agreement, lowest cost)."
)
PROMPT_TITLE = (
    "GLM-5.3-Flash, low reasoning effort, LLM-summary area wording. "
    "Bold is the best value in each column (highest agreement, lowest cost)."
)

WORDING_SIDES = ("Original", "Summary")
LATEX_WORDING_SIDES = ("Orig.", "Summ.")
LATEX_EFFORTS = {"medium": "med."}

LATEX_COMPACT_OPEN = (
    r"{\scriptsize",
    r"\renewcommand{\arraystretch}{0.9}",
    r"\setlength{\tabcolsep}{4pt}",
)
LATEX_COMPACT_CLOSE = "}"

DISPLAY_NAMES = {
    "qwen/qwen3.7-flash": "Qwen3.7-Flash",
    "qwen/qwen3.8-flash": "Qwen3.8-Flash",
    "deepseek/deepseek-v4-flash-0731": "DeepSeek-V4-Flash-0731",
    "deepseek/deepseek-v4.1-flash": "DeepSeek-V4.1-Flash",
    "z-ai/glm-5.3-flash": "GLM-5.3-Flash",
    "google/gemini-2.5-flash-lite": "Gemini-2.5-Flash-Lite",
    "openai/gpt-6-luna": "GPT-6-Luna",
    "mistralai/mistral-small-2603": "Mistral-Small-2603",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    effort: str


@dataclass(frozen=True)
class SideMetrics:
    mean_kappa_binary: float | None
    mean_kappa_graded: float | None
    total_cost: float | None


@dataclass(frozen=True)
class MetricColumn:
    field: str
    latex_header: str
    rich_header: str
    number_format: str
    higher_is_better: bool


METRICS = (
    MetricColumn("mean_kappa_binary", r"$\kappa$ binary", "κ binary", ".3f", True),
    MetricColumn("mean_kappa_graded", r"$\kappa$ graded", "κ graded", ".3f", True),
    MetricColumn("total_cost", "Cost (USD)", "Cost (USD)", ".2f", False),
)

PairedRow = tuple[ModelSpec, SideMetrics | None, SideMetrics | None]


def finite(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    number = float(text)
    if not math.isfinite(number):
        return None
    return number


def mean_finite(values: Sequence[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return mean(usable)


def sum_finite(values: Sequence[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return sum(usable)


def load_manifest_models(directory: Path) -> list[ModelSpec]:
    path = directory / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models")
    if not isinstance(models, list):
        raise SystemExit(f"ERROR: {path} has no 'models' list")
    specs: list[ModelSpec] = []
    for entry in models:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("name"), str):
            raise SystemExit(f"ERROR: {path} has a model entry without a name")
        reasoning = entry.get("reasoning")
        effort = ""
        if isinstance(reasoning, Mapping) and reasoning.get("effort") is not None:
            effort = str(reasoning["effort"])
        display = entry.get("model")
        specs.append(
            ModelSpec(
                name=entry["name"],
                display_name=display if isinstance(display, str) and display else entry["name"],
                effort=effort,
            )
        )
    return specs


def load_side_metrics(directory: Path) -> dict[str, SideMetrics]:
    path = directory / "model_metrics.csv"
    if not path.is_file():
        raise SystemExit(f"ERROR: {path} does not exist")
    grouped: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            n = finite(row.get("n", ""))
            if n is None or n <= 0:
                continue
            grouped.setdefault(row["model"], []).append(row)

    metrics: dict[str, SideMetrics] = {}
    for model, rows in grouped.items():
        metrics[model] = SideMetrics(
            mean_kappa_binary=mean_finite([finite(row.get("kappa_binary", "")) for row in rows]),
            mean_kappa_graded=mean_finite([finite(row.get("kappa_graded", "")) for row in rows]),
            total_cost=sum_finite([finite(row.get("total_cost", "")) for row in rows]),
        )
    return metrics


def paired_models(
    original_models: Sequence[ModelSpec],
    summary_models: Sequence[ModelSpec],
    original_metrics: Mapping[str, SideMetrics],
    summary_metrics: Mapping[str, SideMetrics],
) -> list[ModelSpec]:
    ordered: list[ModelSpec] = []
    seen: set[str] = set()
    for spec in (*original_models, *summary_models):
        if spec.name in seen:
            continue
        seen.add(spec.name)
        ordered.append(spec)
    for name in sorted(set(original_metrics) | set(summary_metrics)):
        if name in seen:
            continue
        console.print(
            f"[yellow]{name} is in model_metrics.csv but not in either manifest[/yellow]"
        )
        ordered.append(ModelSpec(name=name, display_name=name, effort=""))
    return ordered


def format_value(value: float | None, spec: str) -> str:
    if value is None:
        return "--"
    return format(value, spec)


def side_value(metrics: SideMetrics | None, field: str) -> float | None:
    if metrics is None:
        return None
    return getattr(metrics, field)


def series_is_best(values: Sequence[float | None], metric: MetricColumn) -> list[bool]:
    """Whether each displayed value is best in the column (ties included)."""
    shown = [format_value(value, metric.number_format) for value in values]
    ranked = [float(text) for text in shown if text != "--"]
    if not ranked:
        return [False] * len(values)
    target = max(ranked) if metric.higher_is_better else min(ranked)
    return [text != "--" and float(text) == target for text in shown]


def emphasize(text: str, best: bool, *, latex: bool) -> str:
    if not best:
        return text
    if latex:
        return rf"\textbf{{{text}}}"
    return f"[bold green]{text}[/bold green]"


def paired_is_best(rows: Sequence[PairedRow]) -> list[list[tuple[bool, bool]]]:
    """Per row, per metric: whether the original and summary cells are best."""
    by_metric = [
        (
            series_is_best(
                [side_value(original, metric.field) for _, original, _ in rows],
                metric,
            ),
            series_is_best(
                [side_value(summary, metric.field) for _, _, summary in rows],
                metric,
            ),
        )
        for metric in METRICS
    ]
    return [
        [(original[row_index], summary[row_index]) for original, summary in by_metric]
        for row_index in range(len(rows))
    ]


def prompt_is_best(rows: Sequence[tuple[str, SideMetrics]]) -> list[list[bool]]:
    """Per row, per metric: whether that cell is the best in its column."""
    by_metric = [
        series_is_best([side_value(metrics, metric.field) for _, metrics in rows], metric)
        for metric in METRICS
    ]
    return [
        [column[row_index] for column in by_metric] for row_index in range(len(rows))
    ]


def short_name(spec: ModelSpec) -> str:
    return DISPLAY_NAMES.get(spec.display_name, spec.display_name)


def render_latex(rows: Sequence[PairedRow]) -> str:
    """Tabular with each model named once and no rules between its effort rows."""
    flags = paired_is_best(rows)
    group_headers = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{metric.latex_header}}}" for metric in METRICS
    )
    rules = " ".join(
        rf"\cmidrule(lr){{{3 + 2 * index}-{4 + 2 * index}}}" for index in range(len(METRICS))
    )
    subheaders = " & ".join(
        f"{LATEX_WORDING_SIDES[0]} & {LATEX_WORDING_SIDES[1]}" for _ in METRICS
    )
    lines = [
        *LATEX_COMPACT_OPEN,
        r"\begin{tabular}{ll" + "cc" * len(METRICS) + "}",
        r"\toprule",
        f"& & {group_headers} " + r"\\",
        rules,
        f"Model & Eff. & {subheaders} " + r"\\",
        r"\midrule",
    ]
    previous_name: str | None = None
    for row_index, (spec, original, summary) in enumerate(rows):
        name = short_name(spec)
        if name == previous_name:
            name_cell = ""
        else:
            if previous_name is not None:
                lines.append(r"\addlinespace[1pt]")
            name_cell = latex_escape(name)
        previous_name = name
        cells = [name_cell, latex_escape(LATEX_EFFORTS.get(spec.effort, spec.effort))]
        for metric_index, metric in enumerate(METRICS):
            original_best, summary_best = flags[row_index][metric_index]
            for metrics, best in ((original, original_best), (summary, summary_best)):
                cells.append(
                    emphasize(
                        format_value(side_value(metrics, metric.field), metric.number_format),
                        best,
                        latex=True,
                    )
                )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", LATEX_COMPACT_CLOSE])
    return "\n".join(lines)


def render_rich_table(rows: Sequence[PairedRow]) -> Table:
    flags = paired_is_best(rows)
    table = Table(
        title=TITLE,
        box=box.SIMPLE_HEAVY,
        header_style="bold",
        title_style="bold",
    )
    table.add_column("Model", no_wrap=True)
    table.add_column("Effort", no_wrap=True)
    for metric in METRICS:
        table.add_column(f"{metric.rich_header}\n{WORDING_SIDES[0]}", justify="right", no_wrap=True)
        table.add_column(f"\n{WORDING_SIDES[1]}", justify="right", no_wrap=True)
    for row_index, (spec, original, summary) in enumerate(rows):
        cells: list[str] = []
        for metric_index, metric in enumerate(METRICS):
            original_best, summary_best = flags[row_index][metric_index]
            for metrics, best in ((original, original_best), (summary, summary_best)):
                cells.append(
                    emphasize(
                        format_value(side_value(metrics, metric.field), metric.number_format),
                        best,
                        latex=False,
                    )
                )
        table.add_row(short_name(spec), spec.effort, *cells)
    return table


def build_rows(
    original_dir: Path,
    summary_dir: Path,
) -> list[tuple[ModelSpec, SideMetrics | None, SideMetrics | None]]:
    if not original_dir.is_dir():
        raise SystemExit(f"ERROR: {original_dir} is not a directory")
    if not summary_dir.is_dir():
        raise SystemExit(f"ERROR: {summary_dir} is not a directory")
    original_metrics = load_side_metrics(original_dir)
    summary_metrics = load_side_metrics(summary_dir)
    specs = paired_models(
        load_manifest_models(original_dir),
        load_manifest_models(summary_dir),
        original_metrics,
        summary_metrics,
    )
    return [
        (spec, original_metrics.get(spec.name), summary_metrics.get(spec.name))
        for spec in specs
    ]


def glm_low_metrics(directory: Path) -> SideMetrics:
    """Metrics for GLM-5.3-Flash at low effort in one results directory."""
    if not directory.is_dir():
        raise SystemExit(f"ERROR: {directory} is not a directory")
    matches = [
        spec
        for spec in load_manifest_models(directory)
        if spec.display_name == PROMPT_MODEL and spec.effort == PROMPT_EFFORT
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"ERROR: {directory} should contain exactly one {PROMPT_MODEL} "
            f"entry at effort {PROMPT_EFFORT!r}, found {len(matches)}"
        )
    metrics = load_side_metrics(directory).get(matches[0].name)
    if metrics is None:
        raise SystemExit(f"ERROR: {directory} has no usable rows for {matches[0].name}")
    return metrics


def build_prompt_rows(
    original_prompt_dir: Path,
    shorter_dir: Path,
    shortest_dir: Path,
) -> list[tuple[str, SideMetrics]]:
    directories = {
        "original": original_prompt_dir,
        "shorter": shorter_dir,
        "shortest": shortest_dir,
    }
    return [(variant, glm_low_metrics(directories[variant])) for variant in PROMPT_VARIANTS]


def render_prompt_latex(rows: Sequence[tuple[str, SideMetrics]]) -> str:
    flags = prompt_is_best(rows)
    lines = [
        *LATEX_COMPACT_OPEN,
        r"\begin{tabular}{l" + "c" * len(METRICS) + "}",
        r"\toprule",
        "Prompt variant & " + " & ".join(metric.latex_header for metric in METRICS) + r" \\",
        r"\midrule",
    ]
    for row_index, (variant, metrics) in enumerate(rows):
        cells = [latex_escape(variant)]
        for metric_index, metric in enumerate(METRICS):
            cells.append(
                emphasize(
                    format_value(side_value(metrics, metric.field), metric.number_format),
                    flags[row_index][metric_index],
                    latex=True,
                )
            )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", LATEX_COMPACT_CLOSE])
    return "\n".join(lines)


def render_prompt_rich_table(rows: Sequence[tuple[str, SideMetrics]]) -> Table:
    flags = prompt_is_best(rows)
    table = Table(
        title=PROMPT_TITLE,
        box=box.SIMPLE_HEAVY,
        header_style="bold",
        title_style="bold",
    )
    table.add_column("Prompt variant", no_wrap=True)
    for metric in METRICS:
        table.add_column(metric.rich_header, justify="right", no_wrap=True)
    for row_index, (variant, metrics) in enumerate(rows):
        cells = [
            emphasize(
                format_value(side_value(metrics, metric.field), metric.number_format),
                flags[row_index][metric_index],
                latex=False,
            )
            for metric_index, metric in enumerate(METRICS)
        ]
        table.add_row(variant, *cells)
    return table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the paired original-vs-summary judge table and the "
            "GLM-5.3-Flash prompt-length table"
        )
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=DEFAULT_ORIGINAL,
        help=f"Original-wording results directory (default: {DEFAULT_ORIGINAL})",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help=f"LLM-summary results directory (default: {DEFAULT_SUMMARY})",
    )
    parser.add_argument(
        "--shorter",
        type=Path,
        default=DEFAULT_SHORTER,
        help=f"Shorter-prompt results directory (default: {DEFAULT_SHORTER})",
    )
    parser.add_argument(
        "--shortest",
        type=Path,
        default=DEFAULT_SHORTEST,
        help=f"Shortest-prompt results directory (default: {DEFAULT_SHORTEST})",
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=None,
        help=(
            "Path to write the paired LaTeX table to (default: "
            f"<original>/../{DEFAULT_LATEX_NAME})"
        ),
    )
    parser.add_argument(
        "--prompt-latex-output",
        type=Path,
        default=None,
        help=(
            "Path to write the prompt-length LaTeX table to (default: "
            f"<shorter>/../{DEFAULT_PROMPT_LATEX_NAME})"
        ),
    )
    parser.add_argument(
        "--no-latex-file",
        action="store_true",
        help="Do not write the LaTeX tables to disk, only print them",
    )
    return parser


def write_latex(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source + "\n", encoding="utf-8")
    console.print(f"\nWrote LaTeX table to [bold]{path}[/bold]")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = build_rows(args.original, args.summary)
    if not rows:
        raise SystemExit("ERROR: no models found in the two manifests")
    prompt_rows = build_prompt_rows(args.summary, args.shorter, args.shortest)

    console.print(render_rich_table(rows))
    latex_source = render_latex(rows)
    console.print()
    console.print(Syntax(latex_source, "latex", theme="ansi_dark", word_wrap=True))

    console.print()
    console.print(render_prompt_rich_table(prompt_rows))
    prompt_latex = render_prompt_latex(prompt_rows)
    console.print()
    console.print(Syntax(prompt_latex, "latex", theme="ansi_dark", word_wrap=True))

    if not args.no_latex_file:
        output_path = args.latex_output or (args.original.parent / DEFAULT_LATEX_NAME)
        prompt_output = args.prompt_latex_output or (args.shorter.parent / DEFAULT_PROMPT_LATEX_NAME)
        write_latex(output_path, latex_source)
        write_latex(prompt_output, prompt_latex)


if __name__ == "__main__":
    main()
