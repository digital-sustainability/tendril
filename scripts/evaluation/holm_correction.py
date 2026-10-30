"""Holm correction of paired bootstrap p-values over a family of model comparisons.

Reads the JSON files written by ``paired_bootstrap.py --output`` and adjusts the
two-sided p-values with the Holm step-down procedure. By default, each metric is
its own family (e.g. "the four fine-tuning comparisons for nDCG@10");
``--family all`` treats every comparison of every metric as one family.

A bootstrap p-value of 0 means that no iteration crossed zero; the true two-sided
value is below 2 / iterations. It enters the correction as that bound, and
adjusted values derived from it are printed as upper bounds ("<").

Example::

    python scripts/evaluation/holm_correction.py results/bootstrap/*__TENDRIL.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Test:
    comparison: str
    metric: str
    difference: float
    p: float
    bound: bool  # True if p is only an upper bound (no iteration crossed zero)


@dataclass(frozen=True)
class Adjusted:
    test: Test
    p_holm: float
    bound: bool


def comparison_name(payload: dict) -> str:
    return f"{Path(payload['system_a']).name} -> {Path(payload['system_b']).name}"


def load_tests(paths: Sequence[Path]) -> list[Test]:
    tests = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        # The two-sided p-value doubles the smaller tail, so its bound is 2 / iterations.
        resolution = 2 / payload["iterations"]
        for item in payload["comparisons"]:
            p = item["p_two_sided"]
            tests.append(
                Test(
                    comparison=comparison_name(payload),
                    metric=item["metric"],
                    difference=item["difference"],
                    p=max(p, resolution),
                    bound=p == 0,
                )
            )
    return tests


def holm(tests: Sequence[Test]) -> list[Adjusted]:
    """Holm step-down: multiply the i-th smallest of m p-values by m - i and keep the running maximum."""
    ordered = sorted(tests, key=lambda test: test.p)
    adjusted = []
    running, running_bound = 0.0, False
    for index, test in enumerate(ordered):
        value = min(1.0, (len(ordered) - index) * test.p)
        if value > running:
            running, running_bound = value, test.bound
        adjusted.append(Adjusted(test, running, running_bound))
    return adjusted


def correct(tests: Sequence[Test], family: str) -> list[Adjusted]:
    groups: dict[str, list[Test]] = defaultdict(list)
    for test in tests:
        groups[test.metric if family == "metric" else "all"].append(test)
    return [item for group in groups.values() for item in holm(group)]


def format_p(value: float, bound: bool) -> str:
    return f"<{value:.4f}" if bound else f"{value:.4f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Holm-correct paired bootstrap p-values over several comparisons")
    parser.add_argument("results", nargs="+", type=Path, help="JSON files written by paired_bootstrap.py --output")
    parser.add_argument(
        "--family",
        choices=("metric", "all"),
        default="metric",
        help="Correct within each metric (default) or over all comparisons and metrics at once",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for the last column")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON file for the adjusted p-values")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    tests = load_tests(args.results)
    comparisons = len({test.comparison for test in tests})
    if comparisons < 2:
        raise SystemExit("ERROR: a correction needs at least two comparisons")
    adjusted = correct(tests, args.family)
    family = "per metric" if args.family == "metric" else "over all metrics"
    print(f"Holm correction {family}: {comparisons} comparisons, {len(tests)} tests, alpha={args.alpha}")
    width = max(len(test.comparison) for test in tests)
    print(f"{'metric':<16}{'comparison':<{width + 2}}{'B - A':>9}{'p':>10}{'p (Holm)':>11}  significant")
    for item in sorted(adjusted, key=lambda item: (item.test.metric, item.test.comparison)):
        test = item.test
        print(
            f"{test.metric:<16}{test.comparison:<{width + 2}}{test.difference:>+9.4f}"
            f"{format_p(test.p, test.bound):>10}{format_p(item.p_holm, item.bound):>11}  "
            f"{'yes' if item.p_holm < args.alpha else 'no'}"
        )
    if args.output is not None:
        payload = {
            "family": args.family,
            "alpha": args.alpha,
            "results": [str(path) for path in args.results],
            "tests": [
                {
                    "comparison": item.test.comparison,
                    "metric": item.test.metric,
                    "difference": item.test.difference,
                    "p": item.test.p,
                    "p_holm": item.p_holm,
                    "upper_bound": item.bound,
                    "significant": item.p_holm < args.alpha,
                }
                for item in adjusted
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
