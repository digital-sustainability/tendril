"""Evaluate any Sentence Transformers model on TENDRIL with MTEB.

Loads every TENDRIL project from the Hugging Face Hub as one MTEB retrieval task,
evaluates the model, and prints nDCG@10 per project and the mean over projects,
which is the score reported in the paper. TENDRIL is gated: accept its terms on
the Hub and log in (``hf auth login``) before running this.

Example::

    uv run python examples/evaluate_tendril.py sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    uv run python examples/evaluate_tendril.py path/to/fine-tuned/model --batch-size 8
"""

from __future__ import annotations

import argparse
from statistics import mean

import mteb
from sentence_transformers import SentenceTransformer

from tendril.hub import TEST_REPO
from tendril.tasks import load_hub_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", help="Hugging Face model id or local directory of a Sentence Transformers model")
    parser.add_argument("--revision", default="main", help="Revision of the TENDRIL dataset on the Hub (default: main)")
    parser.add_argument("--batch-size", type=int, default=32, help="Passages per forward pass (default: 32)")
    args = parser.parse_args()

    tasks = load_hub_tasks(TEST_REPO, revision=args.revision)
    model = SentenceTransformer(args.model)
    results = mteb.evaluate(model, tasks=tasks, cache=None, encode_kwargs={"batch_size": args.batch_size})

    scores = {result.task_name.removeprefix("TENDRIL-"): result.get_score() for result in results.task_results}
    for project, score in sorted(scores.items()):
        print(f"{project:<55} nDCG@10 = {score:.3f}")
    print(f"{'Mean over ' + str(len(scores)) + ' projects':<55} nDCG@10 = {mean(scores.values()):.3f}")


if __name__ == "__main__":
    main()
