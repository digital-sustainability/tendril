import json
from collections import Counter

import pandas as pd
from loguru import logger
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from tqdm import tqdm

from tendril.paths import REPOSITORY_PATH


def clean_string_for_excel(value: str) -> str:
    # Replace illegal characters
    clean = ILLEGAL_CHARACTERS_RE.sub("", value)

    # Texts that start with "=" are interpreted as formulas by Excel
    if clean.startswith("="):
        # Add apostrophe before equal sign
        clean = f"'{clean}"

    return clean


def majority_vote(judgments: list[str | None]) -> tuple[str | None, int]:
    """Return (value, count) of a strict majority among the non-missing votes."""
    votes = [j for j in judgments if j is not None]
    if not votes:
        return None, 0
    value, count = Counter(votes).most_common(1)[0]
    return (value, count) if count > len(votes) / 2 else (None, count)


DATA_DIR = REPOSITORY_PATH / "data"

with open(DATA_DIR / "exports" / "gold.json", encoding="utf-8", mode="r") as file:
    data = json.load(file)

logger.warning("Loaded annotations from gold.json. Make sure that the annotations are up-to-date and download a new export from Label Studio otherwise.")


ANNOTATOR_IDS = sorted({
    annotation["completed_by"]
    for item in data
    for annotation in item["annotations"]
})
logger.info(f"Annotators found: {ANNOTATOR_IDS}")

BASE_COLS = [
    "sample_id", "project_id", "wdg_id", "band", "project_indicator",
    "criterion_id", "criterion_text", "scope_of_action", "scope_of_action_id",
    "criterion_top_percentile", "rank", "cosine_score", "file_name",
    "sentence_cleaned", "context_before", "context_after",
]
ANNOTATOR_COLS = [
    f"annotator_{aid}_{field}"
    for aid in ANNOTATOR_IDS
    for field in ("area_relevance", "criterion_relevance", "note")
]
DERIVED_COLS = [
    "area_judgments", "criterion_judgments", "notes",
    "has_disagreement_area", "has_majority_vote_area",
    "has_disagreement_criterion", "has_majority_vote_criterion",
    "has_disagreement", "area_majority_vote", "criterion_majority_vote",
    "quality_check_needed",
]

rows = []

for item in tqdm(data, desc="Processing annotations", unit="task"):
    row_data = {
        "sample_id": item["id"],
        "project_id": item["data"]["project_id"],
        "wdg_id": item["data"]["wdg_id"],
        "band": item["data"]["band"],
        "project_indicator": item["data"]["project_indicator"],
        "criterion_id": item["data"]["criterion_id"],
        "criterion_text": item["data"]["criterion_text"],
        "scope_of_action": item["data"]["scope_of_action"],
        "scope_of_action_id": item["data"]["scope_of_action_id"],
        "criterion_top_percentile": item["data"]["criterion_top_percentile"],
        "rank": item["data"]["rank"],
        "cosine_score": item["data"]["cosine_score"],
        "file_name": item["data"]["file_name"],
        "sentence_cleaned": clean_string_for_excel(item["data"]["sentence_cleaned"]),
        "context_before": clean_string_for_excel(item["data"]["context_before"]),
        "context_after": clean_string_for_excel(item["data"]["context_after"]),
    }

    for annotation in item["annotations"]:
        annotator_id = annotation["completed_by"]

        for res in annotation["result"]:
            field = res["from_name"]
            selector = "choices" if field in ["area_relevance", "criterion_relevance"] else "text"

            row_data[f"annotator_{annotator_id}_{field}"] = res["value"][selector][0]
    
    area_judgments = [row_data.get(f"annotator_{aid}_area_relevance") for aid in ANNOTATOR_IDS]
    criterion_judgments = [row_data.get(f"annotator_{aid}_criterion_relevance") for aid in ANNOTATOR_IDS]
    notes = [
        f"[{aid}] {note}"
        for aid in ANNOTATOR_IDS
        if (note := row_data.get(f"annotator_{aid}_note"))
    ]
    
    area_majority_vote, num_votes_area = majority_vote(area_judgments)
    criterion_majority_vote, num_votes_criterion = majority_vote(criterion_judgments)
    has_majority_vote_area = area_majority_vote is not None
    has_majority_vote_criterion = criterion_majority_vote is not None

    has_disagreement_area = len({j for j in area_judgments if j is not None}) > 1
    has_disagreement_criterion = len({j for j in criterion_judgments if j is not None}) > 1

    quality_check_needed = "Nicht auswertbar (Parsing-Fehler)" in row_data.values()
    
    row_data.update({
        "area_judgments": area_judgments,
        "criterion_judgments": criterion_judgments,
        "notes": notes,
        "has_disagreement_area": has_disagreement_area,
        "has_majority_vote_area": has_majority_vote_area,
        "has_disagreement_criterion": has_disagreement_criterion,
        "has_majority_vote_criterion": has_majority_vote_criterion,
        "has_disagreement": has_disagreement_area or has_disagreement_criterion,
        "area_majority_vote": area_majority_vote,
        "criterion_majority_vote": criterion_majority_vote,
        "quality_check_needed": quality_check_needed
    })
    

    rows.append(row_data)


df = pd.DataFrame(rows)

if dropped := set(df.columns) - set(BASE_COLS + ANNOTATOR_COLS + DERIVED_COLS):
    raise ValueError(f"Columns not covered by the explicit order: {sorted(dropped)}")

df = df.reindex(columns=BASE_COLS + ANNOTATOR_COLS + DERIVED_COLS)

df.to_excel(DATA_DIR / "annotation_overview.xlsx", index=False)