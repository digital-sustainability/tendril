"""
This script is used to compute the inter-annotator agreement between our assesors
on both the area of action and the criterion level. It assumes the presence of an
Excel file with columns corresponding to the different assesors' relevance judgments.
"""

from itertools import combinations

import krippendorff
import numpy as np
import pandas as pd
from loguru import logger

from tendril.paths import REPOSITORY_PATH

df = pd.read_excel(REPOSITORY_PATH / "data" / "annotation_overview.xlsx")


# Find annotator IDs; column are named like this: annotator_(\d)_(area|criterion)_relevance
relevance_cols = [col for col in df.columns if "relevance" in col]
ANNOTATOR_IDS = sorted(set(int(col.split("_")[1]) for col in relevance_cols))
logger.info(f"Annotators found: {ANNOTATOR_IDS}")


# Strip explanatory strings from relevance labels, e.g. "0 – Nicht relevant" --> 0.
# Parsing errors are not relevance judgments, so they count as missing.
LABEL_MAP = {
    "0 – Nicht relevant": 0,
    "1 – Teilweise relevant": 1,
    "2 – Klar relevant": 2,
    "Nicht auswertbar (Parsing-Fehler)": np.nan,
}
VALUE_DOMAIN = [0, 1, 2]

df[relevance_cols] = df[relevance_cols].replace(LABEL_MAP)

# Fail loudly on labels that are not in LABEL_MAP (e.g. a different dash or a trailing space)
for col in relevance_cols:
    labels = df[col].dropna()
    if not (unmapped := labels[~labels.isin(VALUE_DOMAIN)]).empty:
        raise ValueError(f"Unmapped labels in {col}: {unmapped.unique()}")

reliability_data_area = df[[f"annotator_{aid}_area_relevance" for aid in ANNOTATOR_IDS]].to_numpy(dtype=float).T
reliability_data_criterion = df[[f"annotator_{aid}_criterion_relevance" for aid in ANNOTATOR_IDS]].to_numpy(dtype=float).T


def ordinal_alpha(reliability_data: np.ndarray) -> float:
    return krippendorff.alpha(
        reliability_data=reliability_data, level_of_measurement="ordinal", value_domain=VALUE_DOMAIN
    )


# Look first at agreement between pairs of assessors rather than all together
for i, j in combinations(range(len(ANNOTATOR_IDS)), 2):
    print(f"Agreement among assessors {ANNOTATOR_IDS[i]} and {ANNOTATOR_IDS[j]}:")
    print(f"-> Krippendorffs alpha for areas: {ordinal_alpha(reliability_data_area[[i, j]]):.3f}")
    print(f"-> Krippendorffs alpha for criteria: {ordinal_alpha(reliability_data_criterion[[i, j]]):.3f}")


print(f"Agreement among all {len(ANNOTATOR_IDS)} assessors:")
print(f"-> Krippendorffs alpha for areas: {ordinal_alpha(reliability_data_area):.3f}")
print(f"-> Krippendorffs alpha for criteria: {ordinal_alpha(reliability_data_criterion):.3f}")
