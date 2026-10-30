# Scripts related to annotations

## Introduction

This directory includes scripts that were used to analyze annotations and compute inter-annotator agreement. To keep this repository lean we do not include the annotation exports but rather postprocessed variants that are easier to use. The scripts that were used to produce these variants are included for transparency. 

The curated calibration sample (`validation_set_curated.xlsx`) is not part of this repository, because its passages are not anonymized. 

If you want to know how we created this dataset, please read the section below.

## How the scripts in this directory were used

### Curation of relevance judgments from Label Studio

First, we exported the annotations for our 1465 samples from label studio into `gold.json`. This file was then processed using the `create_annotation_overview.py` script, resulting in `data/annotation_overview.xlsx`. We then imported this file in [Grist Desktop](https://www.getgrist.com/) due to its card view feature that facilitated curation and made it easy to perform quick analyses. Whenever no majority vote was present for the area of action or criterion relevance label, we discussed the sample and noted our reconciled judgments in columns `reconciled_area_relevance` and `reconciled_criterion_relevance`. Finally, we converted our final reconciled judgments into integer labels in columns `int_rel_area` and `int_rel_criterion`. We then exported the Grist database as another Excel file `validation_set_curated.xlsx` and removed unnecessary columns.

### Inter-Annotator Agreement (Validation Set)

For our validation set, we computed Krippendorff's alpha with `compute_agreement_validation_set.py`. It reads the individual annotators' labels from `data/annotation_overview.xlsx` (created above), not the curated labels.

### Inter-Annotator Agreement (Test Set)

For the test set, `compute_agreement_test_set.py` reads the INCEpTION project export and computes document- and
project-level Krippendorff's alpha and Cohen's kappa, and the concept-level gamma, per annotator pair and per sector
(the module docstring describes the units and filters). It runs in two stages, so the large export is parsed only once:

```bash
uv run python scripts/annotation/compute_agreement_test_set.py extract --export <INCEpTION export.zip> --output data/swissspark_iaa
uv run python scripts/annotation/compute_agreement_test_set.py iaa --input data/swissspark_iaa --output data/swissspark_iaa
```

The test collection itself is built by the scripts in [`scripts/test_set`](../test_set/README.md).