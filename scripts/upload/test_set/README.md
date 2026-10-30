---
pretty_name: TENDRIL
language:
- de
- fr
- en
license: other
license_name: openrail-d
license_link: LICENSE.md
task_categories:
- text-retrieval
task_ids:
- document-retrieval
annotations_creators:
- expert-generated
source_datasets:
- original
size_categories:
- 10K<n<100K
tags:
- public-procurement
- sustainability
- switzerland
- mteb
- retrieval
extra_gated_prompt: >-
  TENDRIL is released under the TENDRIL Open RAIL-D License (see LICENSE.md). Its use-based restrictions
  (Attachment A) include a ban on extracting personal data or re-identifying persons, and a duty to disclose
  training on the test collection. They must be passed on to anyone who receives the data or a derivative.
extra_gated_fields:
  I agree to the TENDRIL Open RAIL-D License and its use-based restrictions: checkbox
extra_gated_button_content: Agree and access the dataset
---

# TENDRIL: Tender Retrieval Benchmark

TENDRIL is a German retrieval benchmark for finding sustainability-relevant passages in Swiss public procurement
tenders. Each query describes an *area of action* (Handlungsfeld) from our sustainability catalogs, which are based
on Swiss and EU recommendations for sustainable public procurement. Within one tender, the task is to retrieve the
passages that address that area. The query text is a German summary of the
criteria of the area of action, generated with DeepSeek V4 Flash, because the original area texts are often short.

This repository contains the **human-annotated test set**. The LLM-judged training data is published separately as
[IntelliProcure/TENDRIL-Train](https://huggingface.co/datasets/IntelliProcure/TENDRIL-Train).

## Dataset structure

The benchmark is split into one retrieval task per tender project. Each project has three configs in MTEB
retrieval format, all with the split `test`:

| Config | Columns | Content |
|---|---|---|
| `<project>-corpus` | `id`, `text` | Passages from the project's tender documents |
| `<project>-queries` | `id`, `text` | Areas of action annotated in at least one of the project's documents |
| `<project>-qrels` | `query-id`, `corpus-id`, `score` | Binary relevance (1 = relevant, 0 = not relevant); unjudged pairs count as not relevant |

Project names are anonymized aliases of the form `<sector>-<hash>`, for example `Food-60cd3a`. The hash is a keyed
hash of the project and stays the same across releases. Query IDs have the form `<catalog>::HF-<nnn>`.
`mteb_dataset.json` lists all projects.

| | Count |
|---|---|
| Projects (tasks) | 33 |
| Sectors | 14 |
| Documents | 139 |
| Project-query pairs | 469 |
| Distinct queries | 208 |
| Passages | 4,845 (1,557 relevant spans, 3,288 other passages) |
| Qrels | 66,771 (1,870 relevant) |

Projects per sector: Construction_Area 4, Construction_Buildings 3, Construction_Civil_Engineering_Infrastructure 2,
Electricity 4, Food 4, Furniture 2, Green_Spaces 2, Info_Com_Tech 2, Light_Elect_Appl 2, Paper_and_Stationery 3,
Print_Services 1, Road_Transport 2, Textiles 1, Wastepaper_Cleaning_Refuse 1.

## Usage

Each project is an MTEB retrieval task. Results are reported as the mean over all projects, with each project
counting once.

```python
from mteb.abstasks.retrieval_dataset_loaders import RetrievalDatasetLoader

split = RetrievalDatasetLoader(
    hf_repo="IntelliProcure/TENDRIL", revision="main", split="test", config="Food-60cd3a"
).load()
```

To load a single component with `datasets`:

```python
from datasets import load_dataset

corpus = load_dataset("IntelliProcure/TENDRIL", "Food-60cd3a-corpus", split="test")
```

## Dataset creation

**Source documents.** Tender documents of Swiss public procurement procedures published on
[simap.ch](https://www.simap.ch). Passages were extracted from the PDF and Office attachments with the IntelliProcure
document pipeline. Most documents are German; some are French or English.

**Annotation.** Seven annotators with expertise in sustainability or public procurement marked text spans in the
tender documents that address an area of action or one of its criteria, using
[INCEpTION](https://inception-project.github.io/) and written guidelines. Each catalog was annotated by at least two
annotators. Two annotators then discussed their disagreements and agreed on a curated version, which is used as the
gold standard. Only documents whose curation is marked as finished are included. Criterion annotations
are mapped to their parent area of action. Spans that overlap or lie within five characters of each other are merged.

**Corpus construction.** Every curated span becomes a relevant passage for its areas of action. All other pipeline
passages of the same documents that do not overlap a relevant span are added as non-relevant passages. Documents
without any curated span are left out, because an empty document does not necessarily indicate the absence of
relevant content, for example when it was out of scope or a duplicate of a document annotated in another project.
Every query of a project is judged against every non-relevant passage of that project, while relevant spans are only
judged for the areas of action they were annotated with. Pairs without a qrel count as not relevant, as in standard
MTEB evaluation.

**Inter-annotator agreement.** Agreement was computed per annotator pair on the documents both annotated, before
curation. Passages were matched one-to-one at an intersection over union of at least 0.1, and the sets of areas of
action of matched passages were compared. Krippendorff's α with the MASI distance has a median of about 0.48 over 11
pairs, and 0.44–0.56 for the five pairs with the most matched passages. Agreement is moderate, as expected for long
documents with many partly overlapping areas of action.

**Restricted documents.** Projects whose tender documents limit their use to the preparation of an offer, or
prohibit publication or disclosure to third parties, were identified with keyword rules plus an LLM check, and
excluded. Of the 46 annotated projects, 13 were removed for this reason.

**Personal data.** Passages were anonymized with
[gheim](https://huggingface.co/joelbarmettler/gheim-ch-560m): person names, e-mail addresses, phone numbers, URLs,
account numbers and credentials are replaced by placeholders such as `<PERSON_1>`. Placeholders are consistent within a
document. Dates and postal addresses are kept. Automatic anonymization can miss entities; please
report any personal data you find.

**Project identity.** Project names are aliases. The mapping to SIMAP project numbers is not published.

**Access and versions.** The dataset is gated: access is granted automatically after accepting the license terms.
Passages are removed on request, for example when a person reports remaining personal data. Each release is marked
with a git tag, so that earlier versions of the benchmark stay available; pin a tag or commit with `revision=`.

## Limitations

- **Small sectors.** Several sectors have only one or two projects, so per-sector scores have wide confidence
  intervals.
- **Binary labels.** Relevance is binary, and passages are spans of variable length, so relevant spans can be much
  shorter than pipeline passages.
- **Swiss context.** Queries use Swiss German spelling conventions and Swiss procurement terminology.

## License

The annotations, queries, selection and anonymization (the Licensed Contributions) are released under the
[TENDRIL Open RAIL-D License](LICENSE.md), which permits commercial use but adds use-based restrictions, such as a ban
on re-identifying persons and a duty to disclose training on the test data. The tender documents themselves remain the
property of their authors; see the license for details.

## Citation

```bibtex
@misc{rolshoven2026tendril,
  title  = {{TENDRIL}: Benchmarking Sustainability-Focused Retrieval in Swiss Public Procurement},
  author = {Rolshoven, Luca and Probst, Sophie and Matoshi, Veton and Ellendorff, Tilia and Meili, Rahel and
            Hostettler, Sarah and Jann, Jeremy Austin and Jans, Meret and Tschumi, Pascal and St{\"u}rmer, Matthias},
  year   = {2026}
}
```
