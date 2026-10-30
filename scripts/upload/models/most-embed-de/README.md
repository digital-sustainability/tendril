---
license: cc-by-nc-4.0
base_model: malteos/most-embed-de
library_name: sentence-transformers
pipeline_tag: sentence-similarity
language:
- de
datasets:
- IntelliProcure/TENDRIL-Train
tags:
- sentence-transformers
- feature-extraction
- retrieval
- public-procurement
- sustainability
---

# TENDRIL most-embed-de

A fine-tuned version of [malteos/most-embed-de](https://huggingface.co/malteos/most-embed-de) for retrieving
sustainability-relevant passages from Swiss public procurement tenders. Queries describe areas of action or criteria
from our sustainability catalogs, which are based on Swiss and EU recommendations for sustainable public procurement.

**Non-commercial use only.** The base model is licensed under CC-BY-NC-4.0, and so is this fine-tune.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("IntelliProcure/TENDRIL-most-embed-de")
queries = model.encode_query(["Regionale und saisonale Lebensmittel bevorzugen ..."])
passages = model.encode_document(["Die Anbieterin liefert vorwiegend Produkte aus der Region ..."])
scores = model.similarity(queries, passages)
```

`encode_query` adds the prompt `query: ` and `encode_document` adds `passage: `. Embeddings use mean pooling, are
L2-normalized and have 2,048 dimensions, and inputs are truncated at 32,768 tokens.

## Results

Test set: [IntelliProcure/TENDRIL](https://huggingface.co/datasets/IntelliProcure/TENDRIL) (revision `15eef0bc`),
33 projects. Scores are macro-averaged over projects, and 95% intervals come from 10,000 project-level bootstrap
resamples.

| Model | nDCG@10 | Recall@10 | Recall@100 | MRR@10 |
|---|---|---|---|---|
| BM25 | 0.421 [0.381, 0.462] | 0.511 | 0.929 | 0.463 |
| most-embed-de (base) | 0.640 [0.588, 0.692] | 0.760 | 0.980 | 0.664 |
| **TENDRIL most-embed-de** | **0.660 [0.613, 0.707]** | **0.787** | **0.987** | **0.673** |

The gain over the base model is small: +0.020 nDCG@10 [−0.003, +0.044] in a two-sided paired bootstrap over projects.
After a Holm correction over the four fine-tuned models in the paper, it is significant only for precision@10
(p < 0.05). With an nDCG@10 of 0.660, the model is on par with the best proprietary model evaluated in the paper
(Voyage 4 Large, 0.649). On the validation projects, the fine-tuned model reaches a balanced nDCG@10 (the mean
over areas of action and criteria) of 0.840.

## Training

- **Data.** Training examples were built from the LLM-judged
  [TENDRIL-Train](https://huggingface.co/datasets/IntelliProcure/TENDRIL-Train) data: a query, one relevant passage
  and three hard negatives from the same project. Area-of-action and criterion examples were balanced, giving 9,000
  of 13,390 available examples. Validation used held-out projects (63 for areas of action, 62 for criteria).
- **Loss.** InfoNCE over the positive and the three negatives, with the cosine similarity of L2-normalized embeddings
  and a similarity scale of 20. Positives are passages the judge rated relevant (2); negatives are passages rated
  irrelevant (0) for the same query, two from the positive's document and one from another document of the same
  project (the other pool fills in when one has too few).
- **Hyperparameters.** Learning rate 1e-5 with linear decay and 10% warmup, AdamW without weight decay, effective
  batch size 32 (2 × 16 gradient accumulation), at most 2 epochs, bf16, seed 42. The model was evaluated every
  56 steps, and the checkpoint with the best balanced validation nDCG@10 (step 168) was kept.
- **Software.** sentence-transformers 6.0.1, transformers 5.17.0, PyTorch 2.6.0.

## Limitations

- **German-centred.** The model was trained on German queries and mostly German passages; behaviour on French or
  Italian tenders was not evaluated.
- **Noisy labels.** Training labels come from an LLM judge whose agreement with human annotators is moderate
  (Cohen's κ 0.48–0.53 on the 0/1/2 scale).
- **Not a decision tool.** Rankings indicate where a tender may address a sustainability topic. They are not an
  assessment of a tender or a bidder and must not be used for automated procurement decisions.

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC-4.0), the same as the base model. The
underlying Nemotron-3-Embed-1B is licensed under the OpenMDW License v1.1. See [LICENSE](LICENSE) and
[NOTICE](NOTICE) for the upstream attributions.

## Citation

```bibtex
@misc{rolshoven2026tendril,
  title  = {{TENDRIL}: Benchmarking Sustainability-Focused Retrieval in Swiss Public Procurement},
  author = {Rolshoven, Luca and Probst, Sophie and Matoshi, Veton and Ellendorff, Tilia and Meili, Rahel and
            Hostettler, Sarah and Jann, Jeremy Austin and Jans, Meret and Tschumi, Pascal and St{\"u}rmer, Matthias},
  year   = {2026}
}
```
