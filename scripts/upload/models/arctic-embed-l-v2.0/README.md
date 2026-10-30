---
license: apache-2.0
base_model: Snowflake/snowflake-arctic-embed-l-v2.0
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

# TENDRIL Arctic-Embed-L v2.0

A fine-tuned version of [Snowflake/snowflake-arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0)
for retrieving sustainability-relevant passages from Swiss public procurement tenders. Queries describe areas of
action or criteria from our sustainability catalogs, which are based on Swiss and EU recommendations for sustainable
public procurement.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("IntelliProcure/TENDRIL-arctic-embed-l-v2.0")
queries = model.encode_query(["Regionale und saisonale Lebensmittel bevorzugen ..."])
passages = model.encode_document(["Die Anbieterin liefert vorwiegend Produkte aus der Region ..."])
scores = model.similarity(queries, passages)
```

`encode_query` adds the prompt `query: `; documents get no prompt. Embeddings use CLS pooling, are L2-normalized
and have 1,024 dimensions, and inputs are truncated at 8,192 tokens. Truncating to 256 dimensions (Matryoshka)
was supported by the base model but was not evaluated after fine-tuning.

## Results

Test set: [IntelliProcure/TENDRIL](https://huggingface.co/datasets/IntelliProcure/TENDRIL) (revision `15eef0bc`),
33 projects. Scores are macro-averaged over projects, and 95% intervals come from 10,000 project-level bootstrap
resamples.

| Model | nDCG@10 | Recall@10 | Recall@100 | MRR@10 |
|---|---|---|---|---|
| BM25 | 0.421 [0.381, 0.462] | 0.511 | 0.929 | 0.463 |
| snowflake-arctic-embed-l-v2.0 (base) | 0.547 [0.509, 0.585] | 0.676 | 0.966 | 0.573 |
| **TENDRIL Arctic-Embed-L v2.0** | **0.642 [0.596, 0.690]** | **0.778** | **0.990** | **0.652** |

Fine-tuning improves nDCG@10 by +0.095 [+0.063, +0.130] in a two-sided paired bootstrap over projects; the gain is
significant for all reported metrics after a Holm correction over the four fine-tuned models in the paper
(p < 0.01).

On the validation projects, balanced nDCG@10 (the mean over areas of action and criteria) rose from 0.620 to 0.823.

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
  batch size 32 (1 × 32 gradient accumulation), at most 2 epochs, bf16, maximum sequence length 8,192, seed 42.
  The model was evaluated every 56 steps, and the checkpoint with the best balanced validation nDCG@10 was kept.
- **Software.** sentence-transformers 6.0.1, transformers 5.17.0, PyTorch 2.6.0.

## Limitations

- **German-centred.** The model was trained on German queries and mostly German passages; behaviour on French or
  Italian tenders was not evaluated.
- **Noisy labels.** Training labels come from an LLM judge whose agreement with human annotators is moderate
  (Cohen's κ 0.48–0.53 on the 0/1/2 scale).
- **Not a decision tool.** Rankings indicate where a tender may address a sustainability topic. They are not an
  assessment of a tender or a bidder and must not be used for automated procurement decisions.

## License

Apache License 2.0, the same as the base model. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Citation

```bibtex
@misc{rolshoven2026tendril,
  title  = {{TENDRIL}: Benchmarking Sustainability-Focused Retrieval in Swiss Public Procurement},
  author = {Rolshoven, Luca and Probst, Sophie and Matoshi, Veton and Ellendorff, Tilia and Meili, Rahel and
            Hostettler, Sarah and Jann, Jeremy Austin and Jans, Meret and Tschumi, Pascal and St{\"u}rmer, Matthias},
  year   = {2026}
}
```
