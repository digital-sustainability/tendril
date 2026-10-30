# Hugging Face uploads

`upload.py` stages, validates and uploads the TENDRIL artifacts to the
[IntelliProcure](https://huggingface.co/IntelliProcure) organization. New repositories are created **private**.

| Key | Repository | Type | Card and license files |
|---|---|---|---|
| `test` | `IntelliProcure/TENDRIL` | dataset | `test_set/`, `licenses/OpenRAIL-D.md` |
| `train` | `IntelliProcure/TENDRIL-Train` | dataset | `train_set/`, `licenses/OpenRAIL-D.md` |
| `arctic` | `IntelliProcure/TENDRIL-arctic-embed-l-v2.0` | model | `models/arctic-embed-l-v2.0/` (Apache-2.0) |
| `most-embed-de` | `IntelliProcure/TENDRIL-most-embed-de` | model | `models/most-embed-de/` (CC-BY-NC-4.0) |

## Usage

Log in once with an account that can write to IntelliProcure (`hf auth login`), then:

```bash
# Stage and validate the test set without uploading
uv run python scripts/upload/upload.py test --dry-run

# Upload selected artifacts (private); the training set and the models need their local directories
uv run python scripts/upload/upload.py test
uv run python scripts/upload/upload.py train --train-source <TENDRIL-Train build> --test-pipeline <parsed test projects>
uv run python scripts/upload/upload.py arctic most-embed-de \
    --arctic-model <fine-tuned Arctic Embed> --most-embed-model <fine-tuned Most Embed>
```

Staged files go to `tmp/upload/<key>/` (override with `--staging`). Each upload is a single commit that mirrors the
staged folder, so files removed locally are also removed from the repository.

## Guards

- **Cards with `TODO`.** The upload is refused while a card still contains `TODO`. `--allow-todo` overrides this for
  private drafts only; it cannot be combined with `--public`.
- **Draft licenses.** `--public` is refused while a license file is marked `DRAFT` (the OpenRAIL-D text needs legal
  review first).
- **Forbidden files.** The upload is refused if anonymization replacement logs, request databases, backups or
  `training_args.bin` were staged.
- **Visibility.** Existing repositories keep their visibility. To publish one, change it on the Hub after review.

## What is staged

- **Test set.** Built from `data/exports/tendril_test_anon`, one set of MTEB configs per project (`<alias>-corpus`,
  `<alias>-queries`, `<alias>-qrels`). The staging step checks the shards against
  `private/simap_project_aliases.json` and the restriction list.
- **Training set.** Built from the TENDRIL-Train build directory (`--train-source`): only judged passages, with the
  restricted and test projects checked to be absent. Project IDs and file names are replaced by HMAC hashes keyed
  with `private/hash_key`. That key is created on first use; keep it, so the hashes stay stable across re-uploads.
- **Models.** Copied from the fine-tuned model directories (`--arctic-model`, `--most-embed-model`), without
  `training_args.bin`, and loaded once with sentence-transformers on the CPU as a check.
