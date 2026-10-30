# Test set (TENDRIL)

Scripts that turn the curated INCEpTION export into the TENDRIL test collection. Run all commands from the
repository root. The inputs are not part of the repository: the INCEpTION project export, the parsed tender
documents (`simap_<id>.json.zst` pipeline outputs), and the private files under `private/` and `data/exports/`.

## Pipeline

1. **Find projects with reuse restrictions** (`find_restricted_projects.py`). Regex candidates in German, French and
   Italian, checked by an LLM judge via OpenRouter (`OPENROUTER_API_KEY`). `--skip-llm` lists the regex candidates
   only. The list is private and only used to exclude projects.

   ```bash
   uv run python scripts/test_set/find_restricted_projects.py <pipeline dir> \
       --output data/exports/restrictions/test_projects.txt \
       --details data/exports/restrictions/test_projects_details.jsonl
   ```

2. **Build the collection** (`build_dataset.py`). One BEIR shard per project with `corpus.jsonl`, `queries.jsonl` and
   `qrels/test.tsv`. The defaults are the settings of the published version:
   - only documents whose curation is finished (`--finished-only`);
   - documents without any curated span are left out (`--no-empty-negatives`);
   - queries are the LLM summaries of the areas of action (`data/queries/llm_summary.jsonl`);
   - projects listed in `data/exports/restrictions/test_projects.txt` are excluded;
   - project aliases are `<sector>-<6 hex>`, an HMAC of the SIMAP ID keyed with `private/test_alias_key`. Keep that
     key: without it, aliases cannot be reproduced. The SIMAP-to-alias mapping is written to
     `private/simap_project_aliases.json`.

   ```bash
   uv run python scripts/test_set/build_dataset.py --export <INCEpTION export.zip> \
       --pipeline <pipeline dir> --output data/exports/tendril_test
   ```

3. **Anonymize the passages** (`scripts/anonymize/anonymize_collections.py`). Writes `data/exports/tendril_test_anon`;
   queries and qrels are copied unchanged.

   ```bash
   uv run python scripts/anonymize/anonymize_collections.py data/exports/tendril_test
   ```

4. **Upload** with `scripts/upload/upload.py test` (see [the upload README](../upload/README.md)). Before uploading,
   compare the staged files of a `--dry-run` with the current Hub revision and check that only the intended files
   change.

## Tests

```bash
uv run pytest scripts/test_set
```
