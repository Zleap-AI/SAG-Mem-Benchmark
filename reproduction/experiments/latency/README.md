# Serial Warm-Index Latency

[Project overview](../../../README.md) · [Experiment guide](../../README.md)

Measure SAG-Mem and SAG-Mem-Lite on the same selected questions, one request at a time. Each trial records retrieval time and Reader generation time separately. Index construction and Judge calls are outside the timed interval.

## Prepare

Use the repository-root `.env` to configure Qwen Reader/Judge, JEV and embedding services. Complete the main runs first so both methods have prepared indexes.

The selection is fixed:

| Dataset | Questions | Selection |
|---|---:|---|
| LoCoMo | 100 | 10 per conversation, seed 20261118 |
| LongMemEval-S | 50 | Fixed category quotas, seed 20261105 |

## Run

From the repository root:

```bash
python -m reproduction.experiments.latency.run \
  --data-root reproduction/data --model-family qwen \
  --main-index-root reproduction/indexes \
  --lite-index-root reproduction/indexes \
  --lite-cache-root reproduction/indexes/cache
```

The command makes real retrieval and Reader requests. Add `--dataset locomo` or `--dataset longmemeval` to select one dataset, and `--output-root` for a separate output directory.

`--prepare-missing-lite-index` builds a missing Lite index before timed queries. Each timed Lite query uses a fresh question embedding; document vectors are reused. Method order alternates across questions.

## Outputs

| File | Content |
|---|---|
| `trials.jsonl` | Per-question method, selected evidence, retrieval time, Reader time and token usage |
| `requests.jsonl` | Model request accounting |
| `selection_<dataset>.json` | Selected questions and dataset identity |
| `environment_<dataset>.json` | Host, runtime and source identity |
| `STATUS.json` | Completed trials |

Defaults are under `reproduction/experiments/latency/`. Reuse the same host, model endpoints, indexes and configuration when comparing methods. Host load is recorded per trial.

## Verify

```bash
python -m pytest -q reproduction/experiments/latency
```

The tests check deterministic selection, query-cache isolation and native BGE dimensions. They do not call a model service.
