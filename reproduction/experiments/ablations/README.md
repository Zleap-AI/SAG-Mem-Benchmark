# Ablations

These scripts use the repository `src/` implementation and the shared paper chunker. Offline preparation makes no model requests; Reader/Judge execution uses the repository-root `.env`.

## Offline input preparation

```bash
python -m reproduction.experiments.ablations.prepare \
  --source reproduction/experiments/main/results/locomo/full/run.jsonl \
  --method sag-mem --dataset locomo --scope temporal \
  --output reproduction/experiments/ablations/prepared/sag-mem/locomo/temporal
```

For LongMemEval, use `longmemeval/full/run.jsonl`; repeat `--source` only when your run is split into batches. The default data root is `reproduction/data`; it must contain the exact input shards referenced by the manifests. For the full initial-pass control, use `--scope all --no-expansion --only-no-expansion --sources-root <retained-sources>`.

`raw`, `normalized` and `main_cached` preserve the same selected records and order. Their contexts respectively use original headers, normalized headers only, and normalized headers plus temporal explanations. Only `main_cached` retains the existing answer/grade. The initial-pass control reconstructs candidates from cached h0 scores, applies the original front-10 DPP rule and validates the complete source map against the shared chunker. Preparation makes zero model calls.

```bash
python -m reproduction.experiments.ablations.verify \
  reproduction/experiments/ablations/prepared/sag-mem/locomo/temporal
python -m reproduction.experiments.ablations.validate_three \
  --locomo-source reproduction/experiments/main/results/locomo/full/run.jsonl \
  --longmemeval-source reproduction/experiments/main/results/longmemeval/full/run.jsonl \
  --data-root reproduction/data
```

The three-input check uses a deterministic local scorer, never JEV. A moved native replay can provide `--lineage-map <json>` mapping its recorded source paths to local files; the source run IDs and hashes still have to match.

## Reader/Judge execution

Configure Qwen Reader/Judge in the repository-root `.env`, then run:

```bash
python -m reproduction.experiments.ablations.run --data-root reproduction/data
```

Executing `run` makes new Qwen Reader/Judge requests. The fixed route is Qwen3.6-35B-A3B, DeepInfra FP8, reasoning disabled. `OPENAI_API_KEY` is also accepted; use `OPENROUTER_BASE_URL`, `OPENAI_BASE_URL` or `--base-url` for a compatible endpoint. `--api-key-env` selects another credential variable. Keys are never put in manifests.

The controller waits for complete local main/Lite results. Defaults preserve `experiments/main/results/{dataset}/*/run.jsonl`, `experiments/lite/results/{dataset}/qwen/run.jsonl` and `ablations/runs/{method}/{dataset}/{condition}/run.jsonl`; roots are configurable. Maximum global concurrency is 20. Successful records and saved Reader answers are reused on restart. `STATUS.json` and `requests.jsonl` separate new charges from cached-main accuracy. Main retrieval and main answers are not called again.

Temporal controls contain 321 LoCoMo and 133 LongMemEval questions per method. The SAG-Mem initial-pass control covers all 1,986 + 500 questions.

## Message-count controls

The current default allows at most three complete messages per chunk. Use an explicit zero cap and a separate output/index directory to compare against packing limited only by the token budget:

```bash
python reproduction/run.py --method sag-mem --dataset locomo \
  --messages-per-chunk 0 \
  --index-root reproduction/experiments/ablations/uncapped_messages/indexes \
  --output reproduction/experiments/ablations/uncapped_messages/locomo/run.jsonl
```

This makes new JEV, Reader and Judge requests.
