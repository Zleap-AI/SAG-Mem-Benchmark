"""Build publication inputs from complete, identified per-question results."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

if __package__:
    from .costs import response_usage, token_cost
else:
    from costs import response_usage, token_cost
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis"
DATA_ROOT = ROOT / "data"
EXPECTED = {"locomo": 1986, "longmemeval": 500}
CATEGORIES = {
    "locomo": {"1": "multi_hop", "2": "temporal", "3": "open_domain", "4": "single_hop", "5": "adversarial"},
    "longmemeval": {},
}
SEED = 20261105
REPLICATES = 20000


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=2)
def independent_questions(dataset):
    name = "locomo10.json" if dataset == "locomo" else "longmemeval_s_cleaned.json"
    path = DATA_ROOT / name
    plan = json.loads((ROOT / "experiments/main/run_plan.json").read_text())
    if digest(path) != plan["original_sources"][dataset]["sha256"]:
        raise ValueError("Source dataset checksum changed")
    data = json.loads(path.read_text())
    if dataset == "locomo":
        return {f"{item['sample_id']}:{i}": (q["question"], str(q["category"]),
                    "No information available" if q["category"] == 5 else str(q["answer"]),
                    q["category"] == 5, "")
                for item in data for i, q in enumerate(item["qa"])}
    return {item["question_id"]: (item["question"], item["question_type"], str(item["answer"]),
                                  item["question_id"].endswith("_abs"), str(item.get("question_date") or ""))
            for item in data}


def verify_manifest(manifest, path, method, model):
    identity = manifest["identity"]
    if fingerprint(identity) != manifest["run_id"]:
        raise ValueError(f"Manifest identity digest mismatch: {path}")
    expected_model = "qwen/qwen3.6-35b-a3b" if model == "qwen" else "deepseek/deepseek-v4.1-flash-1017"
    for role in ["reader", "judge"]:
        service = identity[f"{role}_service"]
        if service["model"] != expected_model or identity["config"][f"{role}_model"] != expected_model:
            raise ValueError(f"Wrong {role} model: {path}")
        body = service["extra_body"]
        provider = "DeepInfra" if model == "qwen" else "DeepSeek"
        if body["reasoning"]["enabled"] is not False or body["provider"]["allow_fallbacks"] is not False:
            raise ValueError("Reasoning or provider fallback differs")
        if body["provider"].get("only", body["provider"].get("order")) != [provider]:
            raise ValueError("Provider route differs")
        if model == "qwen" and body["provider"].get("quantizations") != ["fp8"]:
            raise ValueError("Qwen quantization differs")
    if method in {"SAG-Mem", "SAG-Mem-Lite"}:
        config = identity["config"]
        expected_retrieval = (100, 2, 20) if method == "SAG-Mem-Lite" else (50, 1, 20)
        if (config["backend_options"]["candidate_k"], config["backend_options"]["max_hops"], config["top_k"]) != expected_retrieval:
            raise ValueError("Primary retrieval settings differ")
        if lineage := identity.get("verified_replay"):
            source = (ROOT / lineage["source_relative_path"] if lineage.get("source_relative_path")
                      else Path(lineage["source_path"]))
            source_manifest_path = source.with_suffix(".jsonl.manifest.json")
            if digest(source) != lineage["source_sha256"] or digest(source_manifest_path) != lineage["source_manifest_sha256"]:
                raise ValueError("Verified replay source content changed")
            source_manifest = json.loads(source_manifest_path.read_text())
            if source_manifest["run_id"] != lineage["source_run_id"]:
                raise ValueError("Verified replay source identity changed")
        # Saved-context manifests retain their source run identity and byte hash.
        if "source_sha256" in identity:
            source = Path(manifest["source_path"])
            if not source.exists():
                parts = source.parts
                if "experiments" in parts:
                    source = ROOT / Path(*parts[parts.index("experiments"):])
                else:
                    position = parts.index("results")
                    source = ROOT / "experiments/main" / Path(*parts[position:])
            if digest(source) != identity["source_sha256"]:
                raise ValueError("Replay source content changed")
            source_manifest = json.loads(source.with_suffix(".jsonl.manifest.json").read_text())
            if source_manifest["run_id"] != identity["source_run"]:
                raise ValueError("Replay source identity changed")
        else:
            chunking = identity["build_dependencies"]["chunking"]
            if (chunking["policy"], chunking["messages_per_chunk"]) not in {
                ("soft512_atomic_v1", None), ("three_messages_soft512_atomic_v1", 3),
                ("3_messages_soft512_atomic_v1", 3),
            }:
                raise ValueError("Chunk policy differs")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bootstrap(rows, other=None):
    """Resample histories; LoCoMo has only ten independent histories."""
    groups = defaultdict(list)
    for qid, row in rows.items():
        value = int(row["grade"])
        if other is not None:
            value -= int(other[qid]["grade"])
        groups[row["history"]].append(value)
    values = list(groups.values())
    counts = np.array([len(v) for v in values])
    totals = np.array([sum(v) for v in values])
    rng = np.random.default_rng(SEED)
    samples = []
    for start in range(0, REPLICATES, 500):
        selected = rng.integers(0, len(values), size=(min(500, REPLICATES - start), len(values)))
        samples.extend((100 * totals[selected].sum(axis=1) / counts[selected].sum(axis=1)).tolist())
    return [float(v) for v in np.percentile(samples, [2.5, 97.5])]


def read_run(files, dataset, method, model):
    rows, manifests, sources, identities = {}, [], [], {}
    independent = independent_questions(dataset)
    for path in sorted(files):
        manifest_path = path.with_suffix(".jsonl.manifest.json")
        manifest = json.loads(manifest_path.read_text())
        verify_manifest(manifest, manifest_path, method, model)
        manifests.append(manifest)
        identity = manifest["identity"]
        questions = identity.get("questions") or manifest.get("reporting_questions")
        if questions:
            for item in questions:
                qid = item["id"]
                if qid in identities:
                    raise ValueError(f"Overlapping shards: {qid}")
                identities[qid] = item
        own = {}
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["run_id"] != manifest["run_id"]:
                    raise ValueError(f"Run identity mismatch: {path}")
                qid = row["question_id"]
                if (row["query"]["text"], row["question_type"], row["gold_answer"],
                    row["is_abstention"], row["query"]["date_raw"]) != independent[qid]:
                    raise ValueError(f"Question, target, or source date differs from frozen data: {qid}")
                stats = row.get("backend_stats", {})
                jev = stats.get("source_jev_usage", stats.get("jev_usage"))
                own[qid] = {
                    "grade": row.get("grade"), "status": row["status"],
                    "category": CATEGORIES[dataset].get(row["question_type"], row["question_type"]),
                    "history": row.get("source_history_sha256", row["corpus_id"]),
                    "build_id": row["build_id"], "truncated": row["context_truncated"] if row["status"] == "succeeded" else None,
                    "reader_input_tokens": row.get("reader_prompt_tokens"),
                    "reader_output_tokens": row.get("reader_completion_tokens"),
                    "grader_input_tokens": row.get("grader_prompt_tokens"),
                    "grader_output_tokens": row.get("grader_completion_tokens"),
                    "reader_usage": response_usage(row, "reader") if row.get("reader_prompt_tokens") is not None else None,
                    "grader_usage": response_usage(row, "judge") if row.get("grader_prompt_tokens") is not None else None,
                    "jev_input_tokens": jev["input_tokens"] if jev else 0,
                    "jev_output_tokens": jev["output_tokens"] if jev else 0,
                    "jev_requests": jev["requests"] if jev else 0,
                    "jev_judgments": jev["judgments"] if jev else 0,
                    "time_annotations": len(stats.get("time_explanations", [])),
                    "reader_seconds_batch": row.get("reader_duration"),
                    "retrieval_seconds_batch": row.get("retrieval_duration"),
                }
                if row["backend"] == "sag-mem" and jev is None:
                    raise ValueError(f"Missing JEV usage: {qid}")
        if set(rows) & set(own):
            raise ValueError(f"Duplicate shard questions: {path}")
        rows.update(own)
        sources.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path),
                        "manifest_sha256": digest(manifest_path), "run_id": manifest["run_id"]})
    if identities and not set(rows) <= set(identities):
        raise ValueError("Unknown question IDs")
    complete = len(rows) == EXPECTED[dataset] and (not identities or set(rows) == set(identities))
    complete &= set(rows) == set(independent)
    complete &= all(r["status"] == "succeeded" and type(r["grade"]) is bool for r in rows.values())
    complete &= all(not r["truncated"] for r in rows.values())
    return rows, sources, bool(complete)


def summarize(method, model, dataset, files):
    rows, sources, complete = read_run(files, dataset, method, model)
    progress = {"method": method, "model_family": model, "dataset": dataset,
                "questions_present": len(rows), "expected": EXPECTED[dataset],
                "successful": sum(r["status"] == "succeeded" and type(r["grade"]) is bool for r in rows.values()),
                "truncated": sum(r["truncated"] is True for r in rows.values()), "complete": complete}
    if not complete:
        return progress, None, None
    prices = json.loads((OUT / "pricing.json").read_text())["models"]
    model_id = "qwen/qwen3.6-35b-a3b" if model == "qwen" else "deepseek/deepseek-v4.1-flash-1017"
    price = prices[model_id]["pricing"]
    summary = {**progress, "questions": len(rows), "correct": sum(r["grade"] for r in rows.values()),
               "provenance": "replay" if model == "deepseek" and method != "Full Context" else "new",
               "sources": sources, "latency": None, "model_version": model_id}
    summary["accuracy_pct"] = 100 * summary["correct"] / len(rows)
    summary["accuracy_ci_pct"] = bootstrap(rows)
    summary["history_clusters"] = len({r["history"] for r in rows.values()})
    category_rows = defaultdict(list)
    for row in rows.values():
        category_rows[row["category"]].append(row)
    summary["categories"] = {cat: {"questions": len(v), "correct": sum(r["grade"] for r in v),
                                    "accuracy_pct": 100 * sum(r["grade"] for r in v) / len(v)}
                             for cat, v in sorted(category_rows.items())}
    for key in ["reader_input_tokens", "reader_output_tokens", "grader_input_tokens", "grader_output_tokens",
                "jev_input_tokens", "jev_output_tokens", "jev_requests", "jev_judgments", "time_annotations"]:
        if any(r[key] is None for r in rows.values()):
            raise ValueError(f"Missing {key}: {method}/{model}/{dataset}")
        summary[key] = sum(r[key] for r in rows.values())
    summary["reader_cost_usd"] = sum(token_cost(row["reader_usage"], price["prompt"], price["completion"],
                                               price.get("input_cache_read")) for row in rows.values())
    summary["offline_grader_cost_usd"] = sum(token_cost(row["grader_usage"], price["prompt"], price["completion"],
                                                       price.get("input_cache_read")) for row in rows.values())
    jp = prices["typesafe/jev-1.14-20261014"]["pricing"]
    summary["jev_cost_usd"] = (summary["jev_input_tokens"] * float(jp["prompt"])
                              + summary["jev_output_tokens"] * float(jp["completion"]))
    summary["embedding_build_cost_usd"] = 0.0
    summary["query_embedding_cost_usd"] = 0.0
    summary["extraction_build_cost_usd"] = 0.0
    summary["other_retrieval_cost_usd"] = 0.0
    if method == "SAG-Mem-Lite":
        resource = ROOT / "experiments/lite/results" / dataset / "qwen/run.resources.json"
        if resource.exists():
            logical = json.loads(resource.read_text())
            if logical["questions"] != len(rows):
                raise ValueError("Incomplete Lite resource accounting")
            summary["embedding_build_cost_usd"] = logical["logical_embedding_build_standardized_usd"]
            summary["query_embedding_cost_usd"] = logical["canonical_query_embedding_standardized_usd"]
            summary["embedding_build_tokens"] = logical["logical_embedding_build_tokens"]
            summary["query_embedding_tokens"] = logical["canonical_query_embedding_tokens"]
            summary["index_bytes"] = logical["index_bytes"]
            summary["resource_source_sha256"] = digest(resource)
        else:
            summary["embedding_build_cost_usd"] = None
            summary["query_embedding_cost_usd"] = None
    components = [summary[k] for k in ("reader_cost_usd", "jev_cost_usd", "embedding_build_cost_usd", "query_embedding_cost_usd")]
    summary["total_cost_usd"] = None if None in components else sum(components)
    summary["cost_accounting_version"] = "operational_v2"
    summary["cost_basis"] = ("Fixed list prices with recorded prompt-cache discounts; price uncached if cache metadata "
                             "is unavailable; unique history build once; retrieval plus Reader; excludes offline grader.")
    if method not in {"SAG-Mem", "SAG-Mem-Lite", "Full Context"}:
        summary.update(embedding_build_cost_usd=None, query_embedding_cost_usd=None,
                       extraction_build_cost_usd=None, other_retrieval_cost_usd=None, total_cost_usd=None)
    return progress, summary, rows


def read_ablations(records):
    summaries = []
    for path in sorted((ROOT / "experiments/ablations/runs").glob("*/*/*/run.jsonl")):
        method_slug, dataset, condition = path.parts[-4:-1]
        method = {"sag-mem": "SAG-Mem", "sag-mem-lite": "SAG-Mem-Lite"}[method_slug]
        source_key = (method, "qwen", dataset)
        if source_key not in records:
            continue
        manifest_path = path.with_suffix(".jsonl.manifest.json")
        manifest = json.loads(manifest_path.read_text())
        expected = set(manifest["question_order"])
        parent = {qid: records[source_key][qid] for qid in expected}
        rows = {}
        for line in path.open():
            row = json.loads(line)
            if row["run_id"] != manifest["run_id"]:
                raise ValueError("Ablation run identity mismatch")
            qid = row["question_id"]
            if row["source_main_grade"] != parent[qid]["grade"]:
                raise ValueError("Ablation parent grade mismatch")
            rows[qid] = {"history": parent[qid]["history"], "grade": row["grade"], "status": row["status"]}
        if set(rows) != expected or any(r["status"] != "succeeded" or type(r["grade"]) is not bool for r in rows.values()):
            continue
        correct = sum(r["grade"] for r in rows.values())
        count = len(rows)
        if count != (EXPECTED[dataset] if condition == "no_expansion" else {"locomo": 321, "longmemeval": 133}[dataset]):
            raise ValueError("Unexpected ablation population")
        base_correct = sum(r["grade"] for r in parent.values())
        summaries.append({"method": method, "model_family": "qwen", "dataset": dataset,
                          "condition": condition, "scope": "all" if condition == "no_expansion" else "temporal",
                          "provenance": "new", "complete": True, "questions": count, "correct": correct,
                          "accuracy_pct": 100 * correct / count, "accuracy_ci_pct": bootstrap(rows),
                          "reference": {"questions": count, "correct": base_correct,
                                        "accuracy_pct": 100 * base_correct / count, "accuracy_ci_pct": bootstrap(parent)},
                          "difference_pp": 100 * (correct - base_correct) / count,
                          "difference_ci_pp": bootstrap(rows, parent),
                          "condition_only_correct": sum(rows[q]["grade"] and not parent[q]["grade"] for q in rows),
                          "main_only_correct": sum(parent[q]["grade"] and not rows[q]["grade"] for q in rows),
                          "question_ids_sha256": hashlib.sha256(json.dumps(sorted(expected)).encode()).hexdigest(),
                          "source": str(path.relative_to(ROOT)), "source_sha256": digest(path),
                          "manifest_sha256": digest(manifest_path)})
    return summaries


def add_latency(runs):
    paths = [p for p in [ROOT / "experiments/latency/trials.jsonl",
                         ROOT / "experiments/full_context/latency_trials.jsonl"] if p.exists()]
    trials = {}
    for path in paths:
        for line in path.open():
            row = json.loads(line)
            trials[row["key"]] = {**row, "source_path": path}
    for run in runs:
        selected = [r for r in trials.values() if r["status"] == "succeeded"
                    and (r["method"], r["model_family"], r["dataset"]) == (run["method"], run["model_family"], run["dataset"])]
        if not selected:
            continue
        if len(selected) != 50:
            continue
        times = [r["total_seconds"] for r in selected]
        run["latency"] = {"questions": len(times), "mean_seconds": float(np.mean(times)),
                          "median_seconds": float(np.median(times)), "p95_seconds": float(np.percentile(times, 95)),
                          "retrieval_median_seconds": float(np.median([r["retrieval_seconds"] for r in selected])),
                          "reader_median_seconds": float(np.median([r["reader_seconds"] for r in selected])),
                          "concurrency": 1, "source_sha256": digest(selected[0]["source_path"])}


def representation_ablations(records):
    path = OUT / "representation_controls.json"
    if not path.exists():
        return []
    document = json.loads(path.read_text())
    if document["matching_effective_settings"] is not True or document["new_model_requests"] != 0:
        raise ValueError("Representation cache validation is incomplete")
    results = []
    for control in document["controls"]:
        key = ("SAG-Mem", "qwen", control["dataset"])
        if key not in records:
            continue
        parent = records[key]
        if set(control["records"]) != set(parent):
            raise ValueError("Representation-control question sets differ")
        compared = {}
        for qid, row in control["records"].items():
            if row["main_grade"] != parent[qid]["grade"] or row["status"] != "succeeded":
                raise ValueError("Representation-control parent identity differs")
            compared[qid] = {"history": parent[qid]["history"], "grade": row["grade"]}
        n = len(compared)
        correct = sum(r["grade"] for r in compared.values())
        reference_correct = sum(r["grade"] for r in parent.values())
        results.append({
            "method": "SAG-Mem", "model_family": "qwen", "dataset": control["dataset"],
            "condition": "uncapped_messages", "scope": "all", "provenance": "compatible_cache",
            "complete": True, "questions": n, "correct": correct, "accuracy_pct": 100 * correct / n,
            "accuracy_ci_pct": bootstrap(compared),
            "reference": {"questions": n, "correct": reference_correct,
                          "accuracy_pct": 100 * reference_correct / n, "accuracy_ci_pct": bootstrap(parent)},
            "difference_pp": 100 * (correct - reference_correct) / n,
            "difference_ci_pp": bootstrap(compared, parent),
            "reader_input_tokens": control["reader_input_tokens"],
            "reader_output_tokens": control["reader_output_tokens"],
            "jev_input_tokens": control["jev_input_tokens"], "jev_output_tokens": control["jev_output_tokens"],
            "reference_reader_input_tokens": control["reference"]["reader_input_tokens"],
            "reader_input_reduction_pct": control["main_reader_input_token_reduction_pct"],
            "question_ids_sha256": control["question_ids_sha256"],
            "source_sha256": digest(path), "source": str(path.relative_to(ROOT)),
        })
    return results


def main():
    global DATA_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    DATA_ROOT = args.data_root.resolve()
    runs, progress, records = [], [], {}
    for method in ["SAG-Mem", "SAG-Mem-Lite"]:
        for model in ["qwen", "deepseek"]:
            for dataset, expected_questions in EXPECTED.items():
                if method == "SAG-Mem":
                    group = "results" if model == "qwen" else "results_deepseek"
                    files = list((ROOT / "experiments/main" / group / dataset).glob("*/run.jsonl"))
                elif method == "SAG-Mem-Lite":
                    files = list((ROOT / "experiments/lite/results" / dataset / model).glob("run.jsonl"))
                if not files:
                    progress.append({"method": method, "model_family": model, "dataset": dataset,
                                     "questions_present": 0, "expected": expected_questions, "complete": False})
                    continue
                state, summary, rows = summarize(method, model, dataset, files)
                progress.append(state)
                if summary:
                    runs.append(summary)
                    records[(method, model, dataset)] = rows
    reference_path = OUT / "baseline_reference.json"
    reference = json.loads(reference_path.read_text())
    source = reference["sources"]["manuscript"]
    if args.baseline_source and digest(args.baseline_source) != source["source_sha256"]:
        raise ValueError("Baseline source changed after extraction")
    baselines = []
    for row in reference["baselines"]:
        if row["accuracy_pct"] is None:
            continue
        if not 0 <= row["accuracy_pct"] <= 100:
            raise ValueError("Baseline percentage is outside 0..100")
        baselines.append({**row, "questions": EXPECTED[row["dataset"]],
                          "provenance": "reported-only", "complete": True})
    paired = []
    for model in ["qwen", "deepseek"]:
        for dataset in EXPECTED:
            key = ("SAG-Mem", model, dataset)
            for control in ["SAG-Mem-Lite"]:
                other_key = (control, model, dataset)
                if key not in records or other_key not in records:
                    continue
                a, b = records[key], records[other_key]
                # Question IDs form the pair; main-run histories define bootstrap clusters.
                if set(a) != set(b):
                    raise ValueError("Pairing question IDs differ")
                paired.append({"method": "SAG-Mem", "control": control, "model_family": model,
                               "dataset": dataset, "difference_pp": 100 * sum(int(a[q]["grade"]) - int(b[q]["grade"]) for q in a) / len(a),
                               "difference_ci_pp": bootstrap(a, b),
                               "a_only_correct": sum(a[q]["grade"] and not b[q]["grade"] for q in a),
                               "b_only_correct": sum(b[q]["grade"] and not a[q]["grade"] for q in a)})
    add_latency(runs)
    storage_path = ROOT / "experiments/storage/storage_summary.json"
    storage = json.loads(storage_path.read_text()) if storage_path.exists() else None
    storage_rows = storage["summary"] if storage and storage["state"] == "complete" else []
    result = {"runs": runs, "baselines": baselines,
              "ablations": [*read_ablations(records), *representation_ablations(records)], "paired_comparisons": paired,
              "storage": storage_rows,
              "metadata": {"bootstrap_seed": SEED, "bootstrap_replicates": REPLICATES,
                           "bootstrap_unit": "conversation history", "x_axis": "linear, equally spaced",
                           "complete_runs_only": True, "progress": progress,
                           "baseline_reference_sha256": digest(reference_path),
                           "baseline_resource_units": reference["sources"]["manuscript"]["units"],
                           "storage_source_sha256": digest(storage_path) if storage_rows else None}}
    temporary = OUT / "final_results.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(OUT / "final_results.json")
    (OUT / "question_metrics.json").write_text(json.dumps({"|".join(k): v for k, v in records.items()}) + "\n")
    print(json.dumps(progress, indent=2))
    if args.require_all and not all(r["complete"] for r in progress):
        raise SystemExit("Full primary experiment coverage is required before publication generation")


if __name__ == "__main__":
    main()
