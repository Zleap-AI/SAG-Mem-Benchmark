"""Report complete coverage, actual request costs and fresh/cache latency separately."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from reproduction.analysis.costs import token_cost
from .jsonl_io import read_jsonl

PRICE_PATH = Path(__file__).with_name("prices.json")

def summarize(path):
    manifest = json.loads(path.with_suffix(".jsonl.manifest.json").read_text())
    rows = {}
    for row in read_jsonl(path):
        if row["run_id"] != manifest["run_id"]:
            raise ValueError("Result identity mismatch")
        rows[row["question_id"]] = row
    expected = {r["id"] for r in manifest["identity"]["questions"]}
    if not set(rows) <= expected:
        raise ValueError("Unexpected question in results")
    request_path = path.parent / "requests.jsonl"
    calls = read_jsonl(request_path)
    stages = defaultdict(list)
    for call in calls:
        stages[call["stage"]].append(call)
    accounting = {}
    price_path = PRICE_PATH
    prices = json.loads(price_path.read_text())["models"]
    for stage, values in stages.items():
        costs = [r["cost_usd"] for r in values if r.get("cost_usd") is not None]
        times = [r["elapsed_seconds"] for r in values]
        accounting[stage] = {
            "requests": len(values), "failed_requests": sum(r.get("status") != "succeeded" for r in values),
            "interrupted_requests": sum(r.get("status") in (None, "interrupted") for r in values),
            "actual_cost_usd": sum(costs), "missing_cost_requests": len(values) - len(costs),
            "prompt_tokens": sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in values),
            "completion_tokens": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in values),
            "latency_mean_seconds": statistics.mean(times), "latency_p50_seconds": float(np.median(times)),
            "latency_p95_seconds": float(np.percentile(times, 95)),
            "standardized_cost_usd": sum(
                token_cost(r["usage"], prices[r["model"]]["input_per_token"],
                           prices[r["model"]]["output_per_token"], prices[r["model"]].get("cached_input_per_token"))
                for r in values if r.get("usage")),
        }
    categories = defaultdict(list)
    for row in rows.values():
        categories[row["question_type"]].append(row)
    fresh = [r["retrieval_duration"] for r in rows.values()
             if "retrieval_duration" in r and not r["retrieval_cache_reused"]]
    result = {
        "run_id": manifest["run_id"], "expected": len(expected), "completed": len(rows),
        "missing_question_ids": sorted(expected - set(rows)),
        "succeeded": sum(r["status"] == "succeeded" for r in rows.values()),
        "failed": [{"question_id": r["question_id"], "stage": r.get("failure_stage"), "error": r["error"]}
                   for r in rows.values() if r["status"] != "succeeded"],
        "accuracy_over_planned": sum(r.get("grade") is True for r in rows.values()) / len(expected),
        "by_category": {k: {"completed": len(v), "correct": sum(r.get("grade") is True for r in v)}
                        for k, v in categories.items()},
        "requests": accounting,
        "actual_cost_usd": sum(s["actual_cost_usd"] for s in accounting.values()),
        "standardized_cost_usd": sum(s["standardized_cost_usd"] for s in accounting.values()),
        "standardized_price_table": prices,
        "standardized_price_source_sha256": __import__("hashlib").sha256(
            price_path.read_bytes()).hexdigest(),
        "missing_cost_requests": sum(s["missing_cost_requests"] for s in accounting.values()),
        "retrieval_cache_hits": sum(bool(r.get("retrieval_cache_reused")) for r in rows.values()),
        "fresh_retrieval_latency_p50_seconds": float(np.median(fresh)) if fresh else None,
        "fresh_retrieval_latency_p95_seconds": float(np.percentile(fresh, 95)) if fresh else None,
        "cost_scope": "Actual requests in this run; reused build/retrieval cost retained in source artifacts",
        "failed_request_cost_note": "A provider failure without usage has unknown charge; never replaced by zero",
    }
    path.with_suffix(".accounting.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.result), ensure_ascii=False, indent=2))
