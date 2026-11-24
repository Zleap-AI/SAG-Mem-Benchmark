"""Paired serial queries over warm indexes; no application answer/retrieval cache."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
REPRO_ROOT = EXP.parent
REPO = REPRO_ROOT.parent
OUTPUT_ROOT = ROOT
DATA_ROOT = REPRO_ROOT / "data"
LITE_ROOT = EXP / "lite"
MAIN_INDEX_ROOT = ROOT / "indexes"
LITE_INDEX_ROOT = None
LITE_CACHE_ROOT = None
sys.path.insert(0, str(REPO))
from reproduction.load_env import load_repo_env

load_repo_env()
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

# Settings read environment variables during import.

from src.harness.benchmark import Query, atomic_json, file_digest, make_build, prepare_run, resolve

from reproduction.experiments.lite.backend import BatchedLite, LiteBackend
from reproduction.experiments.shared.runtime import PROFILES, QUESTION, STAGE, PaperReader, RequestLedger


def select(examples, dataset):
    if dataset == "locomo":
        groups = defaultdict(list)
        for example in examples:
            groups[example.history_digest].append(example)
        if len(groups) != 10:
            raise ValueError(f"Expected 10 LoCoMo samples, found {len(groups)}")
        selected = []
        for digest, members in sorted(groups.items()):
            ordered = sorted(members, key=lambda e: hashlib.sha256(("20261118:" + e.id).encode()).digest())
            if len(ordered) < 10:
                raise ValueError(f"LoCoMo sample {digest} has fewer than 10 questions")
            selected.extend(ordered[:10])
        return sorted(selected, key=lambda e: (e.history_digest, e.id))
    quotas = ({"1": 7, "2": 8, "3": 2, "4": 21, "5": 12} if dataset == "locomo" else
              {"multi-session": 13, "knowledge-update": 8, "temporal-reasoning": 13,
               "single-session-preference": 3, "single-session-assistant": 6, "single-session-user": 7})
    groups = defaultdict(list)
    for example in examples:
        groups[example.target.category].append(example)
    selected = []
    for category, quota in quotas.items():
        ordered = sorted(groups[category], key=lambda e: hashlib.sha256(("20261105:" + e.id).encode()).digest())
        selected.extend(ordered[:quota])
    assert len(selected) == 50
    return sorted(selected, key=lambda e: (e.history_digest, e.id))


def fresh_query_cache(backend):
    # Only document vectors are reusable. Query embeddings must incur a fresh call.
    backend.cache.close()
    backend.cache = sqlite3.connect(":memory:")
    backend.cache.execute("CREATE TABLE vectors (key TEXT PRIMARY KEY, text_hash TEXT NOT NULL, "
                          "vector BLOB NOT NULL, vector_hash TEXT NOT NULL)")


async def main(args):
    from reproduction.experiments.main.paper_backend import PaperSAGMem
    from reproduction.run import configure, parse_cli

    output = OUTPUT_ROOT / "trials.jsonl"
    prior = {r["key"]: r for r in map(json.loads, output.read_text().splitlines())} if output.exists() else {}
    ledger = RequestLedger(OUTPUT_ROOT / "requests.jsonl")
    readers = {name: PaperReader(profile, ledger) for name, profile in PROFILES.items()}
    datasets = [args.dataset] if args.dataset else ["locomo", "longmemeval"]
    try:
        for dataset in datasets:
            data = DATA_ROOT / ("locomo10.json" if dataset == "locomo" else "longmemeval_s_cleaned.json")
            common = ["--dataset", dataset, "--model", args.model_family, "--data-path", str(data),
                      "--top-k", "20", "--concurrency", "1", "--output", str(OUTPUT_ROOT / "unused.jsonl")]
            options = configure(parse_cli([*common, "--method", "sag-mem",
                                           "--index-root", str(MAIN_INDEX_ROOT)]))
            examples, dependencies, _manifest = prepare_run(options)
            chosen = select(examples, dataset)
            atomic_json(OUTPUT_ROOT / f"selection_{dataset}.json", {
                "seed": 20261118 if dataset == "locomo" else 20261105,
                "selection_uses": "question IDs and history hashes only" if dataset == "locomo" else
                                  "question IDs, categories, and history hashes only",
                "question_ids": [e.id for e in chosen], "categories": dict(__import__("collections").Counter(e.target.category for e in chosen)),
                "history_clusters": len({e.history_digest for e in chosen}), "data_sha256": file_digest(data)})
            lite_options = configure(parse_cli([*common, "--method", "sag-mem-lite",
                                                "--index-root", str(LITE_INDEX_ROOT or OUTPUT_ROOT / "lite_indexes"),
                                                "--cache-root", str(LITE_CACHE_ROOT or OUTPUT_ROOT / "lite_cache")]))
            lite_type = BatchedLite if dataset == "longmemeval" else LiteBackend
            lite_dependencies = lite_type.dependencies(lite_options)
            protocol = resolve("protocol", options["protocol"])()
            dataset_loader = resolve("dataset", dataset)()
            backends, active_history = {}, None
            try:
                for position, example in enumerate(chosen):
                    if active_history != example.history_digest:
                        for backend in backends.values():
                            await backend.aclose()
                        history = dataset_loader.load_history(example.history_ref)
                        backends = {"SAG-Mem": PaperSAGMem(options), "SAG-Mem-Lite": lite_type(lite_options, ledger)}
                        STAGE.set("index_prepare")
                        await backends["SAG-Mem"].prepare(history, make_build(example, options, dependencies))
                        lite_spec = make_build(example, lite_options, lite_dependencies)
                        if not (lite_spec.directory / "build.json").exists() and not args.prepare_missing_lite_index:
                            raise ValueError("Wait for the verified Lite index before latency measurement")
                        if (lite_spec.directory / "build.json").exists() and not (lite_spec.directory / "sag2.sqlite").exists():
                            raise ValueError("Latency requires an uncompressed, prebuilt Lite index")
                        await backends["SAG-Mem-Lite"].prepare(history, lite_spec)
                        active_history = example.history_digest
                    question = protocol.question(example.query, options["question_date_text"])
                    for family, reader in readers.items():
                        if family != args.model_family:
                            continue
                        names = ["SAG-Mem", "SAG-Mem-Lite"]
                        if (position + list(readers).index(family)) % 2:
                            names.reverse()
                        for method in names:
                            key = f"{dataset}|{family}|{example.id}|{method}"
                            if prior.get(key, {}).get("status") == "succeeded":
                                continue
                            backend = backends[method]
                            if method == "SAG-Mem-Lite":
                                fresh_query_cache(backend)
                            QUESTION.set(key)
                            STAGE.set("retrieval")
                            started = time.perf_counter()
                            result = await backend.retrieve(Query(question, example.query.date_raw, example.query.timestamp), options)
                            retrieved = time.perf_counter()
                            response = await reader.respond(result.context, question, system_prompt=protocol.reader_system)
                            ended = time.perf_counter()
                            trial = {"key": key, "dataset": dataset, "model_family": family, "method": method,
                                     "reader_model": reader.reader_model,
                                     "embedding_model": lite_dependencies["embedding"]["model"] if method == "SAG-Mem-Lite" else None,
                                     "embedding_dimensions": lite_dependencies["embedding"]["dimensions"] if method == "SAG-Mem-Lite" else None,
                                     "question_id": example.id, "category": example.target.category,
                                     "history_sha256": example.history_digest, "position": position,
                                     "method_order": names, "status": "succeeded", "timestamp": time.time(),
                                     "retrieval_seconds": retrieved - started, "reader_seconds": ended - retrieved,
                                     "total_seconds": ended - started, "reader_prompt_tokens": response.prompt_tokens,
                                     "reader_completion_tokens": response.completion_tokens,
                                     "reader_answer": response.hypothesis, "context_sha256": hashlib.sha256(result.context.encode()).hexdigest(),
                                     "selected_ids": [item["sag2_chunk_id"] for item in result.items],
                                     "jev_usage": result.stats.get("jev_usage"),
                                     "embedding_query_usage": result.stats.get("embedding_query_usage"),
                                     "load_average": os.getloadavg(), "index_build_timed": False, "judge_called": False,
                                     "application_retrieval_cache": False, "application_query_embedding_cache": False}
                            with output.open("a") as stream:
                                stream.write(json.dumps(trial) + "\n")
                            prior[key] = trial
                            atomic_json(OUTPUT_ROOT / "STATUS.json", {"completed": len(prior),
                                                                "expected": sum(100 if d == "locomo" else 50 for d in datasets) * 2,
                                                                "last": key, "phase": "serial_queries"})
                            print(json.dumps({"completed": len(prior), "last": key, "seconds": trial["total_seconds"]}), flush=True)
            finally:
                for backend in backends.values():
                    await backend.aclose()
    finally:
        for reader in readers.values():
            await reader.aclose()
        ledger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["locomo", "longmemeval"])
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--lite-root", type=Path, default=LITE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--main-index-root", type=Path)
    parser.add_argument("--lite-index-root", type=Path)
    parser.add_argument("--lite-cache-root", type=Path)
    parser.add_argument("--model-family", choices=list(PROFILES), default="qwen")
    parser.add_argument("--prepare-missing-lite-index", action="store_true")
    args = parser.parse_args()
    DATA_ROOT, LITE_ROOT, OUTPUT_ROOT = args.data_root, args.lite_root, args.output_root
    MAIN_INDEX_ROOT = args.main_index_root or OUTPUT_ROOT / "indexes"
    LITE_INDEX_ROOT, LITE_CACHE_ROOT = args.lite_index_root, args.lite_cache_root
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUTPUT_ROOT / f"environment_{args.dataset or 'all'}.json", {"python": sys.version, "platform": platform.platform(),
                "cpu_count": os.cpu_count(), "code_sha256": file_digest(Path(__file__)),
                "runtime_source_sha256": {name: file_digest(REPO / name) for name in (
                    "src/backends/sag2_package.py", "src/backends/sag2_adapter.py",
                    "src/backends/sag2_chunk_time.py", "src/harness/relative_time.py",
                    "src/harness/reader_grader.py")},
                "concurrency": 1, "selection_count_locomo": 100, "selection_count_longmemeval": 50,
                "timing": "warm index lookup, fresh retrieval and query embedding, temporal rendering, Reader; no Judge",
                "provider_cache": "Provider-side caching is not controlled; returned usage is retained in requests.jsonl.",
                "background_activity": "Shared development host; load average recorded per trial.",
                "connection_warmup": "No synthetic warm-up request; connection establishment, if required, is included."})
    asyncio.run(main(args))
