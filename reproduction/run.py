#!/usr/bin/env python3
"""Portable paper runner; --help and --preflight never call a model API."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from reproduction.experiments.shared.profiles import (
    DEFAULT_EMBEDDING_PROFILE,
    EMBEDDING_OVERFLOW_RULES,
    EMBEDDING_PROFILES,
    PROFILES,
    api_settings,
    apply_paper_defaults_only_when_unset,
    reader_judge_models,
)
from reproduction.load_env import load_repo_env


def parse_cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["sag-mem", "sag-mem-lite"],
                        required=True)
    parser.add_argument("--dataset", choices=["locomo", "longmemeval"], required=True)
    parser.add_argument("--model", choices=list(PROFILES), default="qwen")
    parser.add_argument("--embedding-profile", choices=list(EMBEDDING_PROFILES), default=DEFAULT_EMBEDDING_PROFILE)
    parser.add_argument("--embedding-overflow", choices=EMBEDDING_OVERFLOW_RULES, default="error")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--index-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument(
        "--build-concurrency",
        type=int,
        default=4,
        help="Parallel history builds (default 4). Paper used 2; does not change retrieval scores.",
    )
    parser.add_argument("--candidate-k", type=int, help="Default: Lite 100; SAG-Mem 50")
    parser.add_argument("--max-hops", type=int, help="Default: Lite 2; SAG-Mem 1")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--messages-per-chunk", type=int, default=3,
                        help="Maximum complete messages per chunk (default: 3). 0 uses only the soft 512-token target.")
    parser.add_argument("--questions-file", type=Path,
                        help="JSON list of question ids. Used for a frozen development split.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--replay-from", type=Path, help="Verified Reader/Judge-only replay of a saved run")
    args = parser.parse_args(argv)
    if args.candidate_k is None:
        args.candidate_k = 100 if args.method == "sag-mem-lite" else 50
    if args.max_hops is None:
        args.max_hops = 2 if args.method == "sag-mem-lite" else 1
    args.data_path_supplied = args.data_path is not None
    args.index_root_supplied = args.index_root is not None
    args.cache_root_supplied = args.cache_root is not None
    if (
        args.concurrency < 1
        or args.build_concurrency < 1
        or not 1 <= args.top_k <= args.candidate_k
        or not 0 <= args.max_hops <= 2
    ):
        parser.error(
            "Require concurrency >= 1, build-concurrency >= 1, "
            "1 <= top-k <= candidate-k, and 0 <= max-hops <= 2"
        )
    if args.limit < 0:
        parser.error("limit cannot be negative")
    if not 0 <= args.messages_per_chunk <= 12:
        parser.error("messages-per-chunk must be an integer from 0 to 12")
    if args.questions_file is not None:
        try:
            question_ids = json.loads(args.questions_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"questions-file is not readable JSON: {exc}")
        if not isinstance(question_ids, list) or not question_ids or any(
            type(item) is not str or not item for item in question_ids
        ):
            parser.error("questions-file must be a non-empty JSON list of question id strings")
        args.question_ids = question_ids
    else:
        args.question_ids = []
    if args.data_path is None:
        name = "locomo10.json" if args.dataset == "locomo" else "longmemeval_s_cleaned.json"
        args.data_path = ROOT / "data" / name
    if args.output is None:
        if args.method == "sag-mem-lite":
            args.output = ROOT / "experiments/lite/results" / args.dataset / args.model / "run.jsonl"
        else:
            base = "results" if args.model == "qwen" else "results_deepseek"
            args.output = ROOT / "experiments/main" / base / args.dataset / "full/run.jsonl"
    args.index_root = (args.index_root or ROOT / "indexes").resolve()
    args.cache_root = (args.cache_root or args.index_root / "cache").resolve()
    args.output = args.output.resolve()
    args.data_path = args.data_path.resolve()
    return args


def configure(args):
    load_repo_env()
    profile = PROFILES[args.model]
    apply_paper_defaults_only_when_unset()
    if args.preflight or args.replay_from:
        from src.config import settings
        reader_model, judge_model = settings.search_llm.model, settings.judge_llm.model
    else:
        reader_model, judge_model = reader_judge_models(
            profile, require_jev=args.method == "sag-mem")
    from scripts.run_benchmark import parse_args
    from src.harness.benchmark import REGISTRIES, register

    if args.method == "sag-mem":
        from reproduction.experiments.main.paper_backend import PaperSAGMem
        backend = "sag-mem"
        # The paper module owns registration of this backend.
        if backend not in REGISTRIES["backend"]:
            register("backend", backend, PaperSAGMem)
    else:
        from reproduction.experiments.lite.backend import BatchedLite, LiteBackend
        backend = "sag_mem_lite" if args.dataset == "locomo" else "sag_mem_lite_parallel"
        if backend not in REGISTRIES["backend"]:
            register("backend", backend, LiteBackend if args.dataset == "locomo" else BatchedLite)
    backend_options = {"candidate_k": args.candidate_k, "max_hops": args.max_hops}
    if args.method == "sag-mem-lite":
        backend_options["embedding_batch_size"] = 64
    if args.messages_per_chunk != 0:
        backend_options["messages_per_chunk"] = args.messages_per_chunk
    options = parse_args([
        "--dataset", args.dataset, "--data-path", str(args.data_path), "--backend", backend,
        "--backend-options", json.dumps(backend_options), "--top-k", str(args.top_k),
        "--concurrency", str(args.concurrency),
        "--namespace", args.method,
        "--index-root", str(args.index_root), "--output", str(args.output),
        "--reader-model", reader_model, "--judge-model", judge_model,
        "--model-timeout", "240", "--worker-retries", "0", "--limit", str(args.limit),
        *(["--questions", *args.question_ids] if args.question_ids else []),
        *(["--resume"] if args.resume else []),
    ])
    options.update(cache_root=str(args.cache_root), method=args.method,
                   embedding_profile=args.embedding_profile, embedding_overflow=args.embedding_overflow)
    return options


def preflight(options):
    from src.data.loader import _lme_history, _locomo_history, iter_dataset
    from src.harness.benchmark import resolve, select_examples

    from reproduction.experiments.shared.runtime import audit_chunks, chunking, file_digest

    examples = select_examples(resolve("dataset", options["dataset"])().scan(options), options)
    selected = {example.history_ref["index"] for example in examples}
    audits = []
    messages_per_chunk = options["backend_options"].get("messages_per_chunk")
    for position, item in enumerate(iter_dataset(Path(options["data_path"]))):
        if position in selected:
            history = _locomo_history(item, "text") if options["dataset"] == "locomo" else _lme_history(item)
            if messages_per_chunk:
                from reproduction.experiments.shared.chunk_width import chunk_history
                chunks = chunk_history(history, messages_per_chunk)
            else:
                chunks = chunking.chunk_history(history)
            audits.append(audit_chunks(history, chunks, messages_per_chunk))
    policy = chunking.POLICY if messages_per_chunk is None else f"{messages_per_chunk}_messages_soft512_atomic_v1"
    return {"questions": len(examples), "histories": len(audits), "policy": policy,
            "messages_per_chunk": messages_per_chunk,
            "dataset_sha256": file_digest(Path(options["data_path"])), "model_calls": 0,
            "messages": sum(row["messages"] for row in audits),
            "chunks": sum(row["chunks"] for row in audits),
            "oversized_chunks": sum(row["oversized_chunks"] for row in audits)}


async def execute(args, options, prepared):
    from src.data.loader import _lme_history, _locomo_history, iter_dataset
    from src.harness.benchmark import (
        Query,
        ResultStore,
        RetrievalResult,
        acquire_build_lock,
        atomic_json,
        file_digest,
        fingerprint,
        make_build,
        resolve,
        score_record,
    )
    from src.harness.reader_grader import aggregate_metrics, evidence_metrics

    from reproduction.experiments.shared.runtime import QUESTION, STAGE, PaperReader, RequestLedger, sha256

    examples, dependencies, manifest = prepared
    manifest["identity"]["reproduction"] = {
        "runner_sha256": file_digest(Path(__file__)),
        "runtime": "src",
    }
    manifest["scheduling"] = {"concurrency": args.concurrency, "build_concurrency": args.build_concurrency}
    from src.config import settings as harness_settings
    for role, endpoint in (("reader_service", harness_settings.search_llm),
                           ("judge_service", harness_settings.judge_llm)):
        manifest["identity"][role] = {"base_url": endpoint.base_url, "model": endpoint.model,
                                     "extra_body": endpoint.extra_body}
    manifest["run_id"] = fingerprint(manifest["identity"])
    store = ResultStore(args.output, manifest, args.resume)
    ledger = RequestLedger(args.output.parent / "requests.jsonl")
    reader = PaperReader(PROFILES[args.model], ledger)
    protocol = resolve("protocol", args.dataset)()
    groups = {}
    for example in examples:
        if store.records.get(example.id, {}).get("status") != "succeeded":
            groups.setdefault(example.history_digest, []).append(example)
    waiting = deque(groups.values())
    history_by_index = {index: item for index, item in enumerate(iter_dataset(args.data_path))}
    queue = asyncio.Queue(maxsize=args.concurrency)
    slots = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()

    def finish(record):
        record["api_calls"] = [call for call in ledger.records if call["question_id"] == record["question_id"]]
        store.append(record)
        print(json.dumps({"completed": len(store.records), "total": len(examples),
                          "question_id": record["question_id"], "status": record["status"],
                          "error": record.get("error")}), flush=True)

    def new_record(example, spec):
        return {"question_id": example.id, "corpus_id": example.corpus_id, "build_id": spec.id,
                "run_id": manifest["run_id"], "backend": options["backend"], "protocol": args.dataset,
                "question_type": example.target.category, "query": asdict(example.query),
                "target": asdict(example.target), "gold_answer": example.target.answer,
                "is_abstention": example.target.abstention, "source_history_sha256": example.history_digest,
                "retrieval_only": False, "grade": None}

    async def build():
        while waiting:
            members = waiting.popleft()
            index = members[0].history_ref["index"]
            item = history_by_index.get(index)
            if item is None:
                raise ValueError("History reference is absent from the dataset")
            history = _locomo_history(item, "text") if args.dataset == "locomo" else _lme_history(item)
            if history.digest != members[0].history_digest:
                raise ValueError("History changed after preflight")
            spec = make_build(members[0], options, dependencies)
            lock = await acquire_build_lock(spec)
            backend = resolve("backend", options["backend"])(options)
            backend.ledger = ledger
            try:
                QUESTION.set(members[0].id)
                STAGE.set("embedding_build")
                try:
                    await backend.prepare(history, spec)
                except Exception as error:  # noqa: BLE001 - persist build failures for every affected question
                    for example in members:
                        record = new_record(example, spec)
                        record.update(status="failed", error=f"{type(error).__name__}: {error}",
                                      failure_stage="prepare")
                        finish(record)
                    continue
                pending = deque(members)

                async def retrieve(pending=pending, backend=backend, spec=spec, history=history):
                    while pending:
                        example = pending.popleft()
                        await slots.acquire()
                        QUESTION.set(example.id)
                        STAGE.set("retrieval")
                        record = new_record(example, spec)
                        try:
                            key = fingerprint({"query": asdict(example.query), "history": history.digest,
                                "dependencies": dependencies, "runtime": backend.runtime_dependencies(options),
                                "top_k": args.top_k})
                            cache = args.cache_root / "retrieval" / args.dataset / f"{key}.json"
                            tick = time.perf_counter()
                            if cache.exists():
                                saved = json.loads(cache.read_text())
                                if saved["key"] != key or fingerprint(saved["result"]) != saved["sha256"]:
                                    raise ValueError("Retrieval cache hash mismatch")
                                result = RetrievalResult(**saved["result"])
                                for evidence in result.items:
                                    source = backend.mapping[str(evidence["sag2_chunk_id"])]
                                    if any(evidence[k] != v for k, v in source.items()
                                           if k != "indexed_content" or k in evidence):
                                        raise ValueError("Cached source differs from complete original text")
                                record.update(retrieval_cache_reused=True,
                                              source_retrieval_duration=saved["elapsed_seconds"])
                            else:
                                query = protocol.question(example.query, options["question_date_text"])
                                result = await backend.retrieve(Query(query, example.query.date_raw,
                                                                     example.query.timestamp), options)
                                atomic_json(cache, {"key": key, "sha256": fingerprint(asdict(result)),
                                    "result": asdict(result), "elapsed_seconds": time.perf_counter() - tick})
                                record["retrieval_cache_reused"] = False
                            record.update(context=result.context, evidence=result.items, backend_stats=result.stats,
                                retrieval_duration=time.perf_counter() - tick, retrieval_cache_key=key,
                                source_content_sha256=fingerprint([item["content"] for item in result.items]),
                                reader_context_sha256=sha256(result.context), **evidence_metrics(example.target, result))
                            await queue.put((record, example))
                        except Exception as error:  # noqa: BLE001 - retain failed retrievals for --resume
                            record.update(status="failed", error=f"{type(error).__name__}: {error}",
                                          failure_stage="retrieval")
                            finish(record)
                            slots.release()

                async with asyncio.TaskGroup() as tasks:
                    for _ in range(min(args.concurrency, len(pending))):
                        tasks.create_task(retrieve())
            finally:
                await backend.aclose()
                lock.close()

    async def consume():
        while True:
            item = await queue.get()
            if item is None:
                return
            record, example = item
            QUESTION.set(example.id)
            try:
                await score_record(record, example.target, example.query, protocol, reader, options)
                if record["context_truncated"] or record["reader_context"] != record["context"]:
                    raise ValueError("Reader context was truncated or changed")
            except asyncio.CancelledError:
                record.update(status="interrupted", grade=None, error="Evaluation cancelled")
                raise
            except Exception as error:  # noqa: BLE001 - retain failed model calls for --resume
                record.update(status="failed", grade=None, error=f"{type(error).__name__}: {error}",
                              failure_stage="reader_or_judge")
            finally:
                finish(record)
                slots.release()

    async def produce():
        workers = min(args.build_concurrency, len(waiting) or 1)
        async with asyncio.TaskGroup() as tasks:
            for _ in range(workers):
                tasks.create_task(build())
        for _ in range(args.concurrency):
            await queue.put(None)

    try:
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(produce())
            for _ in range(args.concurrency):
                tasks.create_task(consume())
        summary = aggregate_metrics(list(store.records.values()), manifest)
        summary["wall_seconds"] = time.perf_counter() - started
        atomic_json(args.output.with_suffix(".summary.json"), summary)
        return int(any(row["status"] != "succeeded" for row in store.records.values()))
    finally:
        await reader.aclose()
        ledger.close()
        store.close()


def main(argv=None):
    args = parse_cli(argv)
    options = configure(args)
    if args.preflight:
        print(json.dumps(preflight(options), ensure_ascii=False, indent=2))
        return 0
    if args.replay_from:
        from reproduction.experiments.lite.replay import replay
        return asyncio.run(replay(args.replay_from.resolve(), args.output, args.model, args.resume,
                                  args.index_root if args.index_root_supplied else None,
                                  args.cache_root if args.cache_root_supplied else None, args.concurrency,
                                  args.data_path if args.data_path_supplied else None))
    from src.harness.benchmark import prepare_run
    api_settings(require_jev=args.method == "sag-mem")
    return asyncio.run(execute(args, options, prepare_run(options)))


if __name__ == "__main__":
    raise SystemExit(main())
