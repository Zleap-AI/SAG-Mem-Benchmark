"""Verified full Reader/Judge replay; performs zero retrieval or embedding calls."""
import argparse
import asyncio
import json
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

from src.harness.benchmark import (
    Query,
    ResultStore,
    Target,
    atomic_json,
    latest_records,
    resolve,
    score_record,
)

from ..shared.profiles import llm_settings
from ..shared.runtime import PROFILES, QUESTION, PaperReader, RequestLedger, file_digest, fingerprint, sha256
from .jsonl_io import read_jsonl
from .verify import verify


async def replay(source, output, model="deepseek", resume=False, index_root=None, cache_root=None,
                 concurrency=20, data_path=None):
    verify(source, index_root, cache_root, data_path)
    original = json.loads(source.with_suffix(".jsonl.manifest.json").read_text())
    source_rows = latest_records(read_jsonl(source))
    profile = PROFILES[model]
    manifest = deepcopy(original)
    identity = manifest["identity"]
    options = identity["config"]
    settings = llm_settings(require_jev=False)
    options.update(reader_model=settings.search_llm.model, judge_model=settings.judge_llm.model)
    for role, endpoint in (("reader_service", settings.search_llm), ("judge_service", settings.judge_llm)):
        identity[role] = {"base_url": endpoint.base_url, "model": endpoint.model, "extra_body": endpoint.extra_body}
    identity["verified_replay"] = {"source_run_id": original["run_id"],
        "source_path": str(source.resolve()),
        "source_relative_path": (str(source.resolve().relative_to(Path(__file__).resolve().parents[2]))
                                 if source.resolve().is_relative_to(Path(__file__).resolve().parents[2]) else None),
        "source_sha256": file_digest(source), "source_manifest_sha256": file_digest(
            source.with_suffix(".jsonl.manifest.json")), "code_sha256": file_digest(Path(__file__)),
        "new_retrieval_calls": 0, "new_embedding_calls": 0, "concurrency": concurrency}
    manifest["run_id"] = fingerprint(identity)
    index_root = Path(index_root or options["index_root"])
    cache_root = Path(cache_root or options.get("cache_root", index_root / "cache"))
    dataset = options["dataset"]
    store = ResultStore(output, manifest, resume)
    ledger = RequestLedger(output.parent / "requests.jsonl")
    reader = PaperReader(profile, ledger)
    protocol = resolve("protocol", dataset)()
    pending = deque(row for qid, row in source_rows.items()
                    if store.records.get(qid, {}).get("status") != "succeeded")

    async def consume():
        while pending:
            source_row = pending.popleft()
            qid = source_row["question_id"]
            QUESTION.set(qid)
            cache = cache_root / "retrieval" / dataset / f"{source_row['retrieval_cache_key']}.json"
            saved = json.loads(cache.read_text())
            if (saved["key"] != source_row["retrieval_cache_key"]
                    or saved["sha256"] != fingerprint(saved["result"])
                    or saved["result"]["items"] != source_row["evidence"]
                    or sha256(saved["result"]["context"]) != source_row["reader_context_sha256"]):
                raise ValueError(f"Cached source changed immediately before replay: {qid}")
            keep = {"question_id", "corpus_id", "build_id", "backend", "protocol", "question_type",
                    "query", "target", "gold_answer", "is_abstention", "context", "evidence",
                    "backend_stats", "source_history_sha256", "source_content_sha256",
                    "reader_context_sha256", "retrieval_cache_key"}
            record = {key: deepcopy(value) for key, value in source_row.items() if key in keep}
            record.update(run_id=manifest["run_id"], source_run_id=original["run_id"],
                          source_retrieval_duration=source_row["retrieval_duration"],
                          retrieval_duration=0, retrieval_cache_reused=True, retrieval_only=False, grade=None)
            stats = record["backend_stats"]
            if "embedding_query_usage" in stats:
                stats["source_embedding_query_usage"] = stats["embedding_query_usage"]
                stats["embedding_query_usage"] = {"requests": 0, "texts": 0, "prompt_tokens": 0, "cost_usd": 0}
            if "jev_usage" in stats:
                stats["source_jev_usage"] = stats["jev_usage"]
                stats["jev_usage"] = {"requests": 0, "judgments": 0, "input_tokens": 0, "output_tokens": 0}
            await score_record(record, Target(**record["target"]), Query(**record["query"]),
                               protocol, reader, {**options, "context_tokens": None, "model_timeout": 240})
            if record["context_truncated"] or record["reader_context"] != source_row["reader_context"]:
                raise ValueError(f"Replayed Reader context changed: {qid}")
            record["api_calls"] = [call for call in ledger.records if call["question_id"] == qid]
            record["evaluated_at"] = time.time()
            store.append(record)
            print(json.dumps({"done": len(store.records), "total": len(source_rows),
                              "question_id": qid, "status": record["status"], "grade": record["grade"],
                              "error": record.get("error")}), flush=True)

    try:
        async with asyncio.TaskGroup() as tasks:
            for _ in range(concurrency):
                tasks.create_task(consume())
        failures = sum(row["status"] != "succeeded" for row in store.records.values())
        atomic_json(output.with_suffix(".summary.json"), {"total": len(store.records), "errors": failures,
            "accuracy": sum(r["grade"] is True for r in store.records.values()) / len(source_rows),
            "new_retrieval_calls": 0, "new_embedding_calls": 0})
        return int(bool(failures))
    finally:
        await reader.aclose()
        ledger.close()
        store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", choices=list(PROFILES), default="deepseek")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--index-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--data-path", type=Path)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(replay(args.source, args.output, args.model, args.resume,
                                      args.index_root, args.cache_root, args.concurrency, args.data_path)))
