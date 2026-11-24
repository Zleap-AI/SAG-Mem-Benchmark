"""Canonical method resources are separate from newly billed API requests."""
import argparse
import fcntl
import json
from pathlib import Path

from tokenizers import Tokenizer

from reproduction.analysis.costs import response_usage, token_cost
from ..shared.runtime import file_digest, fingerprint
from .backend import TOKENIZERS
from .jsonl_io import read_jsonl
from .summarize import summarize


def canonical_summary(path, index_root=None, cache_root=None, tokenizer_path=None):
    manifest = json.loads(path.with_suffix(".jsonl.manifest.json").read_text())
    config = manifest["identity"]["config"]
    index_root = Path(index_root or config["index_root"])
    cache_root = Path(cache_root or config.get("cache_root", index_root / "cache"))
    rows = {row["question_id"]: row for row in read_jsonl(path)}
    if len(rows) != len(manifest["identity"]["questions"]) or any(
        row["status"] != "succeeded" for row in rows.values()
    ):
        raise ValueError("Canonical summary requires complete successful question coverage")
    embedding = manifest["identity"]["build_dependencies"]["embedding"]
    window = embedding["max_input_tokens"]
    tokenizer_path = Path(tokenizer_path or TOKENIZERS[embedding["profile"]])
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()

    def input_length(text):
        count = len(tokenizer.encode(text).ids)
        if count > window and embedding["overflow"] == "prefix":
            from src.backends.embedding_tokenizer import EmbeddingTokenizer
            text = EmbeddingTokenizer("BAAI/bge-large-en-v1.5").fitting_prefix(text)
            count = len(tokenizer.encode(text).ids)
        return count

    histories = {}
    memo = path.parent / "logical_counts"
    memo.mkdir(exist_ok=True)
    for row in rows.values():
        history_hash = row["source_history_sha256"]
        if history_hash in histories:
            continue
        directory = index_root / config["namespace"] / row["build_id"]
        sources = directory / "sources.json"
        source_hash = file_digest(sources)
        key = fingerprint({"sources_sha256": source_hash, "tokenizer_sha256": file_digest(tokenizer_path),
                           "embedding": embedding, "statistics_version": 3})
        stored = memo / f"{key}.json"
        if stored.exists():
            count = json.loads(stored.read_text())
        else:
            mapping = json.loads(sources.read_text())
            lengths = [input_length(item["content"]) for item in mapping.values()]
            count = {"logical_embedding_build_tokens": sum(lengths),
                     "documents": len(mapping), "source_content_sha256": source_hash,
                     "max_document_embedding_input_tokens": max(lengths, default=0),
                     "documents_over_embedding_window": sum(length > window for length in lengths)}
            stored.write_text(json.dumps(count))
        index = directory / "sag2.sqlite"
        archived = directory / "index_archive.json"
        with (directory / ".archive.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            if index.exists():
                index_bytes = index.stat().st_size
            else:
                archive = json.loads(archived.read_text())
                build = json.loads((directory / "build.json").read_text())
                if not archive["decompression_verified"] or archive["original_sha256"] != build["index_sha256"]:
                    raise ValueError("Archived index metadata does not match original build")
                index_bytes = archive["original_bytes"]
            is_archived = archived.exists()
            stored_bytes = sum(p.stat().st_size for p in [index, directory / "sag2.sqlite.gz"] if p.exists())
        histories[history_hash] = {"history_sha256": history_hash, "build_id": row["build_id"], **count,
                                  "index_bytes": index_bytes, "index_archived": is_archived,
                                  "index_stored_bytes": stored_bytes,
                                  "source_metadata_bytes": sources.stat().st_size}
    instruction = manifest["identity"]["backend_runtime"]["query_instruction"]
    query_tokens = {qid: input_length(f"Instruct: {instruction}\nQuery:{row['question']}")
                    for qid, row in rows.items()}
    actual = summarize(path)
    price = actual["standardized_price_table"][embedding["model"]]["input_per_token"]
    total_build = sum(item["logical_embedding_build_tokens"] for item in histories.values())
    total_query = sum(query_tokens.values())
    reader_price = actual["standardized_price_table"][manifest["identity"]["config"]["reader_model"]]
    judge_price = actual["standardized_price_table"][manifest["identity"]["config"]["judge_model"]]
    reader_cost = sum(token_cost(response_usage(row, "reader"), reader_price["input_per_token"],
                                reader_price["output_per_token"], reader_price.get("cached_input_per_token"))
                      for row in rows.values())
    judge_cost = sum(token_cost(response_usage(row, "judge"), judge_price["input_per_token"],
                               judge_price["output_per_token"], judge_price.get("cached_input_per_token"))
                     for row in rows.values())
    result = {
        "run_id": manifest["run_id"], "questions": len(rows), "unique_histories": len(histories),
        "cost_accounting_version": "operational_v2", "embedding_model": embedding["model"],
        "embedding_documents": sum(item["documents"] for item in histories.values()),
        "max_document_embedding_input_tokens": max(
            (item["max_document_embedding_input_tokens"] for item in histories.values()), default=0),
        "embedding_input_window_tokens": window,
        "documents_over_embedding_window": sum(item["documents_over_embedding_window"]
                                              for item in histories.values()),
        "embedding_tokenizer_sha256": file_digest(tokenizer_path),
        "logical_embedding_build_tokens": total_build,
        "logical_embedding_build_standardized_usd": total_build * price,
        "canonical_query_embedding_tokens": total_query,
        "canonical_query_embedding_standardized_usd": total_query * price,
        "canonical_reader_standardized_usd": reader_cost,
        "canonical_judge_standardized_usd": judge_cost,
        "canonical_total_standardized_usd": (total_build + total_query) * price + reader_cost,
        "index_bytes": sum(item["index_bytes"] for item in histories.values()),
        "index_stored_bytes": sum(item["index_stored_bytes"] for item in histories.values()),
        "archived_indexes": sum(item["index_archived"] for item in histories.values()),
        "source_metadata_bytes": sum(item["source_metadata_bytes"] for item in histories.values()),
        "shared_embedding_cache_bytes": ((cache_root / "embedding_cache.sqlite").stat().st_size
                                         if (cache_root / "embedding_cache.sqlite").exists() else None),
        "shared_cache_allocation": "global cache size only; not allocated to this method or dataset",
        "actual_incurred_api_cost_usd_recorded": actual["actual_cost_usd"],
        "actual_cost_missing_requests": actual["missing_cost_requests"],
        "incurred_standardized_api_cost_usd": actual["standardized_cost_usd"],
        "actual_api_request_tokens_by_stage": actual["requests"],
        "final_reader_prompt_tokens": sum(row["reader_prompt_tokens"] for row in rows.values()),
        "final_reader_completion_tokens": sum(row["reader_completion_tokens"] for row in rows.values()),
        "final_judge_prompt_tokens": sum(row["grader_prompt_tokens"] for row in rows.values()),
        "final_judge_completion_tokens": sum(row["grader_completion_tokens"] for row in rows.values()),
        "canonical_cost_rule": "Count each unique history once and every question once, including cache hits and DS replay",
        "canonical_cost_scope": "Embedding construction, query embedding and Reader; excludes offline Judge",
        "cache_accounting": "Use recorded prompt-cache tokens; price uncached when cache metadata is unavailable",
        "actual_cost_scope": "This run's complete request ledger including retries; prior attempts are separate and summed in STATUS",
        "histories": list(histories.values()), "query_tokens": query_tokens,
    }
    path.with_suffix(".resources.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return {key: value for key, value in result.items() if key not in {"histories", "query_tokens"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--index-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    args = parser.parse_args()
    with args.result.with_suffix(".resources.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        print(json.dumps(canonical_summary(args.result, args.index_root, args.cache_root, args.tokenizer_path), ensure_ascii=False, indent=2))
