"""Build offline Reader ablations from frozen SAG-Mem / Lite result files.

No network clients, credentials, inference, or production-code writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPRO_ROOT = ROOT.parents[1]
REPO = REPRO_ROOT.parent
RULES = REPO / "src/harness/relative_time.py"
EXPECTED = {"all": {"locomo": 1986, "longmemeval": 500},
            "temporal": {"locomo": 321, "longmemeval": 133}}
CONDITIONS = ("raw", "normalized", "main_cached")

# Import the repository implementation without loading credentials for offline preparation.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
sys.path.insert(0, str(REPRO_ROOT.parent))
from src.backends.sag2_adapter import _corpus
from src.backends.sag2_chunk_time import annotate_chunk, format_message_date
from src.data.loader import _lme_history, _locomo_history
from zleap.sag2._algorithm import diversity_tokens, rank_dpp
from zleap.sag2.lance import normalize_passage

from reproduction.experiments.shared import chunk_width, chunking


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def comparison_profile(profile):
    """Compare model semantics while retaining original transport metadata."""
    result = deepcopy(profile)
    for role in ("reader_service", "judge_service"):
        service = result.get(role)
        if service is None:
            continue
        service.pop("base_url", None)
        provider = service.get("extra_body", {}).get("provider", {})
        if provider.get("allow_fallbacks") is False and provider.get("order") == ["DeepInfra"]:
            provider.pop("order")
            provider.setdefault("only", ["DeepInfra"])
    jev = (result.get("retrieval") or {}).get("jev", {})
    jev.pop("endpoint", None)
    return result


def input_identity(path, manifest, source_map=None):
    """Recover retrieval provenance omitted by a native Reader/Judge replay."""
    identity = deepcopy(manifest["identity"])
    lineage = {}
    if identity.get("evaluation_mode") == "reader_judge_replay":
        source_name = manifest["source_path"]
        source = Path((source_map or {}).get(source_name, source_name))
        if not source.is_absolute():
            source = path.parent / source
        source_manifest_path = source.with_suffix(source.suffix + ".manifest.json")
        if file_hash(source) != identity["source_sha256"]:
            raise ValueError("Native replay source checksum mismatch")
        original = json.loads(source_manifest_path.read_text())
        if original["run_id"] != identity["source_run"] or fingerprint(original["identity"]) != original["run_id"]:
            raise ValueError("Native replay source identity mismatch")
        inherited, lineage = input_identity(source_manifest_path, original, source_map)
        for field in ("build_dependencies", "questions", "data_checksums", "backend_runtime", "dataset_code", "backend_code"):
            if field in identity and identity[field] != inherited[field]:
                raise ValueError("Native replay changed retrieval provenance")
            identity[field] = inherited[field]
        lineage.update({str(source): file_hash(source), str(source_manifest_path): file_hash(source_manifest_path)})
    return identity, lineage


def render(items):
    raw_chunks, normalized_chunks, annotated_chunks = [], [], []
    for item in items:
        remaining = item["content"]
        originals, normalized = [], []
        for span in item["message_spans"]:
            if not remaining:  # Identical indexed text can have additional provenance spans.
                break
            header = re.match(r"([^\r\n]+?) \(date: ([^\r\n]+?)\): ", remaining)
            if header is None:
                raise ValueError("Evidence lacks the original message header")
            length = span["end"] - span["start"]
            end = header.end() + length
            if length < 0 or end > len(remaining):
                raise ValueError("Evidence span exceeds the original text")
            body = remaining[header.end():end]
            new_header = f"{header[1]}: [Message date: {format_message_date(header[2])}]\n"
            originals.append(remaining[:end])
            normalized.append(new_header + body)
            remaining = remaining[end:]
            if remaining:
                if not remaining.startswith("\n"):
                    raise ValueError("Invalid original message boundary")
                remaining = remaining[1:]
        if remaining or "\n".join(originals) != item["content"]:
            raise ValueError("Evidence reconstruction changed original content")
        raw_chunks.append(item["content"])
        normalized_chunks.append("\n".join(normalized))
        annotated_chunks.append(annotate_chunk(item)[0])
    return dict(zip(CONDITIONS, ("\n\n".join(x) for x in
                                (raw_chunks, normalized_chunks, annotated_chunks)), strict=True))


def h0_items(row, mapping, candidate_k, top_k):
    trace = row["backend_stats"]["sag2_trace"]
    initial = [s for s in trace if s["stage"] == "bm25" and s["hop"] == 0]
    judgments = [s for s in trace if s["stage"] == "jev" and s["hop"] == 0]
    if len(initial) != 1 or len(judgments) != 1:
        raise ValueError("Expected exactly one complete h0 BM25/JEV trace")
    initial, judgments = initial[0], judgments[0]
    scores = {s["id"]: s["score"] for s in judgments["scores"]}
    ids = initial["candidate_ids"]
    if len(ids) != len(set(ids)) or len(judgments["scores"]) != len(ids) or set(ids) != set(scores):
        raise ValueError("h0 scores do not cover the initial candidate IDs exactly")
    candidates = []
    for rank, (identifier, document) in enumerate(zip(ids, initial["candidate_document_ids"], strict=True)):
        source = mapping[str(document)]
        candidates.append({"id": identifier, "document_id": document,
                           "content": source["indexed_content"], "jev_score": scores[identifier],
                           "fusion_score": 1 / (61 + rank)})
    candidates.sort(key=lambda c: (-c["jev_score"], -c["fusion_score"], c["id"]))
    candidates = candidates[:candidate_k]
    frequencies = Counter()
    for source in mapping.values():
        frequencies.update(set(diversity_tokens(source["indexed_content"])))
    idf = {term: math.log((1 + len(mapping)) / (1 + count)) + 1
           for term, count in frequencies.items()}
    selected = (rank_dpp(candidates[:10], idf) + candidates[10:])[:top_k]
    items = [{**{k: v for k, v in mapping[str(c["document_id"])].items() if k != "indexed_content"},
              "score": c["jev_score"], "kind": "chunk", "sag2_chunk_id": c["id"], "hop": 0}
             for c in selected]
    first_batches = [b for b in row["backend_stats"]["jev_usage"]["batches"] if b["hop"] == 0]
    usage = {key: (sum(b[key] for b in first_batches) if all(b.get(key) is not None for b in first_batches)
                   else None) for key in ("input_tokens", "output_tokens")}
    usage.update(judgments=sum(b["candidate_count"] for b in first_batches),
                 requests=len(first_batches), batches=first_batches)
    return items, usage


def shared_chunks(raw, dataset, messages_per_chunk=None):
    history = _locomo_history(raw, "text") if dataset == "locomo" else _lme_history(raw)
    if messages_per_chunk is not None:
        return chunk_width.chunk_history(history, messages_per_chunk)
    return chunking.chunk_history(history)


def verify_mapping(mapping, chunks):
    _, expected = _corpus(chunks, normalize_passage)
    if mapping != expected:
        raise ValueError("Source mapping differs from the original shared chunker and corpus builder")


def prepare(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory")
    positions, parents, resolved_parents, hashes, manifest_hashes, lineage_hashes, declared_order = {}, {}, {}, {}, {}, {}, []
    profile = None
    for path in args.source:
        path = path.resolve()
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        manifest = json.loads(manifest_path.read_text())
        manifest_hashes[str(manifest_path)] = file_hash(manifest_path)
        if fingerprint(manifest["identity"]) != manifest["run_id"]:
            raise ValueError("Parent manifest identity hash mismatch")
        effective, lineage = input_identity(manifest_path, manifest, getattr(args, "lineage_map", None))
        resolved_parents[path] = effective
        lineage_hashes.update(lineage)
        config = effective["config"]
        if config["dataset"] != args.dataset or config.get("context_tokens") is not None:
            raise ValueError("Expected matching dataset and complete, untruncated parent contexts")
        if config.get("text_mode", "text") != "text" or config.get("history_prefix") is not None:
            raise ValueError("Only full text-mode histories are supported")
        allowed_backends = ({"sag-mem", "sag2-chunk-time"} if args.method == "sag-mem"
                            else {"sag_mem_lite", "sag_mem_lite_parallel", "sag-mem-lite", "sag2_embedding"})
        if config["backend"] not in allowed_backends:
            raise ValueError("Method label does not match the parent backend")
        declared = manifest.get("question_order") or [q["id"] for q in effective["questions"]]
        declared_order.extend(declared)
        if args.scope != "sample":
            deps = effective["build_dependencies"]
            cap = deps["chunking"].get("messages_per_chunk")
            shared = Path((chunking if cap is None else chunk_width).__file__)
            policy = chunking.POLICY if cap is None else f"{cap}_messages_soft512_atomic_v1"
            chunker_hash = deps["chunking"].get("implementation_sha256") or deps["implementation"].get("shared_chunker")
            if deps["chunking"]["policy"] != policy or chunker_hash != file_hash(shared):
                raise ValueError("Parent chunking configuration does not match the current implementation")
        current_profile = {key: effective.get(key) for key in
                           ("reader_service", "judge_service", "protocol_code", "prompts_code")}
        current_profile["retrieval"] = effective.get("backend_runtime")
        current_profile["context_options"] = {key: config.get(key) for key in
                                               ("top_k", "question_date_text", "text_mode")}
        if profile is not None and comparison_profile(profile) != comparison_profile(current_profile):
            raise ValueError("Parent model or evaluation profiles differ")
        if profile is None:
            profile = current_profile
        hashes[str(path)] = file_hash(path)
        parents[path] = manifest
        with path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    raise ValueError("Parent contains an unfinished JSONL line")
                row = json.loads(line)
                if row["run_id"] != manifest["run_id"]:
                    raise ValueError("Parent row run_id mismatch")
                qid = row["question_id"]
                if qid not in declared:
                    raise ValueError("Parent contains a question outside its declared selection")
                if qid in positions and positions[qid][0] != path:
                    raise ValueError("Question occurs in different parent files")
                positions[qid] = (path, offset, str(row["question_type"]))
    selected = [qid for qid in declared_order if qid in positions and (args.scope != "temporal"
                or positions[qid][2] == ("2" if args.dataset == "locomo" else "temporal-reasoning"))]
    if len(selected) != len(set(selected)):
        raise ValueError("Parents declare overlapping questions")
    if args.scope == "sample":
        if not args.limit or args.limit < 1:
            raise ValueError("sample scope requires a positive --limit")
        selected = selected[:args.limit]
    elif args.limit is not None:
        raise ValueError("--limit is only permitted with --scope sample")
    elif len(selected) != EXPECTED[args.scope][args.dataset]:
        raise ValueError(f"Incomplete {args.scope} coverage: {len(selected)} questions")
    if not selected:
        raise ValueError("No questions selected")
    if args.no_expansion and (args.method != "sag-mem" or not args.sources_root or not args.data_root):
        raise ValueError("--no-expansion requires sag-mem, --sources-root and --data-root")
    only_h0 = getattr(args, "only_no_expansion", False)
    if only_h0 and not args.no_expansion:
        raise ValueError("--only-no-expansion requires --no-expansion")
    first_parent = next(iter(resolved_parents.values()))
    messages_per_chunk = first_parent["build_dependencies"]["chunking"].get("messages_per_chunk")
    shared = Path((chunking if messages_per_chunk is None else chunk_width).__file__)
    identity = {"method": args.method, "dataset": args.dataset, "scope": args.scope,
                "source_files": hashes, "source_manifests": manifest_hashes,
                "native_replay_lineage": lineage_hashes,
                "source_runs": [m["run_id"] for m in parents.values()],
                "transport_comparison": "service_transport_excluded_from_semantic_comparison_v1",
                "question_order": selected, "script_sha256": file_hash(__file__),
                "chunker_sha256": file_hash(shared), "messages_per_chunk": messages_per_chunk,
                "rules_sha256": file_hash(RULES),
                "formatter_sha256": file_hash(REPO / "src/backends/sag2_chunk_time.py"),
                "corpus_builder_sha256": file_hash(REPO / "src/backends/sag2_adapter.py"),
                "sag2_algorithm_sha256": file_hash(rank_dpp.__code__.co_filename),
                "no_expansion": args.no_expansion, "new_model_calls": 0}
    source_config = next(iter(resolved_parents.values()))["config"]
    identity.update(profile)
    identity["config"] = {**source_config, "backend": "saved_context", "retrieval_only": True,
                          "context_tokens": None}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ablations-", dir=output.parent) as temporary:
        staging = Path(temporary)
        names = ("no_expansion",) if only_h0 else ((*CONDITIONS, "no_expansion") if args.no_expansion else CONDITIONS)
        streams = {name: (staging / f"{name}.jsonl").open("w") for name in names}
        validations, dependencies = [], {}
        data_cache, mapping_cache = {}, {}
        try:
            for qid in selected:
                path, offset, category = positions[qid]
                with path.open("rb") as stream:
                    stream.seek(offset)
                    row = json.loads(stream.readline())
                effective_parent = resolved_parents[path]
                if row["status"] != "succeeded" or type(row.get("grade")) is not bool:
                    raise ValueError(f"Parent has no successful cached main answer: {qid}")
                if row.get("retrieval_only") or row.get("context_truncated"):
                    raise ValueError("Parent answer is retrieval-only or truncated")
                contexts = render(row["evidence"])
                if contexts["main_cached"] != row["context"] or row["reader_context"] != row["context"]:
                    raise ValueError(f"Parent main context does not match frozen time formatter: {qid}")
                base = {key: row[key] for key in ("question_id", "corpus_id", "build_id", "query", "target",
                                                 "gold_answer", "is_abstention", "question_type")}
                base.update(question=row["question"], protocol=row["protocol"], backend="saved_context",
                            source_run_id=row["run_id"], method=args.method, dataset=args.dataset,
                            source_main_grade=row["grade"],
                            scope=args.scope, evidence=row["evidence"], source_context_sha256=fingerprint(row["context"]),
                            selection_sha256=fingerprint(row["evidence"]),
                            parent_inference_usage={key: row.get(key) for key in
                                ("reader_prompt_tokens", "reader_completion_tokens", "grader_prompt_tokens",
                                 "grader_completion_tokens")},
                            method_retrieval_stats=row["backend_stats"],
                            preparation_usage={"reader_calls": 0, "judge_calls": 0, "jev_calls": 0, "embedding_calls": 0})
                payloads = {}
                for name in CONDITIONS:
                    payloads[name] = {**base, "condition": name, "context": contexts[name],
                        "reader_context": contexts[name], "status": "succeeded",
                        "retrieval_only": name != "main_cached", "grade": row["grade"] if name == "main_cached" else None,
                        "answer_origin": "parent_main_cache" if name == "main_cached" else "reader_replay_pending",
                        "cost_scope": "parent_main_cached" if name == "main_cached" else "new_reader_pending"}
                    if name == "main_cached":
                        payloads[name]["hypothesis"] = row["hypothesis"]
                if args.no_expansion:
                    config = effective_parent["config"]
                    runtime = effective_parent["backend_runtime"]
                    if runtime["source"]["version"] != "0.4.0":
                        raise ValueError("DPP reconstruction is pinned to zleap-sag2 0.4.0")
                    expected_algorithm = runtime["source"]["sha256"]["zleap/sag2/_algorithm.py"]
                    if expected_algorithm != "0334f38b2e2e634c271f8dc355cd2175ff2fa7a68d6a451d955c28bed8b82bae":
                        raise ValueError("Parent algorithm differs from frozen DPP source")
                    if expected_algorithm != identity["sag2_algorithm_sha256"]:
                        raise ValueError("Installed DPP differs from the parent algorithm")
                    build = row["build_id"]
                    if build not in mapping_cache:
                        paths = list(args.sources_root.glob(f"**/{build}/sources.json"))
                        if len(paths) != 1:
                            raise ValueError(f"Expected one source map for build {build}, found {len(paths)}")
                        mapping_path = paths[0]
                        build_manifest = json.loads(mapping_path.with_name("build.json").read_text())
                        if build_manifest["build_id"] != build:
                            raise ValueError("Source map belongs to a different build")
                        if file_hash(mapping_path) != build_manifest["sources_sha256"]:
                            raise ValueError("Source map checksum mismatch")
                        expected_chunker = build_manifest["dependencies"]["chunking"]["implementation_sha256"]
                        if expected_chunker != identity["chunker_sha256"]:
                            raise ValueError("Parent build did not use the current shared chunker")
                        example = next(e for e in effective_parent["questions"] if e["id"] == qid)
                        reference = example["history_ref"]
                        data_path = args.data_root / Path(reference["path"]).name
                        if data_path not in data_cache:
                            data_hash = file_hash(data_path)
                            if data_hash != effective_parent["data_checksums"][reference["path"]]:
                                raise ValueError("Raw dataset checksum mismatch")
                            data_cache.clear()  # Keep only the current dataset shard in memory.
                            data_cache[data_path] = json.loads(data_path.read_text())
                            dependencies[str(data_path)] = data_hash
                        mapping = json.loads(mapping_path.read_text())
                        cap = effective_parent["build_dependencies"]["chunking"].get("messages_per_chunk")
                        verify_mapping(mapping, shared_chunks(data_cache[data_path][reference["index"]], args.dataset, cap))
                        mapping_cache.clear()
                        mapping_cache[build] = mapping
                        dependencies[str(mapping_path)] = file_hash(mapping_path)
                        dependencies[str(mapping_path.with_name("build.json"))] = file_hash(mapping_path.with_name("build.json"))
                    items, usage = h0_items(row, mapping_cache[build], runtime["retrieval"]["candidate_k"], config["top_k"])
                    context = render(items)["main_cached"]
                    payloads["no_expansion"] = {**base, "condition": "no_expansion", "evidence": items,
                        "selection_sha256": fingerprint(items), "context": context, "reader_context": context,
                        "status": "succeeded", "retrieval_only": True, "grade": None,
                        "answer_origin": "reader_replay_pending", "cost_scope": "cached_h0_usage_plus_new_reader_pending",
                        "method_retrieval_stats": {"jev_usage": usage, "max_hops": 0,
                                                   "reconstructed_from_parent": True}}
                for name in names:
                    payload = payloads[name]
                    streams[name].write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
                validations.append({"question_id": qid, "category": category,
                    "selection_sha256": base["selection_sha256"], "main_exact_match": True,
                    "context_sha256": {name: fingerprint(value["context"]) for name, value in payloads.items()},
                    "selected_ids": {name: [i["sag2_chunk_id"] for i in value["evidence"]] for name, value in payloads.items()}})
        finally:
            for stream in streams.values():
                stream.close()
        for path, checksum in {**hashes, **manifest_hashes, **lineage_hashes, **dependencies}.items():
            if file_hash(path) != checksum:
                raise ValueError("Parent file changed during preparation; use a completed snapshot")
        identity["reconstruction_dependencies"] = dependencies
        run_id = fingerprint(identity)
        for name in names:
            condition_identity = {**identity, "condition": name}
            condition_run = fingerprint(condition_identity)
            original = staging / f"{name}.jsonl"
            rewritten = staging / f".{name}.jsonl"
            with original.open() as source, rewritten.open("w") as target:
                for line in source:
                    row = json.loads(line)
                    row["run_id"] = condition_run
                    target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            rewritten.replace(original)
            meta = {"identity": condition_identity, "run_id": condition_run,
                    "question_order": selected, "condition": name,
                    "parent_manifests": {str(p): m for p, m in parents.items()}}
            (staging / f"{name}.jsonl.manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        report = {"run_id": run_id, "method": args.method, "dataset": args.dataset, "scope": args.scope,
                  "questions": len(selected), "coverage": "sample_only" if args.scope == "sample" else "complete",
                  "formal_new_version_result": False,
                  "conditions": list(names), "new_model_calls": 0, "checks": validations}
        (staging / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        staging.rename(output)
    return {k: v for k, v in report.items() if k != "checks"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--method", choices=("sag-mem", "sag-mem-lite"), required=True)
    parser.add_argument("--dataset", choices=("locomo", "longmemeval"), required=True)
    parser.add_argument("--scope", choices=("all", "temporal", "sample"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-expansion", action="store_true")
    parser.add_argument("--only-no-expansion", action="store_true")
    parser.add_argument("--sources-root", type=Path, default=ROOT.parent / "main/retained_sources")
    parser.add_argument("--data-root", type=Path, default=REPRO_ROOT / "data")
    parser.add_argument("--lineage-map", type=lambda p: json.loads(Path(p).read_text()),
                        help="Optional JSON mapping of archived source paths to local files")
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
