"""Three real-input checks, including h0 equivalence with a fake local scorer.

These are interface tests, not new JEV answers or accuracy measurements.
"""
import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.backends.sag2_package import CorpusIndex
from src.data.loader import iter_dataset
from zleap.sag2._algorithm import search_sag2

from reproduction.experiments.ablations.prepare import (
    ROOT,
    _corpus,
    _lme_history,
    _locomo_history,
    chunk_width,
    chunking,
    file_hash,
    fingerprint,
    h0_items,
    normalize_passage,
    prepare,
    render,
    shared_chunks,
)


class FixedScorer:
    async def score(self, query, candidates, evidence):
        if evidence:
            raise ValueError("The h0 test must not receive expansion evidence")
        return ([{**c, "jev_score": 1 / (int(c["id"]) % 13 + 1), "jev_model": "offline-fixture"}
                 for c in candidates],
                {"requests": 0, "judgments": len(candidates), "input_tokens": 0, "output_tokens": 0,
                 "batches": [{"candidate_count": len(candidates), "input_tokens": 0, "output_tokens": 0}]})


async def validate(args):
    cases = []
    for dataset, limit in (("locomo", 2), ("longmemeval", 1)):
        path = args.locomo_source if dataset == "locomo" else args.longmemeval_source
        manifest = json.loads(path.with_suffix(".jsonl.manifest.json").read_text())
        cap = manifest["identity"]["build_dependencies"]["chunking"].get("messages_per_chunk")
        chunker = chunking if cap is None else chunk_width
        examples = {e["id"]: e for e in manifest["identity"]["questions"]}
        rows = [json.loads(line) for line in path.open()][:limit]
        data_path = args.data_root / ("locomo10.json" if dataset == "locomo" else "longmemeval_s_cleaned.json")
        references = {examples[row["question_id"]]["history_ref"]["index"] for row in rows}
        data = {i: value for i, value in enumerate(iter_dataset(data_path)) if i in references}
        with tempfile.TemporaryDirectory(prefix="offline-index-") as temporary:
            for number, row in enumerate(rows):
                context = render(row["evidence"])
                assert context["raw"] == "\n\n".join(i["content"] for i in row["evidence"])
                assert context["main_cached"] == row["context"] == row["reader_context"]
                raw = data[examples[row["question_id"]]["history_ref"]["index"]]
                history = _locomo_history(raw, "text") if dataset == "locomo" else _lme_history(raw)
                chunks = shared_chunks(raw, dataset, cap)
                corpus, mapping = _corpus(chunks, normalize_passage)
                index = CorpusIndex(Path(temporary) / f"{number}.sqlite", corpus, history.digest)
                try:
                    result = await search_sag2(index.query_view(), FixedScorer(), row["question"],
                                               candidate_k=50, seed_k=10, max_hops=0, expansion_term_limit=20)
                    source = {"backend_stats": {"sag2_trace": result["trace"], "jev_usage": result["jev_usage"]}}
                    before = fingerprint(source)
                    items, usage = h0_items(source, mapping, 50, 20)
                    assert [i["sag2_chunk_id"] for i in items] == [c["id"] for c in result["candidates"][:20]]
                    assert fingerprint(source) == before
                    assert all(i["content"] == mapping[str(c["document_id"])]["content"]
                               for i, c in zip(items, result["candidates"][:20], strict=True))
                    assert usage["input_tokens"] == 0
                    # Exercise the complete no-expansion input pipeline. This parent
                    # is explicitly synthetic; only its conversation and question are real.
                    fixture = Path(temporary) / f"fixture-{number}"
                    fixture.mkdir()
                    data_file = fixture / "data.json"
                    data_file.write_text(json.dumps([raw]))
                    build_id = "offline-build"
                    build_dir = fixture / build_id
                    build_dir.mkdir()
                    mapping_path = build_dir / "sources.json"
                    mapping_path.write_text(json.dumps(mapping))
                    chunk_config = {"messages_per_chunk": cap,
                                    "implementation_sha256": file_hash(Path(chunker.__file__))}
                    (build_dir / "build.json").write_text(json.dumps({
                        "build_id": build_id, "sources_sha256": file_hash(mapping_path),
                        "dependencies": {"chunking": chunk_config}}))
                    source_file = fixture / "parent.jsonl"
                    identity = {"config": {"backend": "sag-mem", "dataset": dataset,
                        "context_tokens": None, "top_k": 20, "text_mode": "text"},
                        "build_dependencies": {"chunking": chunk_config},
                        "questions": [{"id": row["question_id"], "history_ref": {"path": str(data_file), "index": 0}}],
                        "data_checksums": {str(data_file): file_hash(data_file)},
                        "backend_runtime": {"retrieval": {"candidate_k": 50}, "source": {
                            "version": "0.4.0", "sha256": {"zleap/sag2/_algorithm.py":
                                "0334f38b2e2e634c271f8dc355cd2175ff2fa7a68d6a451d955c28bed8b82bae"}}}}
                    parent = {**row, "build_id": build_id, "run_id": fingerprint(identity),
                              "evidence": items, "backend_stats": source["backend_stats"],
                              "context": render(items)["main_cached"], "reader_context": render(items)["main_cached"],
                              "hypothesis": "offline interface fixture", "grade": False}
                    source_file.write_text(json.dumps(parent) + "\n")
                    source_file.with_suffix(".jsonl.manifest.json").write_text(json.dumps({
                        "run_id": parent["run_id"], "identity": identity}))
                    output = fixture / "prepared"
                    prepare(SimpleNamespace(source=[source_file], output=output, method="sag-mem",
                        dataset=dataset, scope="sample", limit=1, no_expansion=True,
                        sources_root=fixture, data_root=fixture))
                    rebuilt = json.loads((output / "no_expansion.jsonl").read_text())
                    assert rebuilt["evidence"] == items
                    assert rebuilt["context"] == parent["context"]
                finally:
                    index.close()
                cases.append({"dataset": dataset, "question_id": row["question_id"],
                              "real_cached_time_context_exact": True,
                              "h0_matches_original_max_hops_zero": True,
                              "full_no_expansion_preparation_passed": True,
                              "source_content_unchanged": True,
                              "h0_scoring": "deterministic_offline_fixture_not_real_JEV",
                              "new_model_calls": 0})
    return {"status": "passed", "real_questions": len(cases), "formal_result": False, "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locomo-source", type=Path, required=True)
    parser.add_argument("--longmemeval-source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT.parents[1] / "data")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = asyncio.run(validate(args))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
