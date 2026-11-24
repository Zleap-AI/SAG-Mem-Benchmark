"""Final artifact audit: complete coverage, fixed identities, full evidence and request hashes."""
import argparse
import json
from pathlib import Path

from ..shared.runtime import file_digest, fingerprint, sha256
from .jsonl_io import read_jsonl


def verify(path, index_root=None, cache_root=None, data_path=None):
    manifest = json.loads(path.with_suffix(".jsonl.manifest.json").read_text())
    if fingerprint(manifest["identity"]) != manifest["run_id"]:
        raise ValueError("Manifest identity is corrupt")
    for source, expected in manifest["identity"]["data_checksums"].items():
        if file_digest(Path(data_path or source)) != expected:
            raise ValueError("Dataset no longer matches its manifest")
    config = manifest["identity"]["config"]
    index_root = Path(index_root or config["index_root"])
    cache_root = Path(cache_root or config.get("cache_root", index_root / "cache"))
    rows = {}
    for row in read_jsonl(path):
        if row["run_id"] != manifest["run_id"]:
            raise ValueError("Result run identity mismatch")
        rows[row["question_id"]] = row
    expected = {e["id"]: e for e in manifest["identity"]["questions"]}
    if set(rows) != set(expected):
        raise ValueError(f"Coverage mismatch: {len(rows)} / {len(expected)}")
    build_sources = {}
    dataset = manifest["identity"]["config"]["dataset"]
    for qid, row in rows.items():
        if row["status"] != "succeeded" or row["grade"] not in (True, False):
            raise ValueError(f"Question has no successful valid judge: {qid}")
        if row["source_history_sha256"] != expected[qid]["history_digest"]:
            raise ValueError(f"Source history changed: {qid}")
        if row["context_truncated"] or row["context"] != row["reader_context"]:
            raise ValueError(f"Reader context is incomplete: {qid}")
        if sha256(row["reader_context"]) != row["reader_context_sha256"]:
            raise ValueError(f"Reader context digest mismatch: {qid}")
        if fingerprint([item["content"] for item in row["evidence"]]) != row["source_content_sha256"]:
            raise ValueError(f"Raw evidence digest mismatch: {qid}")
        if row["build_id"] not in build_sources:
            base = index_root / config["namespace"] / row["build_id"]
            build = json.loads((base / "build.json").read_text())
            if file_digest(base / "sources.json") != build["sources_sha256"]:
                raise ValueError("Saved source mapping hash mismatch")
            build_sources[row["build_id"]] = json.loads((base / "sources.json").read_text())
        for item in row["evidence"]:
            source = build_sources[row["build_id"]][str(item["sag2_chunk_id"])]
            if any(item[key] != value for key, value in source.items()
                   if key != "indexed_content" or key in item):
                raise ValueError(f"Selected evidence differs from original stored chunk: {qid}")
        cache = json.loads((cache_root / "retrieval" / dataset / f"{row['retrieval_cache_key']}.json").read_text())
        if cache["sha256"] != fingerprint(cache["result"]) or cache["result"]["items"] != row["evidence"]:
            raise ValueError(f"Retrieval cache is corrupt or differs from results: {qid}")
        if any(call.get("usage", {}).get("prompt_tokens", 0) == 0
               for call in row["api_calls"] if call["status"] == "succeeded"):
            raise ValueError(f"A successful request has no input token accounting: {qid}")
    result = {"passed": True, "questions": len(rows), "source_builds": len(build_sources),
              "run_id": manifest["run_id"], "full_context_verified": True}
    path.with_suffix(".verified.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--index-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--data-path", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.result, args.index_root, args.cache_root, args.data_path)))
