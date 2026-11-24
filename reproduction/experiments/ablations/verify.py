"""Verify prepared conditions and report cached-main accuracy separately."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from reproduction.experiments.ablations.prepare import CONDITIONS, EXPECTED, fingerprint


def verify(paths):
    results = []
    method_questions = {}
    for directory in paths:
        report = json.loads((directory / "validation.json").read_text())
        all_rows = {}
        for name in report["conditions"]:
            path = directory / f"{name}.jsonl"
            manifest = json.loads(path.with_suffix(".jsonl.manifest.json").read_text())
            if fingerprint(manifest["identity"]) != manifest["run_id"]:
                raise ValueError("Prepared manifest fingerprint mismatch")
            rows = {}
            for line in path.open():
                row = json.loads(line)
                if row["question_id"] in rows or row["run_id"] != manifest["run_id"]:
                    raise ValueError("Duplicate question or wrong prepared run_id")
                if any(row["preparation_usage"].values()):
                    raise ValueError("Offline preparation unexpectedly reports model calls")
                if name != "main_cached" and (row["grade"] is not None or "hypothesis" in row):
                    raise ValueError("Pending replay contains a cached grade or answer")
                if name == "main_cached" and row["answer_origin"] != "parent_main_cache":
                    raise ValueError("Cached main answer not labelled")
                rows[row["question_id"]] = row
            if list(rows) != manifest["question_order"]:
                raise ValueError("Prepared question order differs from manifest")
            all_rows[name] = rows
        first = all_rows["raw"]
        if any(set(rows) != set(first) for rows in all_rows.values()):
            raise ValueError("Condition question sets differ")
        for qid, row in first.items():
            for name in CONDITIONS:
                other = all_rows[name][qid]
                if other["evidence"] != row["evidence"] or other["selection_sha256"] != row["selection_sha256"]:
                    raise ValueError("Time conditions changed retrieval evidence or order")
            if row["context"] != "\n\n".join(i["content"] for i in row["evidence"]):
                raise ValueError("Raw condition differs from original records")
        scope, dataset, method = (report[k] for k in ("scope", "dataset", "method"))
        if scope in EXPECTED and len(first) != EXPECTED[scope][dataset]:
            raise ValueError("Prepared coverage is incomplete")
        key = (method, dataset, scope)
        if key in method_questions:
            raise ValueError("Duplicate method/dataset/scope inputs")
        method_questions[key] = set(first)
        main = all_rows["main_cached"]
        results.append({"method": method, "dataset": dataset, "scope": scope,
                        "questions": len(main), "coverage": report["coverage"],
                        "cached_main_correct": sum(r["grade"] is True for r in main.values()),
                        "cached_main_accuracy": sum(r["grade"] is True for r in main.values()) / len(main),
                        "raw_replay_accuracy": None, "normalized_replay_accuracy": None,
                        "no_expansion_replay_accuracy": None, "new_model_calls": 0,
                        "cost_status": "cached main usage retained; replay costs not yet measured"})
    for (_, dataset, scope), questions in method_questions.items():
        other = [(k, q) for k, q in method_questions.items() if k[1:] == (dataset, scope)]
        if any(q != questions for _, q in other):
            raise ValueError("SAG-Mem/Lite question selection differs")
    return {"offline_validation": "passed", "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.directories)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
