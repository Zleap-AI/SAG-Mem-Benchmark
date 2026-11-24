"""LoCoMo multi-hop scan for K and H, then separate misses from wrong answers.

SAG2 accepts at most two expansion hops, so H=3 is recorded as unsupported.
K=60 uses B=60 because the runner requires K <= B. K=40 keeps B=50.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "reproduction/data/locomo10.json"
OUT = ROOT / "reproduction/experiments/main/results/locomo/multihop_scan"
PYTHON = sys.executable
CONFIGS = [
    {"name": "k20_h1", "candidate_k": 50, "max_hops": 1, "top_k": 20},
    {"name": "k40_h1", "candidate_k": 50, "max_hops": 1, "top_k": 40},
    {"name": "k60_h1", "candidate_k": 60, "max_hops": 1, "top_k": 60},
    {"name": "k20_h2", "candidate_k": 50, "max_hops": 2, "top_k": 20},
]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def multi_hop_ids() -> list[str]:
    ids = []
    for item in json.loads(DATA.read_text(encoding="utf-8")):
        sample = str(item["sample_id"])
        for position, qa in enumerate(item["qa"]):
            if int(qa["category"]) != 1:
                continue
            ids.append(str(qa.get("question_id") or f"{sample}:{position}"))
    if len(ids) != len(set(ids)):
        raise ValueError("LoCoMo multi-hop ids are not unique")
    return ids


def run_split(config: dict, questions_file: Path, output: Path) -> None:
    summary = output.with_suffix(".summary.json")
    expected = len(json.loads(questions_file.read_text(encoding="utf-8")))
    if summary.exists():
        saved = json.loads(summary.read_text(encoding="utf-8"))
        if saved.get("valid_grades") == expected and saved.get("errors") == 0:
            print(f"skip {config['name']}", flush=True)
            return
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "-u", str(ROOT / "reproduction/run.py"),
        "--method", "sag-mem", "--dataset", "locomo", "--model", "qwen",
        "--data-path", str(DATA),
        "--candidate-k", str(config["candidate_k"]),
        "--max-hops", str(config["max_hops"]),
        "--top-k", str(config["top_k"]),
        "--concurrency", "20", "--build-concurrency", "4",
        "--questions-file", str(questions_file),
        "--output", str(output), "--resume",
    ]
    log = output.parent / "run.log"
    print(" ".join(cmd), flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(cmd) + "\n")
        handle.flush()
        process = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
        completed = process.wait()
    if completed != 0:
        raise SystemExit(f"{config['name']} failed with exit {completed}; see {log}")


def classify(path: Path) -> dict:
    latest: dict[str, dict] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                latest[row["question_id"]] = row
    buckets = {"correct": 0, "not_recalled": 0, "recalled_but_wrong": 0, "ungraded": 0}
    examples = {key: [] for key in ("not_recalled", "recalled_but_wrong")}
    for row in latest.values():
        if row.get("status") != "succeeded" or not isinstance(row.get("grade"), bool):
            buckets["ungraded"] += 1
            continue
        coverage = row.get("message_coverage")
        if coverage is None:
            coverage = row.get("session_coverage")
        if row["grade"]:
            buckets["correct"] += 1
        elif not coverage:
            buckets["not_recalled"] += 1
            if len(examples["not_recalled"]) < 8:
                examples["not_recalled"].append(row["question_id"])
        else:
            buckets["recalled_but_wrong"] += 1
            if len(examples["recalled_but_wrong"]) < 8:
                examples["recalled_but_wrong"].append(row["question_id"])
    graded = buckets["correct"] + buckets["not_recalled"] + buckets["recalled_but_wrong"]
    return {"questions": len(latest), "graded": graded, "accuracy": buckets["correct"] / graded if graded else None,
            "buckets": buckets, "example_ids": examples,
            "rule": "message_coverage==0 is not recalled; coverage>0 and an incorrect grade is recalled but wrong"}


def main() -> None:
    ids = multi_hop_ids()
    questions = OUT / "multihop_ids.json"
    write_json(questions, ids)
    write_json(OUT / "unsupported.json", {
        "H": 3,
        "reason": "SAG2 rejects max_hops above 2, so the H=3 cell was not run",
    })
    print(f"locomo multi-hop questions={len(ids)}", flush=True)
    for config in CONFIGS:
        run_split(config, questions, OUT / config["name"] / "run.jsonl")
    report = {"H3": "unsupported", "runs": {}}
    for config in CONFIGS:
        summary_path = OUT / config["name"] / "run.summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        report["runs"][config["name"]] = {
            **config,
            "accuracy": summary.get("accuracy"),
            "correct": summary.get("total", 0) - summary.get("incorrect", 0),
            "valid_grades": summary.get("valid_grades"),
            "errors": summary.get("errors"),
            "wall_seconds": summary.get("wall_seconds"),
            "message_coverage": summary.get("message_coverage"),
            "session_coverage": summary.get("session_coverage"),
            "answer_chunk_recall": summary.get("answer_chunk_recall"),
        }
    report["baseline_failures"] = classify(OUT / "k20_h1" / "run.jsonl")
    write_json(OUT / "report.json", report)
    print(json.dumps(report["baseline_failures"], ensure_ascii=False), flush=True)
    print(json.dumps({name: row["accuracy"] for name, row in report["runs"].items()}), flush=True)


if __name__ == "__main__":
    main()
