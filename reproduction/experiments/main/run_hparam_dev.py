"""Select SAG-Mem K, B and H on a frozen LongMemEval development split.

Chunks have no message-count cap. Messages pack until 512 BGE tokens, and a
single longer message stays whole. The published retrieval setting is
B=50, H=1, K=20. Ties keep that setting.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "reproduction/data/longmemeval_s_cleaned.json"
OLD_SPLIT = ROOT / "reproduction/experiments/main/results/longmemeval/hparam_dev/split.json"
OUT = ROOT / "reproduction/experiments/main/results/longmemeval/hparam_dev_soft512"
PYTHON = sys.executable
SEED = 20261105
FRACTION = 0.25
PAPER = {"name": "paper", "candidate_k": 50, "max_hops": 1, "top_k": 20, "messages_per_chunk": 0}
SWEEP = [
    PAPER,
    {"name": "k10", "candidate_k": 50, "max_hops": 1, "top_k": 10, "messages_per_chunk": 0},
    {"name": "k40", "candidate_k": 50, "max_hops": 1, "top_k": 40, "messages_per_chunk": 0},
    {"name": "b20", "candidate_k": 20, "max_hops": 1, "top_k": 20, "messages_per_chunk": 0},
    {"name": "b100", "candidate_k": 100, "max_hops": 1, "top_k": 20, "messages_per_chunk": 0},
    {"name": "h0", "candidate_k": 50, "max_hops": 0, "top_k": 20, "messages_per_chunk": 0},
    {"name": "h2", "candidate_k": 50, "max_hops": 2, "top_k": 20, "messages_per_chunk": 0},
]


def load_questions() -> list[dict]:
    rows = json.loads(DATA.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError("LongMemEval file is not a list")
    return [{"question_id": str(row["question_id"]), "question_type": str(row["question_type"])}
            for row in rows]


def stratified_dev(rows: list[dict]) -> dict:
    count = round(len(rows) * FRACTION)
    if not 0.20 * len(rows) <= count <= 0.30 * len(rows):
        raise ValueError(f"Development count {count} is outside 20%–30%")
    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(row["question_type"], []).append(row["question_id"])
    for ids in groups.values():
        ids.sort()
    quotas = {key: int(count * len(values) / len(rows)) for key, values in groups.items()}
    remainder = sorted(groups, key=lambda key: (-(count * len(groups[key]) / len(rows) - quotas[key]), key))
    for key in remainder[:count - sum(quotas.values())]:
        quotas[key] += 1
    rng = random.Random(SEED)
    dev = []
    for key, values in groups.items():
        dev.extend(rng.sample(values, quotas[key]))
    dev_set = set(dev)
    if len(dev_set) != count:
        raise ValueError("Development sample was not unique")
    heldout = [row["question_id"] for row in rows if row["question_id"] not in dev_set]
    return {
        "dataset": "longmemeval_s",
        "fraction": FRACTION,
        "seed": SEED,
        "sampling": (
            "stratified by question_type; largest remainder; ids sorted before "
            f"Random({SEED}).sample"
        ),
        "selection_rule": (
            "maximize development accuracy; ties keep B=50, H=1, K=20; "
            "chunks use a soft 512-token target and no message cap"
        ),
        "published": PAPER,
        "sweep": SWEEP,
        "quotas": quotas,
        "dev_count": len(dev),
        "heldout_count": len(heldout),
        "dev_ids": sorted(dev),
        "heldout_ids": heldout,
    }


def latest_grades(path: Path, wanted: set[str]) -> dict:
    found: dict[str, dict] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("question_id") in wanted:
                found[row["question_id"]] = row
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"{path} is missing {len(missing)} questions")
    categories: dict[str, dict] = {}
    correct = graded = errors = 0
    for row in found.values():
        category = categories.setdefault(str(row.get("question_type")), {"total": 0, "correct": 0})
        category["total"] += 1
        if row.get("status") == "succeeded" and isinstance(row.get("grade"), bool):
            graded += 1
            correct += int(row["grade"])
            category["correct"] += int(row["grade"])
        else:
            errors += 1
    return {
        "total": len(found),
        "valid_grades": graded,
        "correct": correct,
        "errors": errors,
        "accuracy": correct / graded if graded else None,
        "by_category": {
            key: {**value, "accuracy": value["correct"] / value["total"] if value["total"] else None}
            for key, value in sorted(categories.items())
        },
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_split(config: dict, questions_file: Path, output: Path) -> None:
    summary = output.with_suffix(".summary.json")
    expected = len(json.loads(questions_file.read_text(encoding="utf-8")))
    if summary.exists():
        saved = json.loads(summary.read_text(encoding="utf-8"))
        if saved.get("valid_grades") == expected and saved.get("errors") == 0:
            print(f"skip {config['name']}: {summary}", flush=True)
            return
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "-u", str(ROOT / "reproduction/run.py"),
        "--method", "sag-mem", "--dataset", "longmemeval", "--model", "qwen",
        "--data-path", str(DATA),
        "--candidate-k", str(config["candidate_k"]),
        "--max-hops", str(config["max_hops"]),
        "--top-k", str(config["top_k"]),
        "--messages-per-chunk", str(config["messages_per_chunk"]),
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


def distance(config: dict) -> int:
    return sum((
        config["candidate_k"] != PAPER["candidate_k"],
        config["max_hops"] != PAPER["max_hops"],
        config["top_k"] != PAPER["top_k"],
        config["messages_per_chunk"] != PAPER["messages_per_chunk"],
    ))


def main() -> None:
    split_path = OUT / "split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
    elif OLD_SPLIT.exists():
        split = json.loads(OLD_SPLIT.read_text(encoding="utf-8"))
        split["chunking"] = "soft512_atomic_v1"
        write_json(split_path, split)
    else:
        split = stratified_dev(load_questions())
        write_json(split_path, split)
    dev_ids = split["dev_ids"]
    heldout_ids = split["heldout_ids"]
    dev_file = OUT / "dev_ids.json"
    heldout_file = OUT / "heldout_ids.json"
    write_json(dev_file, dev_ids)
    write_json(heldout_file, heldout_ids)
    for config in SWEEP:
        run_split(config, dev_file, OUT / "dev" / config["name"] / "run.jsonl")
    rows = []
    for config in SWEEP:
        summary = json.loads((OUT / "dev" / config["name"] / "run.summary.json").read_text(encoding="utf-8"))
        rows.append({**config, "accuracy": summary["accuracy"], "correct": summary["total"] - summary["incorrect"],
                     "valid_grades": summary["valid_grades"], "errors": summary["errors"],
                     "by_category": summary.get("by_category"), "wall_seconds": summary.get("wall_seconds")})
    winner = max(rows, key=lambda row: (row["accuracy"], -distance(row)))
    write_json(OUT / "dev_selection.json", {"winner": winner["name"], "rows": rows})
    print(f"selected {winner['name']} dev accuracy={winner['accuracy']:.4f}", flush=True)
    chosen = next(config for config in SWEEP if config["name"] == winner["name"])
    output = OUT / "heldout" / chosen["name"] / "run.jsonl"
    run_split(chosen, heldout_file, output)
    summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
    heldout = {**chosen, "accuracy": summary["accuracy"], "correct": summary["total"] - summary["incorrect"],
               "valid_grades": summary["valid_grades"], "errors": summary["errors"],
               "by_category": summary.get("by_category"), "wall_seconds": summary.get("wall_seconds")}
    write_json(OUT / "heldout" / "selected.json", heldout)
    print(json.dumps({"selected": winner["name"], "heldout_accuracy": heldout["accuracy"],
                      "heldout_correct": heldout["correct"], "heldout_n": heldout["valid_grades"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
