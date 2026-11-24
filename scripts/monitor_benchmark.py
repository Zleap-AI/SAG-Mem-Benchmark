#!/usr/bin/env python3
"""Inspect persisted benchmark progress; row counts alone do not imply liveness."""

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.harness.benchmark import latest_records, read_records
from src.harness.reader_grader import aggregate_metrics


def snapshot(output: Path) -> dict:
    # 按题取最后状态，避免重试追加行被误算成新增完成题。
    records = latest_records(read_records(output))
    states = []
    for path in sorted(output.with_suffix(".states").glob("*.json")):
        state = json.loads(path.read_text())
        pid = state.get("worker_pid")
        alive = False
        # PID 存在只能说明进程尚在；结果行数和 PID 都不能证明任务正在持续推进。
        if pid and state["stage"] != "completed":
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                pass
            except PermissionError:
                alive = True
        state["pid_exists"] = alive
        state["liveness_note"] = "PID exists; counts do not establish active progress" if alive else "No live worker confirmed"
        directory = Path(state["build_directory"])
        manifest = directory / "build.json"
        state["build"] = json.loads(manifest.read_text()) if manifest.exists() else {"status": "unverified"}
        maintenance = directory / "maintenance.json"
        if maintenance.exists():
            saved = json.loads(maintenance.read_text())
            state["maintenance"] = {key: saved[key] for key in
                                    ("status", "freed_allocated_bytes", "updated_at", "error_type") if key in saved}
        db = directory / "sag" / "sag.db"
        if db.exists():
            try:
                with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
                    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                    state["database_counts"] = {table: conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                                                for table in ("source_chunk", "source_event", "chat_message") if table in tables}
            except sqlite3.Error as exc:
                state["database_error"] = str(exc)
        states.append(state)
    return {"summary": aggregate_metrics(list(records.values())), "groups": states}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0, help="0: one snapshot; positive: refresh seconds")
    args = parser.parse_args()
    if args.interval < 0:
        parser.error("interval cannot be negative")
    while True:
        print(json.dumps(snapshot(args.output), indent=2, ensure_ascii=False), flush=True)
        if not args.interval:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
