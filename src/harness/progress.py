"""Terminal-only progress; it does not define or upload MLflow metrics."""

from __future__ import annotations

import time
from pathlib import Path


class TerminalProgress:
    # 终端日志只反映执行进度，不负责定义或上传 MLflow 指标。
    def __init__(self, total: int, pending: int, groups: int, options: dict) -> None:
        self.total = total
        self.completed = total - pending
        self.skipped = self.completed
        self.pending_groups = groups
        self.options = options
        self.started = time.monotonic()
        self.graded = self.correct = self.errors = 0

    def start(self) -> None:
        print(f"condition={self.options['backend']} dataset={self.options['dataset']} "
              f"split={self.options['split']} concurrency={self.options['concurrency']} "
              f"output={self.options['output']}", flush=True)
        print(f"progress completed={self.completed}/{self.total} skipped={self.skipped} "
              f"pending={self.total - self.completed} history_groups={self.pending_groups}", flush=True)
        print(f"worker logs: {Path(self.options['output']).with_suffix('.workers')}", flush=True)

    def result(self, record: dict, report: str | None = None) -> None:
        self.completed += 1
        grade = record.get("grade")
        self.graded += grade is not None
        self.correct += grade is True
        self.errors += bool(record.get("error"))
        status = "ERR" if record.get("error") else "OK" if grade else "MISS" if grade is False else "SKIP"
        print(f"[{self.completed}/{self.total}] question_id={record['question_id']} {status} "
              f"live_accuracy={self.correct / self.graded if self.graded else 0:.3f} "
              f"graded={self.graded} errors={self.errors} elapsed={time.monotonic() - self.started:.0f}s",
              flush=True)
        if record.get("error"):
            print(f"  stage={record.get('failure_stage', 'unknown')} {record['error']}", flush=True)
        if report is not None:
            print(report, flush=True)

    def retry(self, question_id: str, attempt: int, error: Exception) -> None:
        print(f"retry question_id={question_id} attempt={attempt}/{self.options['worker_retries'] + 1} "
              f"{type(error).__name__}: {error}", flush=True)

    def finish(self, summary: dict | None) -> None:
        if summary is not None:
            score = summary.get("accuracy")
            print(f"DONE records={summary['total']} graded={summary['valid_grades']} "
                  f"accuracy={score if score is not None else 'n/a'} errors={summary['errors']} "
                  f"elapsed={time.monotonic() - self.started:.0f}s", flush=True)
