"""MLflow lifecycle for a single CLI invocation, including unsuccessful attempts."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Self


class ExperimentTracking:
    # 启用 tracking 时每次 CLI 调用对应一个 MLflow run；续跑创建新尝试，实验身份仍可相同。
    def __init__(self, options: dict) -> None:
        self.options = options
        self.enabled = options.get("mlflow", False) and not options.get("preflight", False)
        self.client = None
        self.run_id = None
        self.summary = None
        self.manifest = None
        self.live_completed = 0
        self.live_correct = 0
        self.live_graded = 0
        self.previous_sigterm = None
        self.reporter_metrics = False

    def _batch(self, *, params=None, metrics=None, tags=None, step=0, points=None) -> None:
        from mlflow.entities import Metric, Param, RunTag

        now = int(time.time() * 1000)
        self.client.log_batch(
            self.run_id,
            params=[Param(k, str(v)[:500]) for k, v in (params or {}).items()],
            metrics=[Metric(k, float(v), now, s) for k, v, s in (points or [])]
                    + [Metric(k, float(v), now, step) for k, v in (metrics or {}).items()],
            tags=[RunTag(k, str(v)[:5000]) for k, v in (tags or {}).items()],
            synchronous=True,
        )

    def _safe(self, operation, *args, **kwargs) -> None:
        try:
            operation(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - tracking outages must preserve benchmark results
            print(f"MLflow update failed; local results preserved: {type(exc).__name__}: {exc}",
                  file=sys.stderr)

    def __enter__(self) -> Self:
        if not self.enabled:
            return self
        from ..config import settings

        try:
            from mlflow.tracking import MlflowClient

            self.client = MlflowClient(tracking_uri=settings.mlflow_url)
            experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
            if experiment_id:
                experiment = self.client.get_experiment(experiment_id)
                if experiment.lifecycle_stage != "active":
                    raise ValueError(f"MLflow experiment {experiment_id} is not active")
            else:
                experiment = self.client.get_experiment_by_name(settings.mlflow_experiment_name)
                if experiment is None:
                    try:
                        experiment_id = self.client.create_experiment(settings.mlflow_experiment_name)
                    except Exception:
                        # Another invocation can create the experiment concurrently.
                        experiment = self.client.get_experiment_by_name(settings.mlflow_experiment_name)
                        if experiment is None:
                            raise
                        experiment_id = experiment.experiment_id
                else:
                    experiment_id = experiment.experiment_id
            mode = next((key for key in ("metrics_only", "merge_from", "regrade_from", "replay_from")
                         if self.options.get(key)), "benchmark")
            remote = self.client.create_run(experiment_id, run_name=self.options.get("run_name"), tags={
                "benchmark.stage": "starting", "benchmark.mode": mode,
                "benchmark.output": self.options["output"],
                "benchmark.resume": str(self.options.get("resume", False)),
            })
            self.run_id = remote.info.run_id
            # requested.* 记录本次请求，后续 bind 再记录源结果决定的实际配置，避免默认值混淆。
            self._batch(params={f"requested.{k}": v for k, v in self.options.items() if k != "explicit"})
        except BaseException as exc:
            if self.run_id:
                self._safe(self._batch, tags={"benchmark.stage": "failed", "benchmark.error": str(exc)})
                status = "KILLED" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "FAILED"
                self._safe(self.client.set_terminated, self.run_id, status=status)
            if not isinstance(exc, Exception):
                raise
            raise RuntimeError(f"MLflow startup failed; experiment was not started: {exc}") from exc
        print(f"MLflow run: {settings.mlflow_url.rstrip('/')}/#/experiments/{experiment_id}/runs/{self.run_id}",
              flush=True)
        self.previous_sigterm = signal.signal(signal.SIGTERM, self._terminate)
        return self

    @staticmethod
    def _terminate(signum, frame) -> None:
        raise SystemExit(128 + signum)

    def bind(self, manifest: dict) -> None:
        from .reporting import has_reporter

        self.manifest = manifest
        self.reporter_metrics = has_reporter(manifest)
        if not self.enabled:
            return
        config = manifest.get("identity", {}).get("config", {})
        self._safe(self._batch, params={**config, "run_id": manifest["run_id"]},
                   tags={"benchmark.stage": "running"})

    async def record(self, record: dict, *, extra_metrics: dict | None = None) -> None:
        """Emit every completion, exactly as the legacy live_* curves did."""
        # 实时曲线只统计本次完成的题；最终曲线再从落盘结果中补齐此前成功的题。
        if not self.enabled:
            return
        self.live_completed += 1
        if record.get("grade") is True:
            self.live_correct += 1
        if record.get("grade") is not None:
            self.live_graded += 1
        metrics = {}
        if not self.reporter_metrics:
            if record.get("grade") is not None:
                metrics["live_grade"] = 1.0 if record.get("grade") else 0.0
            metrics["live_llm_judge"] = self.live_correct / self.live_graded if self.live_graded else 0.0
        metrics.update(extra_metrics or {})
        if metrics:
            await asyncio.to_thread(self._safe, self._batch, metrics=metrics, step=self.live_completed)

    def complete(self, summary: dict) -> None:
        self.summary = summary

    def _artifacts(self) -> None:
        output = Path(self.options["output"])
        for path in (output, output.with_suffix(output.suffix + ".manifest.json"),
                     output.with_suffix(".summary.json")):
            if path.exists():
                self._safe(self.client.log_artifact, self.run_id, str(path))
        for path in output.with_suffix(".states").glob("*.json"):
            value = json.loads(path.read_text())
            build = Path(value["build_directory"]) / "build.json"
            if build.exists():
                self._safe(self.client.log_artifact, self.run_id, str(build),
                           artifact_path="builds/" + value["build_id"])

    def _final_results(self) -> None:
        from .benchmark import latest_records
        from .mlflow_metrics import aggregate_metrics, ordered_records, step_metrics
        from .reporting import make_reporter

        output = Path(self.options["output"])
        saved = output.with_suffix(output.suffix + ".manifest.json")
        # 只上传本次已验证身份的结果，不能把同路径下冲突的旧实验记到当前 run。
        if self.manifest is not None:
            if not saved.exists() or json.loads(saved.read_text())["run_id"] != self.manifest["run_id"]:
                return
        elif not self.options.get("metrics_only"):
            return
        # 磁盘写满可能留下半行；上传时只读取已完整落盘的前缀，不修复或改写原文件。
        rows = []
        if output.exists():
            with output.open("rb") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except (ValueError, UnicodeError):
                        if not line.endswith(b"\n"):
                            break
                        raise
                    rows.append(row)
        records = ordered_records(latest_records(rows), self.manifest)
        reporter = make_reporter(self.manifest, records)
        if reporter:
            snapshot = reporter.snapshot()
            # MLflow selects the highest step. Final aggregates must follow running snapshots.
            final_step = self.live_completed + 1 if self.live_completed else 0
            self._safe(self._batch, metrics=reporter.metrics(snapshot, "final"),
                       tags=reporter.tags(snapshot), step=final_step)
        else:
            points = list(step_metrics(records))
            for offset in range(0, len(points), 1000):
                self._safe(self._batch, points=points[offset:offset + 1000])
            # LongMemEval retains the historical dataset-order curves and step=0 aggregates.
            self._safe(self._batch, metrics=aggregate_metrics(records))
        self._safe(self._artifacts)

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self.enabled:
            return False
        signal.signal(signal.SIGTERM, self.previous_sigterm)
        killed = isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError))
        status = "KILLED" if killed else "FAILED" if exc or (self.summary or {}).get("errors") else "FINISHED"
        self._safe(self._batch, tags={"benchmark.stage": status.lower(),
                                     **({"benchmark.error": "".join(traceback.format_exception(exc))[-5000:]}
                                        if exc else {})})
        # 先结束 run 再上传指标和产物，避免上传异常使远端永远停留在 RUNNING。
        self._safe(self.client.set_terminated, self.run_id, status=status)
        self._safe(self._final_results)
        return False
