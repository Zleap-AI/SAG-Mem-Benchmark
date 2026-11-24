"""Replay Qwen ablations after the required local parent datasets are complete."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import re
import sys
import time
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
REPRO_ROOT = ROOT.parents[1]
OUTPUT_ROOT = ROOT
MAIN_ROOT = ROOT.parent / "main"
LITE_ROOT = ROOT.parent / "lite"
DATA_ROOT = REPRO_ROOT / "data"
sys.path.insert(0, str(REPRO_ROOT.parent))
from reproduction.load_env import load_repo_env

load_repo_env()
BASE_URL = os.environ.get("OPENROUTER_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY_ENV = None
MODEL = "qwen/qwen3.6-35b-a3b"
CONCURRENCY = 20
BODY = {"reasoning": {"enabled": False},
        "provider": {"only": ["DeepInfra"], "allow_fallbacks": False, "quantizations": ["fp8"]}}
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

from openai import APIConnectionError, APIStatusError, APITimeoutError
from src.harness.benchmark import Target, atomic_json, factory_digest
from src.harness.reader_grader import LoCoMoProtocol, LongMemEvalProtocol, ReaderGrader
from src.llm_client import ConfiguredAsyncOpenAI

from reproduction.experiments.ablations.prepare import (
    EXPECTED,
    comparison_profile,
    file_hash,
    fingerprint,
    input_identity,
    prepare,
)
from reproduction.experiments.shared.profiles import api_settings

CONTEXT = ContextVar("ablation_call")


def read_latest(path):
    rows = {}
    with path.open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError("Unfinished JSONL record")
            row = json.loads(line)
            rows[row["question_id"]] = row
    return rows


class Ledger:
    """RequestLedger-style accounting with condition and finish-reason attribution."""
    def __init__(self, path):
        self.path = path
        self.records = [json.loads(line) for line in path.open()] if path.exists() else []
        self.stream = path.open("a")

    def wrap(self, client, method):
        container = client.chat.completions
        original = getattr(container, method)

        async def invoke(*args, **kwargs):
            record = {**CONTEXT.get(), "method": method, "model": kwargs["model"],
                      "request_sha256": fingerprint(kwargs["messages"]), "started_at": time.time()}
            start = time.monotonic()
            try:
                response = await original(*args, **kwargs)
                value = response.model_dump(mode="json")
                usage = value.get("usage")
                choices = value.get("choices", [])
                record.update(response_id=value.get("id"), response_model=value.get("model"),
                              provider=value.get("provider"), usage=usage,
                              cost_usd=usage.get("cost") if usage is not None else None,
                              finish_reasons=[c.get("finish_reason") for c in choices], status="succeeded")
                if usage is None:
                    raise ValueError("Provider did not return token usage")
                if (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0):
                    raise ValueError("Reasoning tokens returned with reasoning disabled")
                if any(c.get("finish_reason") == "length" for c in choices):
                    raise ValueError("Model output was truncated")
                if any(c.get("message", {}).get("reasoning") for c in choices):
                    raise ValueError("Reasoning text returned with reasoning disabled")
                return response
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__,
                              status_code=getattr(error, "status_code", None))
                raise
            finally:
                record["elapsed_seconds"] = time.monotonic() - start
                self.records.append(record)
                self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                self.stream.flush()
                os.fsync(self.stream.fileno())

        setattr(container, method, invoke)

    def close(self):
        self.stream.close()


class AuditedReader(ReaderGrader):
    def __init__(self, ledger):
        super().__init__(MODEL, MODEL)
        credentials = {"api_key": os.environ[API_KEY_ENV]} if API_KEY_ENV else api_settings()
        for role in ("reader", "judge"):
            client = ConfiguredAsyncOpenAI(api_key=credentials["api_key"], base_url=BASE_URL,
                        request_extra_body=BODY, timeout=240, max_retries=0)
            self.__dict__[f"_{role}_client"] = client
            for method in ("create", "parse"):
                ledger.wrap(client, method)


class Job:
    def __init__(self, method, dataset, condition, source):
        self.key = f"{method}/{dataset}/{condition}"
        self.method, self.dataset, self.condition = method, dataset, condition
        self.source = source
        self.positions = {}
        self.baseline = {}
        with source.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                row = json.loads(line)
                self.positions[row["question_id"]] = offset
                if "source_main_grade" in row:
                    self.baseline[row["question_id"]] = row["source_main_grade"]
        cached = source.parent / "main_cached.jsonl"
        if cached.exists():
            for line in cached.open():
                item = json.loads(line)
                self.baseline[item["question_id"]] = item["grade"]
        prepared = json.loads(source.with_suffix(".jsonl.manifest.json").read_text())
        identity = {"prepared_run_id": prepared["run_id"], "source_sha256": file_hash(source),
                    "method": method, "dataset": dataset, "condition": condition,
                    "reader_model": MODEL, "judge_model": MODEL, "base_url": BASE_URL, "extra_body": BODY,
                    "global_concurrency": CONCURRENCY, "runner_sha256": file_hash(__file__),
                    "reader_grader_sha256": file_hash(REPRO_ROOT.parent / "src/harness/reader_grader.py"),
                    "prompts_sha256": file_hash(REPRO_ROOT.parent / "src/harness/prompts.py")}
        self.run_id = fingerprint(identity)
        self.directory = OUTPUT_ROOT / "runs" / self.key
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest_path = self.directory / "run.jsonl.manifest.json"
        manifest = {"run_id": self.run_id, "identity": identity,
                    "question_order": list(self.positions), "prepared_manifest": prepared}
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous["run_id"] != self.run_id:
                prior_semantics = {k: v for k, v in previous["identity"].items() if k != "runner_sha256"}
                current_semantics = {k: v for k, v in identity.items() if k != "runner_sha256"}
                completed_path = self.directory / "run.jsonl"
                completed = read_latest(completed_path) if completed_path.exists() else {}
                if prior_semantics != current_semantics or set(completed) != set(self.positions) or not all(
                        r["status"] == "succeeded" and type(r.get("grade")) is bool for r in completed.values()):
                    raise ValueError("Existing replay has a different frozen identity")
                # Completed inference remains immutable when only the waiting/provenance controller changes.
                self.run_id = previous["run_id"]
        if not manifest_path.exists():
            atomic_json(manifest_path, manifest)
        self.output = self.directory / "run.jsonl"
        self.states = {}
        self.offsets = {}
        if self.output.exists():
            with self.output.open("rb") as stream:
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    record = json.loads(line)
                    if record["run_id"] != self.run_id:
                        raise ValueError("Replay row run_id mismatch")
                    self.offsets[record["question_id"]] = offset
                    self.states[record["question_id"]] = {k: record.get(k) for k in
                        ("status", "grade", "failure_stage", "source_main_grade")}
        self.stream = self.output.open("a")
        self.protocol = LoCoMoProtocol() if dataset == "locomo" else LongMemEvalProtocol()

    def row(self, qid):
        path, offset = (self.output, self.offsets[qid]) if qid in self.offsets else (self.source, self.positions[qid])
        with path.open("rb") as stream:
            stream.seek(offset)
            row = json.loads(stream.readline())
        row["source_main_grade"] = self.baseline[qid]
        return row

    def save(self, row):
        self.offsets[row["question_id"]] = self.stream.tell()
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.states[row["question_id"]] = {k: row.get(k) for k in
            ("status", "grade", "failure_stage", "source_main_grade")}

    def summary(self, ledger):
        graded = [r for r in self.states.values() if r["status"] == "succeeded" and type(r["grade"]) is bool]
        calls = [r for r in ledger.records if r["job"] == self.key]
        known = [r["cost_usd"] for r in calls if r.get("cost_usd") is not None]
        result = {"job": self.key, "expected": len(self.positions), "graded": len(graded),
                  "correct": sum(r["grade"] for r in graded),
                  "errors": sum(r["status"] == "failed" for r in self.states.values()),
                  "complete": len(graded) == len(self.positions),
                  "accuracy_valid": sum(r["grade"] for r in graded) / len(graded) if graded else None,
                  "accuracy_all": sum(r["grade"] for r in graded) / len(self.positions),
                  "cached_main_correct": sum(self.baseline.values()),
                  "cached_main_accuracy": sum(self.baseline.values()) / len(self.positions),
                  "cached_main_new_calls": 0,
                  "paired_gained": sum(r["grade"] and not r["source_main_grade"] for r in graded),
                  "paired_lost": sum(not r["grade"] and r["source_main_grade"] for r in graded),
                  "new_calls_including_retries": len(calls), "new_cost_usd_known": sum(known),
                  "new_cost_unknown_calls": len(calls) - len(known),
                  "cost_scope": "new Reader/Judge calls; parent retrieval and main answers are cached",
                  "stages": {stage: {"calls": sum(r["stage"] == stage for r in calls),
                      "cost_usd_known": sum(r.get("cost_usd") or 0 for r in calls if r["stage"] == stage),
                      "cost_unknown_calls": sum(r.get("cost_usd") is None for r in calls if r["stage"] == stage)}
                      for stage in ("reader", "judge")}}
        atomic_json(self.directory / "summary.json", result)
        return result


async def evaluate(job, qid, reader, semaphore):
    row = job.row(qid)
    if row.get("run_id") != job.run_id:
        row.update(run_id=job.run_id, retrieval_only=False, grade=None, status="pending")
    for attempt in range(1, 4):
        stage = "reader"
        try:
            async with semaphore:
                if "hypothesis" not in row:
                    if re.sub(r"\[session\s+[^\]\r\n]+\][ \t]*", "", row["reader_context"]) != row["reader_context"]:
                        raise ValueError("Original Reader would remove source text")
                    CONTEXT.set({"job": job.key, "question_id": qid, "stage": stage, "attempt": attempt})
                    answer = await reader.respond(row["reader_context"], row["question"], job.protocol.reader_system)
                    row.update(hypothesis=answer.hypothesis, reader_prompt_tokens=answer.prompt_tokens,
                               reader_completion_tokens=answer.completion_tokens, reader_duration=answer.duration,
                               status="reader_complete", answer_origin="new_reader")
                    job.save(row)
                stage = "judge"
                CONTEXT.set({"job": job.key, "question_id": qid, "stage": stage, "attempt": attempt})
                target = Target(**row["target"])
                row["metrics"] = job.protocol.score(row["hypothesis"], target)
                grade = await reader.grade(row["question"], target.answer, row["hypothesis"], target.category,
                    prompt_override=job.protocol.judge_prompt(row["question"], row["hypothesis"], target),
                    **getattr(job.protocol, "judge_options", dict)())
                row.update(grade=grade.is_correct, grader_prompt_tokens=grade.prompt_tokens,
                           grader_completion_tokens=grade.completion_tokens, status="succeeded", evaluated_at=time.time())
                row.pop("error_type", None)
                row.pop("failure_stage", None)
                job.save(row)
                return
        except Exception as error:  # noqa: BLE001 -- persist the failed evaluation checkpoint
            row.update(status="failed", grade=None, failure_stage=stage, error_type=type(error).__name__,
                       failure_status_code=getattr(error, "status_code", None), attempt=attempt)
            job.save(row)
            transient = isinstance(error, (APIConnectionError, APITimeoutError, TimeoutError)) or (
                isinstance(error, APIStatusError) and (error.status_code == 429 or error.status_code >= 500))
            if not transient or attempt == 3:
                return
            await asyncio.sleep(2 ** attempt)


def parent_probe(method, dataset):
    base = MAIN_ROOT if method == "sag-mem" else LITE_ROOT
    paths = (sorted((base / "results" / dataset).glob("*/run.jsonl")) if method == "sag-mem"
             else [base / "results" / dataset / "qwen/run.jsonl"])
    if not paths:
        return None, {"state": "waiting_parent", "missing_files": 1}
    missing = [p for p in paths if not p.exists() or not p.with_suffix(".jsonl.manifest.json").exists()]
    if missing:
        return None, {"state": "waiting_parent", "missing_files": len(missing)}
    metadata, identifiers, failed, builds = {}, set(), 0, set()
    for path in paths:
        manifest = json.loads(path.with_suffix(".jsonl.manifest.json").read_text())
        identity = manifest["identity"]
        protocol_type = LoCoMoProtocol if dataset == "locomo" else LongMemEvalProtocol
        if identity["protocol_code"] != factory_digest(protocol_type):
            raise ValueError("Parent and replay protocol code differ")
        for role in ("reader_service", "judge_service"):
            profile = identity[role]
            expected = {"model": MODEL, "base_url": BASE_URL, "extra_body": BODY}
            if comparison_profile({role: profile}) != comparison_profile({role: expected}):
                raise ValueError("Parent is not the frozen Qwen DeepInfra FP8 configuration")
        before = (path.stat().st_size, path.stat().st_mtime_ns)
        rows = read_latest(path)
        if before != (path.stat().st_size, path.stat().st_mtime_ns):
            return None, {"state": "waiting_stable_parent"}
        for qid, row in rows.items():
            if qid in identifiers:
                raise ValueError("Parent datasets overlap")
            identifiers.add(qid)
            failed += row.get("status") != "succeeded" or type(row.get("grade")) is not bool
            builds.add(row["build_id"])
        metadata[str(path)] = before
    expected = EXPECTED["all"][dataset]
    if len(identifiers) != expected or failed:
        return None, {"state": "waiting_complete_parent", "records": len(identifiers), "expected": expected,
                      "failed_or_ungraded": failed}
    return {"paths": paths, "metadata": metadata, "builds": builds}, {"state": "parent_complete", "records": expected}


def build_inputs(method, dataset, kind, parent):
    scope = "all" if kind == "h0" else "temporal"
    destination = OUTPUT_ROOT / "prepared" / method / dataset / scope
    if destination.exists():
        report = json.loads((destination / "validation.json").read_text())
        expected = EXPECTED[scope][dataset]
        if report["questions"] != expected:
            raise ValueError("Existing inputs have incomplete coverage")
        return destination
    main = MAIN_ROOT
    if kind == "h0":
        source_root = main / "retained_sources"
        for build in parent["builds"]:
            matches = list(source_root.glob(f"**/{build}/sources.json"))
            if len(matches) != 1 or not matches[0].with_name("build.json").exists():
                return None
        data_root = DATA_ROOT
        for path in parent["paths"]:
            manifest = json.loads(path.with_suffix(".jsonl.manifest.json").read_text())
            effective, _ = input_identity(path.with_suffix(".jsonl.manifest.json"), manifest)
            if any(not (data_root / Path(p).name).exists() for p in effective["data_checksums"]):
                return None
    else:
        source_root = data_root = None
    prepare(SimpleNamespace(output=destination, source=parent["paths"], method=method, dataset=dataset,
            scope=scope, limit=None, no_expansion=kind == "h0", only_no_expansion=kind == "h0",
            sources_root=source_root, data_root=data_root))
    return destination


async def run():
    ledger = Ledger(OUTPUT_ROOT / "requests.jsonl")
    reader = AuditedReader(ledger)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    queue = asyncio.Queue()
    jobs, loaded, waiting, probe_errors = {}, set(), {}, {}
    groups = [("sag-mem", d, kind) for d in ("locomo", "longmemeval") for kind in ("time", "h0")]
    groups += [("sag-mem-lite", d, "time") for d in ("locomo", "longmemeval")]
    completed = 0
    active = 0

    async def worker():
        nonlocal completed, active
        while True:
            entry = await queue.get()
            if entry is None:
                queue.task_done()
                return
            job, qid = entry
            active += 1
            try:
                await evaluate(job, qid, reader, semaphore)
                completed += 1
                if completed % 20 == 0:
                    print(json.dumps({"finished": completed, "queued": queue.qsize(), "active": active,
                                      "job": job.key, "last_status": job.states[qid]["status"]}), flush=True)
            finally:
                active -= 1
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(CONCURRENCY)]
    try:
        while True:
            for group in groups:
                if group in loaded:
                    continue
                method, dataset, kind = group
                key = "/".join(group)
                try:
                    parent, status = await asyncio.to_thread(parent_probe, method, dataset)
                    waiting[key] = status
                    if parent is None:
                        continue
                    destination = await asyncio.to_thread(build_inputs, method, dataset, kind, parent)
                    if destination is None:
                        waiting[key] = {"state": "waiting_retained_sources_or_data"}
                        continue
                    conditions = ("no_expansion",) if kind == "h0" else ("raw", "normalized")
                    for condition in conditions:
                        job = Job(method, dataset, condition, destination / f"{condition}.jsonl")
                        jobs[job.key] = job
                        for qid in job.positions:
                            if job.states.get(qid, {}).get("status") != "succeeded":
                                queue.put_nowait((job, qid))
                    loaded.add(group)
                    waiting.pop(key, None)
                    probe_errors.pop(key, None)
                    print(json.dumps({"loaded": key, "conditions": conditions, "queued": queue.qsize()}), flush=True)
                except (ValueError, KeyError, OSError) as error:
                    probe_errors[key] = {"error_type": type(error).__name__, "reason": str(error)[:400]}
            summaries = [job.summary(ledger) for job in jobs.values()]
            for task in workers:
                if task.done():
                    task.result()
            all_loaded = len(loaded) == len(groups)
            done = all_loaded and not active and queue.empty()
            status = {"state": "complete" if done and all(s["complete"] for s in summaries) else
                      "finished_with_errors" if done else "running_and_waiting",
                      "pid": os.getpid(), "updated_at": time.time(), "global_concurrency": CONCURRENCY,
                      "active": active, "queued": queue.qsize(), "groups_loaded": len(loaded),
                      "groups_expected": len(groups), "waiting": waiting, "input_errors": probe_errors,
                      "jobs": summaries, "all_new_calls": len(ledger.records),
                      "new_cost_usd_known": sum(r.get("cost_usd") or 0 for r in ledger.records),
                      "new_cost_unknown_calls": sum(r.get("cost_usd") is None for r in ledger.records)}
            atomic_json(OUTPUT_ROOT / "STATUS.json", status)
            if done:
                return 0 if status["state"] == "complete" else 1
            await asyncio.sleep(15)
    finally:
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        for job in jobs.values():
            job.stream.close()
        await reader.aclose()
        ledger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--main-root", type=Path, default=MAIN_ROOT)
    parser.add_argument("--lite-root", type=Path, default=LITE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--api-key-env", default=API_KEY_ENV)
    parser.add_argument("--concurrency", type=int, choices=range(1, 21), default=20)
    args = parser.parse_args()
    DATA_ROOT, MAIN_ROOT, LITE_ROOT = args.data_root, args.main_root, args.lite_root
    OUTPUT_ROOT, BASE_URL, API_KEY_ENV = args.output_root, args.base_url, args.api_key_env
    CONCURRENCY = args.concurrency
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(asyncio.run(run()))
