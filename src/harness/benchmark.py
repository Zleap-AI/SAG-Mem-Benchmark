"""Offline QA contracts, lazy registries and history-group execution.

Plugins register factories here; the runner knows no concrete backend names.
Only History and BuildSpec cross the prepare boundary. Targets stay in the parent.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .tracking import ExperimentTracking

VERSION = 1


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    # 按文件原始字节计算指纹；仅修改注释也会改变引用该文件的运行或建库身份。
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@dataclass(frozen=True)
class Message:
    id: str
    speaker: str
    content: str
    role: str = "user"


@dataclass(frozen=True)
class Session:
    id: str
    messages: tuple[Message, ...]
    date_raw: str = ""
    timestamp: str | None = None
    time_known: bool = False


@dataclass(frozen=True)
class History:
    corpus_id: str
    sessions: tuple[Session, ...]

    def prefix(self, count: int | None) -> History:
        # 按原始消息顺序截取可见历史，允许截到会话中途；不依赖题目日期排序。
        if count is None:
            return self
        if count < 0 or count > sum(len(s.messages) for s in self.sessions):
            raise ValueError(f"Invalid message prefix {count} for {self.corpus_id}")
        sessions = []
        for session in self.sessions:
            if count <= 0:
                break
            messages = session.messages[:count]
            sessions.append(Session(session.id, messages, session.date_raw,
                                    session.timestamp, session.time_known))
            count -= len(messages)
        return History(self.corpus_id, tuple(sessions))

    @property
    def digest(self) -> str:
        # 历史内容和来源 ID 决定复用范围，问题与标准答案不参与历史指纹。
        return fingerprint(asdict(self)["sessions"])


@dataclass(frozen=True)
class Query:
    text: str
    date_raw: str = ""
    timestamp: str | None = None


@dataclass(frozen=True)
class Target:
    # 评分标签仅供父进程使用，不能作为后端建库或检索的输入。
    answer: str
    category: str
    abstention: bool = False
    session_ids: tuple[str, ...] = ()
    message_ids: tuple[str, ...] = ()
    evidence_status: str = "missing"
    invalid_evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class Example:
    id: str
    query: Query
    target: Target
    history_ref: dict
    history_digest: str
    corpus_id: str
    prefix: int | None = None
    unknown_time: bool = False


@dataclass(frozen=True)
class BuildSpec:
    id: str
    namespace: str
    index_root: str
    dependencies: dict

    @property
    def directory(self) -> Path:
        return Path(self.index_root) / self.namespace / self.id


@dataclass
class RetrievalResult:
    # context 用于回答；items 保留证据来源，单独支持检索指标计算和结果审计。
    context: str
    items: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    source_granularity: str = "unavailable"
    metric_family: str | None = None
    metric_items: list[dict] | None = None
    metric_sources_reliable: bool | None = None


class MemoryBackend(Protocol):
    @classmethod
    def validate(cls, options: dict, examples: list[Example]) -> None: ...

    @classmethod
    def dependencies(cls, options: dict) -> dict: ...

    async def prepare(self, history: History, build_spec: BuildSpec) -> None: ...

    async def retrieve(self, query: Query, options: dict) -> RetrievalResult: ...

    async def aclose(self) -> None: ...


class DatasetAdapter(Protocol):
    def scan(self, options: dict) -> list[Example]: ...

    def load_history(self, reference: dict) -> History: ...


class EvaluationProtocol(Protocol):
    reader_system: str

    def question(self, query: Query, include_date: bool) -> str: ...

    def score(self, answer: str, target: Target) -> dict: ...

    def judge_prompt(self, question: str, answer: str, target: Target) -> str: ...


# 按需导入具体实现，使调度器无需按后端名称分支，也避免提前加载全部 SDK。
REGISTRIES: dict[str, dict[str, Any]] = {
    "dataset": {"longmemeval": "src.data.loader:LongMemEval", "locomo": "src.data.loader:LoCoMo"},
    "backend": {
        "full_context": "src.backends.adapters:FullContext",
        "sag2": "src.backends.sag2_adapter:SAG2Backend",
        "sag2-chunk-time": "src.backends.sag2_chunk_time:SAG2ChunkTimeBackend",
    },
    "protocol": {"longmemeval": "src.harness.reader_grader:LongMemEvalProtocol",
                 "locomo": "src.harness.reader_grader:LoCoMoProtocol",
                 "locomo_mem0": "src.harness.reader_grader:LoCoMoProtocol"},
}


def register(kind: str, name: str, factory: Any) -> None:
    if kind not in REGISTRIES or name in REGISTRIES[kind]:
        raise ValueError(f"Duplicate or invalid registration: {kind}/{name}")
    REGISTRIES[kind][name] = factory


def resolve(kind: str, name: str) -> Any:
    value = REGISTRIES[kind].get(name)
    if value is None:
        raise ValueError(f"Unknown {kind} {name!r}; available: {', '.join(REGISTRIES[kind])}")
    if isinstance(value, str):
        module, attribute = value.split(":", 1)
        value = getattr(importlib.import_module(module), attribute)
    return value


def load_plugins(modules: list[str]) -> None:
    for module in modules:
        importlib.import_module(module)


def factory_digest(factory: Any) -> str:
    path = inspect.getsourcefile(factory)
    return file_digest(Path(path)) if path else str(factory)


def make_build(example: Example, options: dict, dependencies: dict) -> BuildSpec:
    # 建库身份只由可见历史和提取依赖决定，Reader/Judge 与选题配置另记入运行身份。
    reused = options.get("_reused_builds", {}).get(example.id)
    if reused is not None:
        if reused["history_digest"] != example.history_digest:
            raise ValueError(f"Reused build history does not match question {example.id}")
        return BuildSpec(
            reused["build_id"], options["namespace"], options["index_root"], reused["dependencies"]
        )
    identity = {"history": example.history_digest, "dependencies": dependencies, "version": VERSION}
    return BuildSpec(fingerprint(identity), options["namespace"], options["index_root"], dependencies)


def read_records(path: Path, repair_tail: bool = False) -> list[dict]:
    """Only an unterminated, invalid last JSON line is repairable."""
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    records = []
    offset = 0
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
            if not isinstance(record, dict) or not record.get("question_id"):
                raise ValueError("Expected record object with question_id")
        except (ValueError, UnicodeError):
            # 续跑只允许修复末尾未写完的一行；中间损坏或完整坏行必须报错。
            if repair_tail and index == len(lines) - 1 and not line.endswith(b"\n"):
                with path.open("r+b") as stream:
                    stream.truncate(offset)
                break
            raise ValueError(f"Corrupt result at {path}:{index + 1}") from None
        records.append(record)
        offset += len(line)
    return records


def latest_records(records: list[dict]) -> dict[str, dict]:
    # JSONL 采用追加写入，同一题重试后的最后一条记录覆盖此前状态。
    return {record["question_id"]: record for record in records}


class ResultStore:
    """OS advisory lock, held through run completion; never removes another lock."""
    def __init__(self, path: Path, manifest: dict, resume: bool = False) -> None:
        import fcntl

        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = path.with_suffix(path.suffix + ".lock").open("a+")
        try:
            # 整个运行持有结果文件锁，并先核对身份，防止不同实验混写同一输出。
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest_path = path.with_suffix(path.suffix + ".manifest.json")
            if manifest_path.exists():
                old = json.loads(manifest_path.read_text())
                if old["run_id"] != manifest["run_id"]:
                    raise ValueError("Output run identity mismatch; select a new output file")
                if not resume:
                    raise ValueError("Output already exists; use --resume or a new output")
            elif path.exists() and path.stat().st_size:
                raise ValueError("Unverified result file without run manifest; use a new output")
            self.records = latest_records(read_records(path, repair_tail=resume))
            if any(r.get("run_id") != manifest["run_id"] for r in self.records.values()):
                raise ValueError("Result rows do not match run manifest")
            atomic_json(manifest_path, manifest)
            # A complete JSON value without its newline is valid after interruption.
            if path.exists() and path.stat().st_size:
                with path.open("rb+") as stream:
                    stream.seek(-1, 2)
                    if stream.read(1) != b"\n":
                        stream.write(b"\n")
            self.stream = path.open("a", encoding="utf-8")
        except BaseException:
            self.lock.close()
            raise

    def append(self, record: dict) -> None:
        self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.records[record["question_id"]] = record

    def close(self) -> None:
        self.stream.close()
        self.lock.close()


async def acquire_build_lock(spec: BuildSpec):
    """The kernel releases this lock on close/worker death, without PID-file races."""
    import fcntl

    spec.directory.mkdir(parents=True, exist_ok=True)
    # 锁覆盖索引的打开、使用、关闭和维护；进程退出后由内核释放。
    handle = (spec.directory / ".build.lock").open("a+")
    try:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return handle
            except BlockingIOError:
                await asyncio.sleep(0.1)
    except BaseException:
        handle.close()
        raise


class WorkerFailure(RuntimeError):
    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(message)


class BackendProcess:
    def __init__(self, options: dict):
        self.options = options
        self.process = None
        self.log = None

    async def start(self, example: Example, spec: BuildSpec) -> None:
        log_path = Path(self.options["output"]).with_suffix(".workers") / f"{spec.id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open("ab")
        script = Path(__file__).parents[2] / "scripts" / "run_benchmark.py"
        self.process = await asyncio.create_subprocess_exec(
            sys.executable, str(script), "--internal-worker", stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=self.log, limit=128 * 1024 * 1024,
        )
        state_path = Path(self.options["output"]).with_suffix(".states") / f"{spec.id}.json"
        atomic_json(state_path, {"build_id": spec.id, "stage": "preparing", "updated_at": time.time(),
                                 "worker_pid": self.process.pid, "build_directory": str(spec.directory)})
        # prepare 只传历史引用和构建配置，当前问题、标准答案及证据标签均不传入。
        await self.call({"method": "prepare", "dataset": self.options["dataset"],
                         "backend": self.options["backend"], "plugins": self.options["plugins"],
                         "options": self.options, "reference": example.history_ref,
                         "prefix": example.prefix, "digest": example.history_digest, "build": asdict(spec)})

    async def call(self, message: dict) -> dict:
        async def exchange():
            self.process.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode())
            await self.process.stdin.drain()
            line = await self.process.stdout.readline()
            if not line:
                raise WorkerFailure(message["method"], f"Worker exited ({self.process.returncode}); see worker log")
            try:
                response = json.loads(line)
            except ValueError as exc:
                raise WorkerFailure(message["method"], "Invalid worker response") from exc
            if "error" in response:
                raise WorkerFailure(response.get("stage", message["method"]), response["error"])
            return response

        try:
            return await asyncio.wait_for(exchange(), self.options["worker_timeout"])
        except TimeoutError as exc:
            await self.close(kill=True)
            raise WorkerFailure(message["method"], "Worker timeout") from exc

    async def retrieve(self, query: Query) -> RetrievalResult:
        response = await self.call({"method": "retrieve", "query": asdict(query)})
        return RetrievalResult(**response)

    async def close(self, kill: bool = False) -> None:
        # 关闭阶段可能执行索引维护，使用独立超时预算；超时只终止本对象的 worker。
        if self.process and self.process.returncode is None:
            if not kill:
                try:
                    self.process.stdin.write(b'{"method":"close"}\n')
                    await self.process.stdin.drain()
                    await asyncio.wait_for(self.process.wait(), self.options.get("worker_close_timeout", 300))
                except (TimeoutError, BrokenPipeError, ConnectionResetError):
                    print("Worker close did not finish; maintenance may need to be retried", file=sys.stderr)
                    kill = True
            if kill and self.process.returncode is None:
                try:
                    self.process.kill()
                except ProcessLookupError:
                    pass
                await self.process.wait()
        if self.log:
            self.log.close()


async def worker_main() -> None:
    """One process per history group, multiple queries, no grading targets."""
    from contextlib import redirect_stdout

    output = sys.stdout
    backend = None
    lease = None
    # stdout 专供 JSON RPC，后端普通日志重定向到 stderr，避免破坏父子进程通信。
    with redirect_stdout(sys.stderr):
        try:
            while True:
                line = await asyncio.to_thread(sys.stdin.readline)
                if not line:
                    break
                request = json.loads(line)
                stage = request["method"]
                if stage == "close":
                    break
                try:
                    if stage == "prepare":
                        load_plugins(request["plugins"])
                        options = request["options"]
                        # 获得并发名额后才加载完整历史，避免等待中的任务占满内存。
                        history = resolve("dataset", request["dataset"])().load_history(request["reference"])
                        history = history.prefix(request["prefix"])
                        if history.digest != request["digest"]:
                            raise ValueError("History changed after preflight")
                        spec = BuildSpec(**request["build"])
                        lease = await acquire_build_lock(spec)
                        backend = resolve("backend", request["backend"])(options)
                        await backend.prepare(history, spec)
                        if not (spec.directory / "build.json").exists():
                            atomic_json(spec.directory / "build.json", {
                                "build_id": spec.id, "status": "succeeded", "dependencies": spec.dependencies,
                                "history_digest": history.digest,
                            })
                        published = json.loads((spec.directory / "build.json").read_text())
                        if (published.get("status") != "succeeded" or published.get("build_id") != spec.id
                                or fingerprint(published.get("dependencies")) != fingerprint(spec.dependencies)):
                            raise ValueError("Prepared backend has an invalid build manifest")
                        response = {"prepared": True}
                    elif stage == "retrieve":
                        response = asdict(await backend.retrieve(Query(**request["query"]), options))
                    else:
                        raise ValueError(f"Unknown RPC method {stage}")
                except Exception as exc:  # noqa: BLE001 - preserve per-stage failures without losing results
                    # 服务或基础设施异常记录失败阶段，不能当成模型正常答错。
                    response = {"error": f"{type(exc).__name__}: {exc}", "stage": (getattr(exc, "stage", None) or stage)}
                output.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
        finally:
            if backend is not None:
                closer = getattr(backend, "aclose", None) or backend.close
                result = closer()
                if inspect.isawaitable(result):
                    await result
            if lease is not None:
                lease.close()


def select_examples(examples: list[Example], options: dict) -> list[Example]:
    import random

    if options.get("question_types"):
        wanted_types = set(options["question_types"])
        available_types = {example.target.category for example in examples}
        missing_types = wanted_types - available_types
        if missing_types:
            raise ValueError(f"Unknown question types: {sorted(missing_types)}")
        examples = [example for example in examples if example.target.category in wanted_types]
    if options.get("questions"):
        wanted = set(options["questions"])
        missing = wanted - {e.id for e in examples}
        if missing:
            raise ValueError(f"Unknown question IDs: {sorted(missing)}")
        return [e for e in examples if e.id in wanted]
    if options.get("stratified"):
        count = options["stratified"]
        if count > len(examples):
            raise ValueError("Stratified sample exceeds dataset size")
        groups = {}
        for example in examples:
            groups.setdefault(example.target.category, []).append(example)
        quotas = {k: int(count * len(v) / len(examples)) for k, v in groups.items()}
        order = sorted(groups, key=lambda k: (-(count * len(groups[k]) / len(examples) - quotas[k]), k))
        for key in order[:count - sum(quotas.values())]:
            quotas[key] += 1
        rng = random.Random(options["seed"])
        chosen = {e.id for key, values in groups.items() for e in rng.sample(values, quotas[key])}
        return [e for e in examples if e.id in chosen]
    start, limit = options["start"], options["limit"]
    return examples[start:start + limit if limit else None]


def evaluation_services(options: dict | None = None, *, include_extract: bool = True) -> dict:
    """Record the effective endpoint and model of the relevant LLM roles.

    Credentials are never recorded. The extract role is included so that a run
    whose memory was built by a different extraction model no longer shares an
    identity with one built by another. Saved-result evaluation excludes the
    current extractor: its evidence is already fixed by the source run.
    """
    from ..config import settings

    options = options or {}

    def service(endpoint, model: str | None = None) -> dict:
        return {"base_url": endpoint.base_url, "model": model or endpoint.model,
                "extra_body": endpoint.extra_body}

    services = {
        "reader_service": service(settings.search_llm, options.get("reader_model")),
        "judge_service": service(settings.judge_llm, options.get("judge_model")),
    }
    if include_extract:
        services["extract_service"] = service(settings.extract_llm)
    return services


def _load_reused_builds(options: dict, examples: list[Example]) -> dict[str, dict[str, Any]]:
    source = Path(str(options["reuse_builds_from"])).resolve()
    if not source.is_file():
        raise ValueError(f"Reuse-build source is missing or not a file: {source}")
    rows = latest_records(read_records(source))
    root = Path(options["index_root"])
    selected: dict[str, dict[str, Any]] = {}
    for example in examples:
        row = rows.get(example.id)
        if row is None:
            raise ValueError(f"Reuse-build source has no result for question {example.id}")
        source_corpus = str(row.get("corpus_id") or "")
        if source_corpus and source_corpus != example.corpus_id:
            raise ValueError(f"Reuse-build corpus does not match question {example.id}")
        build_id = str(row.get("build_id") or "")
        if not build_id:
            raise ValueError(f"Reuse-build source has no build_id for question {example.id}")
        manifest_path = root / options["namespace"] / build_id / "build.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ValueError(f"Reused SAG build manifest is missing: {manifest_path}") from None
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid reused SAG build manifest: {manifest_path}") from exc
        if manifest.get("status") != "succeeded" or manifest.get("build_id") != build_id:
            raise ValueError(f"Reused SAG build is not a verified success: {manifest_path}")
        dependencies = manifest.get("dependencies")
        if not isinstance(dependencies, dict):
            raise TypeError(f"Reused SAG build has invalid dependencies: {manifest_path}")
        selected[example.id] = {
            "build_id": build_id,
            "history_digest": example.history_digest,
            "dependencies": dependencies,
        }
    return selected


def _select_examples_from_reuse_source(
    examples: list[Example], options: dict
) -> list[Example]:
    """Match governance batch order: result-file order, then distinct build ID."""
    source = Path(str(options["reuse_builds_from"])).resolve()
    if not source.is_file():
        raise ValueError(f"Reuse-build source is missing or not a file: {source}")
    by_question = {example.id: example for example in examples}
    question_types = set(options.get("question_types") or ())
    selected: list[Example] = []
    seen_builds: set[str] = set()
    for row in read_records(source):
        if question_types and str(row.get("question_type") or "") not in question_types:
            continue
        build_id = str(row.get("build_id") or "")
        question_id = str(row.get("question_id") or "")
        if not build_id or build_id in seen_builds:
            continue
        example = by_question.get(question_id)
        if example is None:
            raise ValueError(f"Reuse-build source has unknown question {question_id!r}")
        source_corpus = str(row.get("corpus_id") or "")
        if source_corpus and source_corpus != example.corpus_id:
            raise ValueError(f"Reuse-build corpus does not match question {example.id}")
        seen_builds.add(build_id)
        selected.append(example)
    start, limit = options["start"], options["limit"]
    selected = selected[start : start + limit if limit else None]
    if not selected:
        raise ValueError("Reuse-build source selected no questions")
    return selected


def run_manifest(options: dict, examples: list[Example], dependencies: dict) -> dict:
    # 排除续跑、并发和 worker 超时等执行控制项，避免仅调整调度就改变实验身份。
    excluded = {"resume", "rerun_incorrect", "preflight", "metrics_only", "output", "mlflow", "run_name",
                "concurrency", "worker_timeout", "worker_close_timeout", "worker_retries",
                "regrade_from", "replay_from", "merge_from", "explicit", "_reused_builds"}
    config = {k: v for k, v in options.items() if k not in excluded}
    files = {str(e.history_ref["path"]) for e in examples if "path" in e.history_ref}
    identity = {"version": VERSION, "config": config, "build_dependencies": dependencies,
                "questions": [asdict(e) for e in examples],
                "data_checksums": {p: file_digest(Path(p)) for p in sorted(files)},
                "protocol_code": factory_digest(resolve("protocol", options["protocol"])),
                "dataset_code": factory_digest(resolve("dataset", options["dataset"])),
                "backend_code": factory_digest(resolve("backend", options["backend"])),
                "runner_code": file_digest(Path(__file__)),
                "prompts_code": file_digest(Path(__file__).with_name("prompts.py")),
                "backend_runtime": getattr(resolve("backend", options["backend"]), "runtime_dependencies",
                                           lambda _: {})(options),
                **evaluation_services(options),
                "plugins": {m: file_digest(Path(importlib.import_module(m).__file__)) for m in options["plugins"]}}
    manifest = {"run_id": fingerprint(identity), "identity": identity}
    metadata = getattr(resolve("protocol", options["protocol"]), "reporting_metadata", None)
    if metadata:
        manifest["judge_reporting"] = metadata()
        identity["judge_reporting"] = manifest["judge_reporting"]
        manifest["run_id"] = fingerprint(identity)
    return manifest


def prepare_run(options: dict):
    load_plugins(options["plugins"])
    dataset = resolve("dataset", options["dataset"])()
    protocol = resolve("protocol", options["protocol"])()
    accepts = getattr(protocol, "accepts_target", lambda _: True)
    scanned = [example for example in dataset.scan(options) if accepts(example.target)]
    # Governance construction takes distinct build IDs in prior-result order.
    # Reusing that same source must preserve the exact cohort, not merely the
    # same category's dataset order.
    if (
        options.get("reuse_builds_from")
        and options.get("question_types")
        and not options.get("questions")
        and not options.get("stratified")
    ):
        examples = _select_examples_from_reuse_source(scanned, options)
    else:
        examples = select_examples(scanned, options)
    if not examples:
        raise ValueError("No selected questions")
    if len({e.id for e in examples}) != len(examples) or any(not e.id for e in examples):
        raise ValueError("Dataset adapter returned duplicate/empty question IDs")
    backend = resolve("backend", options["backend"])
    backend.validate(options, examples)
    # Validate protocol categories/dependencies before any worker or service call.
    for example in examples:
        protocol.score("", example.target)
    deps = backend.dependencies(options)
    if options.get("reuse_builds_from"):
        options["_reused_builds"] = _load_reused_builds(options, examples)
    return examples, deps, run_manifest(options, examples, deps)


def budget_context(context: str, budget: int | None) -> tuple[str, dict]:
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    # 预算只裁剪交给 Reader 的文本；原始检索上下文仍保存在结果中用于审计。
    tokens = encoding.encode(context, disallowed_special=())
    used = tokens[:budget] if budget is not None else tokens
    return encoding.decode(used), {"context_estimated_tokens": len(tokens),
                                   "reader_context_estimated_tokens": len(used),
                                   "context_truncated": len(used) != len(tokens),
                                   "context_token_estimator": "cl100k_base"}


async def score_record(record: dict, target: Target, query: Query, protocol, reader, options: dict,
                       regrade: bool = False) -> dict:
    question = protocol.question(query, options["question_date_text"])
    record["question"] = question
    stage = "reader"
    try:
        if not regrade:
            context, stats = budget_context(record["context"], options["context_tokens"])
            record.update(stats, reader_context=context)
            call = await asyncio.wait_for(reader.respond(context, question, system_prompt=protocol.reader_system),
                                          options["model_timeout"])
            record.update(hypothesis=call.hypothesis, reader_duration=call.duration,
                          reader_prompt_tokens=call.prompt_tokens, reader_completion_tokens=call.completion_tokens)
        stage = "metrics"
        record["metrics"] = protocol.score(record["hypothesis"], target)
        record.update(record["metrics"])
        stage = "judge"
        grade = await asyncio.wait_for(reader.grade(question, target.answer, record["hypothesis"], target.category,
                                       prompt_override=protocol.judge_prompt(question, record["hypothesis"], target),
                                       **getattr(protocol, "judge_options", dict)()),
                                       options["model_timeout"])
        record.update(grade=grade.is_correct, grader_prompt_tokens=grade.prompt_tokens,
                      grader_completion_tokens=grade.completion_tokens, status="succeeded")
    except Exception as exc:  # noqa: BLE001 - preserve per-stage failures without losing results
        record.update(error=f"{type(exc).__name__}: {exc}", failure_stage=stage, grade=None, status="failed")
    return record


async def run(options: dict, prepared=None, *, tracking: ExperimentTracking | None = None) -> dict:
    from .progress import TerminalProgress
    from .reader_grader import ReaderGrader, aggregate_metrics, evidence_metrics
    from .reporting import make_reporter

    examples, deps, manifest = prepared or prepare_run(options)
    store = ResultStore(Path(options["output"]), manifest, options["resume"] or options["rerun_incorrect"])
    reporter = make_reporter(manifest, store.records.values())
    protocol = resolve("protocol", options["protocol"])()
    try:
        reader = None if options["retrieval_only"] else ReaderGrader(options["reader_model"], options["judge_model"])
    except BaseException:
        store.close()
        raise
    groups = {}
    # 续跑跳过已成功题；同一构建身份的剩余问题共用 worker，组内顺序执行。
    for example in examples:
        previous = store.records.get(example.id)
        if options["rerun_incorrect"]:
            if not previous or previous.get("status") != "succeeded" or previous.get("grade") is not False:
                continue
        elif options["resume"] and previous and previous.get("status") == "succeeded":
            continue
        spec = make_build(example, options, deps)
        groups.setdefault(spec.id, (spec, []))[1].append(example)
    semaphore = asyncio.Semaphore(options["concurrency"])
    progress = TerminalProgress(len(examples), sum(len(members) for _, members in groups.values()),
                                len(groups), options)
    progress.start()
    summary = None

    async def group(spec, members):
        async with semaphore:
            worker = None
            state_path = Path(options["output"]).with_suffix(".states") / f"{spec.id}.json"

            def state(stage, **extra):
                atomic_json(state_path, {"build_id": spec.id, "stage": stage, "updated_at": time.time(),
                                        "worker_pid": worker.process.pid if worker and worker.process else None,
                                        "build_directory": str(spec.directory), **extra})

            try:
                for example in members:
                    record = {"question_id": example.id, "corpus_id": example.corpus_id, "build_id": spec.id,
                              "run_id": manifest["run_id"], "backend": options["backend"],
                              "protocol": options["protocol"], "question_type": example.target.category,
                              "query": asdict(example.query), "target": asdict(example.target),
                              "gold_answer": example.target.answer, "is_abstention": example.target.abstention,
                              "retrieval_only": options["retrieval_only"], "grade": None}
                    for attempt in range(options["worker_retries"] + 1):
                        try:
                            if worker is None:
                                worker = BackendProcess(options)
                                state("preparing")
                                await worker.start(example, spec)
                            state("retrieving", question_id=example.id)
                            text = protocol.question(example.query, options["question_date_text"])
                            start = time.monotonic()
                            result = await worker.retrieve(Query(text, example.query.date_raw, example.query.timestamp))
                            record.update(context=result.context, evidence=result.items, backend_stats=result.stats,
                                          retrieval_duration=time.monotonic() - start,
                                          **evidence_metrics(example.target, result))
                            record["attempt"] = attempt + 1
                            break
                        except Exception as exc:  # noqa: BLE001 - preserve per-stage failures without losing results
                            if worker:
                                await worker.close()
                            worker = None
                            if attempt == options["worker_retries"]:
                                record.update(error=f"{type(exc).__name__}: {exc}",
                                              failure_stage=(getattr(exc, "stage", None) or "prepare"), status="failed")
                            else:
                                progress.retry(example.id, attempt + 2, exc)
                    if not record.get("error"):
                        if reader is not None:
                            state("scoring", question_id=example.id)
                            await score_record(record, example.target, example.query, protocol, reader, options)
                        else:
                            context, stats = budget_context(record["context"], options["context_tokens"])
                            record.update(stats, reader_context=context, status="succeeded",
                                          question=protocol.question(example.query, options["question_date_text"]))
                    record["evaluated_at"] = datetime.now(UTC).isoformat()
                    store.append(record)
                    snapshot = reporter.update(record) if reporter else None
                    progress.result(record, reporter.terminal(snapshot) if reporter else None)
                    if tracking:
                        await tracking.record(record, extra_metrics=(
                            reporter.metrics(snapshot, "live") if reporter else None))
            finally:
                if worker:
                    state("closing")
                    await worker.close()
                state("completed")

    try:
        async with asyncio.TaskGroup() as tasks:
            for spec, members in groups.values():
                tasks.create_task(group(spec, members))
        summary = aggregate_metrics(list(store.records.values()), manifest)
        atomic_json(Path(options["output"]).with_suffix(".summary.json"), summary)
        return summary
    finally:
        progress.finish(summary)
        store.close()
        if reader:
            await reader.aclose()


def saved_evaluation_input(source: Path, options: dict) -> tuple[dict, dict]:
    """Read a manifest-backed run or the historical LongMemEval JSONL format."""
    rows = latest_records(read_records(source))
    if not rows:
        raise ValueError("Saved-result evaluation requires a nonempty source")
    manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    if manifest_path.exists():
        return rows, json.loads(manifest_path.read_text())
    # 只有明确符合旧版 LongMemEval 结构的 JSONL 才能无 manifest 导入。
    if options["protocol"] != "longmemeval":
        raise ValueError("Historical JSONL without a manifest requires the longmemeval protocol")
    source_id = "legacy-jsonl:" + file_digest(source)
    for row in rows.values():
        if "target" in row or "query" in row or not all(
            key in row for key in ("question", "gold_answer", "question_type", "condition")
        ):
            raise ValueError("Source manifest missing; expected historical LongMemEval records")
        # The historical question already includes its date. Preserve it byte
        # for byte instead of adding a second date or reconstructing the text.
        row.update(query=asdict(Query(row["question"])),
                   target=asdict(Target(row["gold_answer"], row["question_type"],
                                        bool(row.get("is_abstention")),
                                        tuple(row.get("answer_session_ids") or []))),
                   run_id=source_id)
    config = {key: options[key] for key in (
        "dataset", "split", "protocol", "plugins", "reader_model", "judge_model", "model_timeout",
    )}
    config.update(backend="saved_context", question_date_text=False, context_tokens=None)
    return rows, {"run_id": source_id, "identity": {"config": config},
                  "question_order": list(rows), "source_format": "historical_longmemeval"}


async def regrade(options: dict, *, tracking: ExperimentTracking | None = None) -> dict:
    from .mlflow_metrics import ordered_records
    from .progress import TerminalProgress
    from .reader_grader import ReaderGrader, aggregate_metrics
    from .reporting import make_reporter

    replay = bool(options.get("replay_from"))
    source = Path(options.get("replay_from") or options["regrade_from"]).resolve()
    output = Path(options["output"]).resolve()
    if source == output:
        raise ValueError("Saved-result evaluation output must differ from source")
    rows, source_manifest = saved_evaluation_input(source, options)
    saved = source_manifest["identity"]["config"]
    effective = {**saved, **{k: v for k, v in options.items() if k in
                 ("output", "resume")}}
    # Saved-result evaluation always scores answers, even when its source only retrieved context.
    effective["retrieval_only"] = False
    # 回放沿用源实验配置，只有显式传入的参数才能覆盖，避免 CLI 默认值悄悄改变口径。
    for key in ("protocol", "question_date_text", "judge_model", "reader_model", "model_timeout"):
        if key in options.get("explicit", []):
            effective[key] = options[key]
    effective["plugins"] = list(dict.fromkeys([*saved.get("plugins", []), *options.get("plugins", [])]))
    load_plugins(effective["plugins"])
    protocol = resolve("protocol", effective["protocol"])()
    source_records = len(rows)
    accepts = getattr(protocol, "accepts_target", lambda _: True)
    rows = {qid: row for qid, row in rows.items() if accepts(Target(**row["target"]))}
    if not rows:
        raise ValueError("No saved questions accepted by the evaluation protocol")
    identity = {"config": effective, "source_sha256": file_digest(source), "source_run": source_manifest["run_id"],
                "protocol_code": factory_digest(type(protocol)),
                "prompts_code": file_digest(Path(__file__).with_name("prompts.py")),
                "evaluator_code": file_digest(Path(__file__)),
                **evaluation_services(effective, include_extract=False)}
    if replay:
        identity["evaluation_mode"] = "reader_judge_replay"
    identity["config"] = {k: v for k, v in effective.items() if k not in {"resume", "output"}}
    manifest = {"run_id": fingerprint(identity), "identity": identity,
                "source_path": str(source), "source_records": source_records,
                "selected_records": len(rows), "excluded_records": source_records - len(rows),
                "source_format": source_manifest.get("source_format", "manifest_backed"),
                "question_order": [r["question_id"] for r in ordered_records(rows, source_manifest)]}
    manifest["reporting_questions"] = [{"id": r["question_id"], "category": str(r["question_type"])}
                                       for r in ordered_records(rows, source_manifest)]
    metadata = getattr(protocol, "reporting_metadata", None)
    if metadata:
        manifest["judge_reporting"] = metadata()
        identity["judge_reporting"] = manifest["judge_reporting"]
        manifest["run_id"] = fingerprint(identity)
    print(f"Saved source: records={source_records}; selected={len(rows)}; "
          f"excluded={source_records - len(rows)}; mode={'reader+judge' if replay else 'judge only'}; "
          "missing historical rows are not reconstructed", flush=True)
    if tracking:
        tracking.bind(manifest)
    store = ResultStore(output, manifest, options["resume"])
    reporter = make_reporter(manifest, store.records.values())
    try:
        reader = ReaderGrader(effective["reader_model"], effective["judge_model"])
    except BaseException:
        store.close()
        raise
    pending = sum(not (options["resume"] and store.records.get(qid, {}).get("status") == "succeeded")
                  for qid in rows)
    replay_concurrency = options["concurrency"] if "concurrency" in options.get("explicit", []) else 1
    progress = TerminalProgress(len(rows), pending, 1,
                                {**options, **effective, "concurrency": replay_concurrency})
    progress.start()
    summary = None
    try:
        semaphore = asyncio.Semaphore(replay_concurrency)
        result_lock = asyncio.Lock()

        async def evaluate(qid: str, original: dict) -> None:
            if options["resume"] and store.records.get(qid, {}).get("status") == "succeeded":
                return
            async with semaphore:
                record = dict(original)
                for key in ("error", "failure_stage", "bleu1", "locomo_rule_score", "metrics", "grade",
                            "grader_prompt_tokens", "grader_completion_tokens"):
                    record.pop(key, None)
                record.update(run_id=manifest["run_id"], source_run_id=original["run_id"],
                              source_grade=original.get("grade"), grade=None,
                              protocol=effective["protocol"], retrieval_only=False)
                if replay:
                    # 优先复用原先裁剪后的上下文；仅重评模式直接使用保存的答案。
                    record["source_hypothesis"] = record.pop("hypothesis", None)
                    for key in ("reader_prompt_tokens", "reader_completion_tokens", "reader_duration", "duration"):
                        record.pop(key, None)
                    record["context"] = original.get("reader_context", original.get("context"))
                saved_answer = original.get("hypothesis")
                missing = (not isinstance(record.get("context"), str)) if replay else (
                    not isinstance(saved_answer, str) or not saved_answer.strip()
                )
                if missing:
                    record.update(status="failed", error="No saved context" if replay else "No saved Reader answer",
                                  failure_stage="replay_input" if replay else "regrade_input")
                else:
                    await score_record(record, Target(**record["target"]), Query(**record["query"]),
                                       protocol, reader, effective, regrade=not replay)
                record["evaluated_at"] = datetime.now(UTC).isoformat()
                async with result_lock:
                    store.append(record)
                    snapshot = reporter.update(record) if reporter else None
                    progress.result(record, reporter.terminal(snapshot) if reporter else None)
                    if tracking:
                        await tracking.record(record, extra_metrics=(
                            reporter.metrics(snapshot, "live") if reporter else None))

        async with asyncio.TaskGroup() as tasks:
            for qid, original in rows.items():
                tasks.create_task(evaluate(qid, original))
        summary = aggregate_metrics(list(store.records.values()), manifest)
        paired = [r for r in store.records.values()
                  if isinstance(r.get("source_grade"), bool) and isinstance(r.get("grade"), bool)]
        summary["source_comparison"] = {
            "paired_records": len(paired),
            "source_correct": sum(r["source_grade"] for r in paired),
            "current_correct": sum(r["grade"] for r in paired),
            "lost": sum(r["source_grade"] and not r["grade"] for r in paired),
            "gained": sum(not r["source_grade"] and r["grade"] for r in paired),
        }
        atomic_json(output.with_suffix(".summary.json"), summary)
        return summary
    finally:
        progress.finish(summary)
        store.close()
        await reader.aclose()


def merge_results(options: dict, *, tracking: ExperimentTracking | None = None) -> dict:
    """Merge compatible shards into a new output, preserving source attribution."""
    from .mlflow_metrics import ordered_records
    from .reader_grader import aggregate_metrics
    from .reporting import make_reporter, selected_questions

    output = Path(options["output"]).resolve()
    paths = [Path(p).resolve() for p in options["merge_from"]]
    if output in paths:
        raise ValueError("Merge output must differ from every input")
    # 在创建输出前验证输入，错误必须带源文件路径，方便定位缺失或损坏的分片。
    manifests = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Merge input is missing or not a file: {path}")
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        try:
            manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            raise ValueError(f"Missing merge manifest for {path}: {manifest_path}") from None
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid merge manifest JSON at {manifest_path}: {exc}") from None
    # 分片允许选题不同，其余实验配置和数据来源必须一致。
    selection = {"start", "limit", "questions", "question_types", "stratified", "seed"}

    def comparable(manifest):
        identity = manifest["identity"]
        return {**{k: v for k, v in identity.items() if k not in {"questions", "config"}},
                "config": {k: v for k, v in identity["config"].items() if k not in selection}}

    if any(comparable(m) != comparable(manifests[0]) for m in manifests[1:]):
        raise ValueError("Cannot merge different experiment settings or data provenance")
    combined = {}
    for path, manifest in zip(paths, manifests, strict=True):
        for row in read_records(path):
            if row.get("run_id") != manifest["run_id"]:
                raise ValueError(f"Row identity mismatch in {path}")
            current = combined.get(row["question_id"])
            if current is None or row.get("evaluated_at", "") >= current.get("evaluated_at", ""):
                combined[row["question_id"]] = row
    identity = {**manifests[0]["identity"], "merge_sources": {str(p): file_digest(p) for p in paths}}
    order = list(dict.fromkeys(r["question_id"] for m in manifests for r in ordered_records(combined, m)))
    manifest = {"run_id": fingerprint(identity), "identity": identity, "question_order": order}
    manifest["reporting_questions"] = list({q["id"]: q for m in manifests
                                             for q in selected_questions(m, combined.values())}.values())
    if "judge_reporting" in manifests[0]:
        manifest["judge_reporting"] = manifests[0]["judge_reporting"]
    load_plugins(list(dict.fromkeys([
        *(plugin for m in manifests for plugin in m["identity"]["config"].get("plugins", [])),
        *options.get("plugins", []),
    ])))
    # Validate legacy category metadata before creating a partial merge output.
    make_reporter(manifest, combined.values())
    if tracking:
        tracking.bind(manifest)
    store = ResultStore(output, manifest)
    try:
        for row in combined.values():
            store.append({**row, "source_run_id": row["run_id"], "run_id": manifest["run_id"]})
        summary = aggregate_metrics(list(store.records.values()), manifest)
        atomic_json(output.with_suffix(".summary.json"), summary)
        return summary
    finally:
        store.close()
