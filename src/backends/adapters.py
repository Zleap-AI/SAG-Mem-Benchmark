"""Lazy unified adapters around the existing retrieval algorithms."""

from __future__ import annotations

import importlib.metadata
import json
from datetime import datetime

from ..harness.benchmark import (
    BuildSpec,
    History,
    Query,
    RetrievalResult,
    atomic_json,
    fingerprint,
)


def version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def storage_session_ids(history: History) -> list[str]:
    """Disambiguate repeated source labels without overwriting another session."""
    # 同名会话可能有不同日期；只为存储生成唯一 ID，评测仍使用数据集原始标签。
    reserved = {s.id for s in history.sessions}
    used = set()
    ids = []
    for position, session in enumerate(history.sessions):
        sid = session.id
        if sid in used:
            sid = f"{session.id}#occurrence-{position}"
            while sid in reserved or sid in used:
                sid += "#"
        used.add(sid)
        ids.append(sid)
    return ids


def legacy_history(history: History) -> dict:
    return {
        "haystack_sessions": [[{"role": m.speaker, "speaker": m.speaker, "content": m.content, "id": m.id}
                               for m in s.messages] for s in history.sessions],
        "haystack_session_ids": storage_session_ids(history),
        "haystack_dates": [datetime.fromisoformat(s.timestamp).strftime("%Y/%m/%d %H:%M")
                           if s.timestamp else "" for s in history.sessions],
    }


def manifest_matches(spec: BuildSpec) -> bool:
    path = spec.directory / "build.json"
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    if value.get("build_id") != spec.id or fingerprint(value.get("dependencies")) != fingerprint(spec.dependencies):
        raise ValueError(f"Build manifest mismatch: {path}")
    return value.get("status") == "succeeded"


def publish(spec: BuildSpec, **extra) -> None:
    atomic_json(spec.directory / "build.json", {"status": "succeeded", "build_id": spec.id,
                                                "dependencies": spec.dependencies, **extra})


def reject_options(options: dict, names: set[str]) -> None:
    unsupported = names & set(options.get("explicit", []))
    if unsupported:
        raise ValueError(f"Unsupported backend options: {sorted(unsupported)}")


class FullContext:
    def __init__(self, options: dict):
        self.options = options

    @classmethod
    def validate(cls, options: dict, examples: list) -> None:
        reject_options(options, {"strategy", "return_type", "evidence_mode", "extraction_profile", "graph_limit"})
        if options.get("time_filter", "none") != "none":
            raise ValueError("time_filter applies only to sag_time")

    @classmethod
    def dependencies(cls, options: dict) -> dict:
        return {"family": "full_context", "version": 1}

    @classmethod
    def runtime_dependencies(cls, options: dict) -> dict:
        return {"context_format": "all_visible_messages_v1"}

    async def prepare(self, history: History, build_spec: BuildSpec) -> None:
        self.history = history

    async def retrieve(self, query: Query, options: dict) -> RetrievalResult:
        # Full Context 返回全部可见消息；长度限制统一交给 Reader 上下文预算处理。
        items = [{"content": f"{m.speaker} (date: {s.date_raw or 'unknown'}): {m.content}",
                  "sources": [{"session_id": s.id, "message_id": m.id}]}
                 for s in self.history.sessions for m in s.messages]
        return RetrievalResult("\n".join(item["content"] for item in items), items,
                               source_granularity="message")

    async def aclose(self) -> None:
        pass
