"""LongMemEval dataset loading."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from ..config import settings

DatasetName = Literal["oracle", "s"]

_FILES: dict[DatasetName, str] = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
}

_READ_CHUNK_SIZE = 1024 * 1024
_METADATA_KEYS = (
    "question_id",
    "question_type",
    "question",
    "answer",
    "question_date",
    "answer_session_ids",
)


def _dataset_path(name: DatasetName) -> Path:
    return settings.data_dir / _FILES[name]


def iter_dataset(name: DatasetName | Path) -> Iterator[dict]:
    """Yield top-level array items while retaining at most one item at a time."""
    path = name if isinstance(name, Path) else _dataset_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    array_started = False
    expect_separator = False
    after_separator = False
    eof = False

    def ensure_nothing_after(rest: str, stream) -> None:
        """Reject content after the closing ']'.

        ``json.load`` used to validate the whole file, so a truncated or
        twice-written dataset raised instead of silently yielding a short run.
        Only whitespace may follow, otherwise a file like '[...][...]' would
        quietly report half its instances and the accuracy computed from it
        would be attributed to the full split.
        """
        if rest.strip():
            raise ValueError(f"Trailing content after dataset array in {path}")
        while True:
            chunk = stream.read(_READ_CHUNK_SIZE)
            if not chunk:
                return
            if chunk.strip():
                raise ValueError(f"Trailing content after dataset array in {path}")

    with open(path, encoding="utf-8") as stream:
        while not eof:
            chunk = stream.read(_READ_CHUNK_SIZE)
            eof = not chunk
            buffer += chunk

            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position == len(buffer):
                    break

                if not array_started:
                    if buffer[position] != "[":
                        raise ValueError(f"Unexpected dataset format in {path}: expected an array")
                    array_started = True
                    position += 1
                    continue

                if expect_separator:
                    if buffer[position] == ",":
                        expect_separator = False
                        after_separator = True
                        position += 1
                        continue
                    if buffer[position] == "]":
                        ensure_nothing_after(buffer[position + 1 :], stream)
                        return
                    raise ValueError(f"Unexpected dataset format in {path}: expected ',' or ']'")

                if buffer[position] == "]":
                    # '[{...},]' is invalid JSON; json.load rejected it too.
                    if after_separator:
                        raise ValueError(f"Trailing comma before ']' in {path}")
                    ensure_nothing_after(buffer[position + 1 :], stream)
                    return

                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError(f"Incomplete dataset JSON in {path}") from exc
                    break
                if not isinstance(value, dict):
                    raise TypeError(f"Unexpected dataset item in {path}: expected an object")
                yield value
                position = end
                expect_separator = True

            if position:
                buffer = buffer[position:]
                position = 0

    if not array_started:
        raise ValueError(f"Unexpected dataset format in {path}: expected an array")
    raise ValueError(f"Incomplete dataset JSON in {path}")


def load_dataset(name: DatasetName) -> list[dict]:
    """Load one LongMemEval split.

    Returns a list of 500 instances, each with keys:
      question_id, question_type, question, answer, question_date,
      haystack_dates, haystack_session_ids, haystack_sessions, answer_session_ids
    """
    return list(iter_dataset(name))


def load_dataset_metadata(name: DatasetName) -> list[dict]:
    """Load only fields needed by the parent experiment scheduler.

    Backend workers independently load their selected instance. Keeping the
    full 500-instance dataset in the scheduler duplicated roughly 2 GiB of
    resident memory without helping those workers.
    """
    metadata: list[dict] = []
    for index, item in enumerate(iter_dataset(name)):
        try:
            metadata.append({key: item[key] for key in _METADATA_KEYS})
        except KeyError as exc:
            raise ValueError(f"Dataset item {index} is missing required field {exc.args[0]!r}") from exc
    return metadata


def load_instance(name: DatasetName, index: int) -> dict:
    """Read one array item without materializing the complete dataset."""
    if index < 0:
        raise IndexError(f"Dataset index must be non-negative: {index}")
    for item_index, item in enumerate(iter_dataset(name)):
        if item_index == index:
            return item
    raise IndexError(f"Dataset index out of range: {index}")


# Unified dataset adapters. Streaming scan retains metadata, never all histories.
def parse_time(raw: str) -> str | None:
    """Preserve unknown time; never substitute wall clock or a made-up date."""
    # 日期无法解析时保留未知状态，由各后端预检决定是否支持，不编造时间。
    import re
    from datetime import UTC, datetime

    if not raw:
        return None
    cleaned = re.sub(r"\s*\([^)]*\)", "", raw.strip())
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        for pattern in ("%Y/%m/%d %H:%M", "%I:%M %p on %d %B, %Y", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(cleaned, pattern).replace(tzinfo=UTC)
                break
            except ValueError:
                continue
        else:
            return None
    return parsed.replace(tzinfo=parsed.tzinfo or UTC).isoformat()


def _unique(values: list[str], label: str) -> None:
    if any(not value for value in values) or len(values) != len(set(values)):
        raise ValueError(f"Empty or duplicate {label}")


def _lme_history(item: dict):
    from ..harness.benchmark import History, Message, Session

    sessions, dates, ids = (item[k] for k in ("haystack_sessions", "haystack_dates", "haystack_session_ids"))
    if not len(sessions) == len(dates) == len(ids):
        raise ValueError(f"Mismatched history arrays for {item['question_id']}")
    if any(sid is None or not str(sid).strip() for sid in ids):
        raise ValueError("Empty session IDs")
    result = []
    for position, (turns, raw, sid) in enumerate(zip(sessions, dates, ids, strict=True)):
        raw = str(raw or "")
        timestamp = parse_time(raw)
        # LongMemEval 的填充会话可能同名但日期不同，必须保留每次出现及原始证据标签。
        # 消息 ID 加入会话位置，避免重复 session ID 导致不同消息互相覆盖。
        messages = tuple(Message(f"s{position}:{sid}#t{i}", str(turn.get("speaker") or turn["role"]),
                                 str(turn["content"]), str(turn["role"])) for i, turn in enumerate(turns))
        result.append(Session(str(sid), messages, raw, timestamp, timestamp is not None))
    # question_id 不能代表共享历史；按历史内容决定哪些问题可以复用同一索引。
    history = History("", tuple(result))
    return History(history.digest, history.sessions)


def _locomo_history(item: dict, text_mode: str):
    import re

    from ..harness.benchmark import History, Message, Session

    conv = item["conversation"]
    sessions = []
    message_ids = []
    for key in sorted((k for k in conv if re.fullmatch(r"session_\d+", k)),
                      key=lambda k: int(k.split("_")[1])):
        messages = []
        for turn in conv[key]:
            mid = str(turn["dia_id"])
            message_ids.append(mid)
            content = str(turn.get("text") or "")
            if text_mode == "text_caption" and turn.get("blip_caption"):
                content += "\n[Image caption] " + str(turn["blip_caption"])
            messages.append(Message(mid, str(turn["speaker"]), content))
        raw = str(conv.get(key + "_date_time") or "")
        timestamp = parse_time(raw)
        sessions.append(Session(key, tuple(messages), raw, timestamp, timestamp is not None))
    _unique(message_ids, "message IDs")
    return History(str(item["sample_id"]), tuple(sessions))


def _target(answer, category, abstention, evidence, known_sessions, known_messages=None):
    from ..harness.benchmark import Target

    # gold 来源缺失或无效时明确标记，不能把不可计算的覆盖率当作零分。
    evidence = evidence or []
    if not isinstance(evidence, list):
        raise TypeError("Evidence must be an array")
    evidence = list(dict.fromkeys(str(value) for value in evidence))
    valid, invalid = [], []
    for value in evidence:
        (valid if value in (known_messages if known_messages is not None else known_sessions)
         else invalid).append(value)
    sessions = [known_messages[value] for value in valid] if known_messages is not None else valid
    return Target(str(answer), str(category), abstention, tuple(dict.fromkeys(sessions)),
                  tuple(valid) if known_messages is not None else (),
                  "invalid" if invalid else "valid" if valid else "missing", tuple(invalid))


class LongMemEval:
    # scan 仅保留题目元信息和历史引用，完整历史由获得并发名额的 worker 按需加载。
    def scan(self, options: dict):
        from dataclasses import replace
        from ..harness.benchmark import Example, Query

        if options["text_mode"] != "text":
            raise ValueError("LongMemEval supports text mode only")
        path = Path(options.get("data_path") or _dataset_path(options["split"]))
        examples = []
        for index, item in enumerate(iter_dataset(path)):
            history = _lme_history(item)
            visible = history.prefix(options.get("history_prefix"))
            qid = str(item["question_id"])
            target = _target(item["answer"], item["question_type"], qid.endswith("_abs"),
                             item.get("answer_session_ids"), {s.id for s in history.sessions})
            target = replace(target, message_ids=tuple(
                message.id for session, turns in zip(history.sessions, item["haystack_sessions"], strict=True)
                for message, turn in zip(session.messages, turns, strict=True) if turn.get("has_answer") is True))
            raw = str(item.get("question_date") or "")
            examples.append(Example(qid, Query(str(item["question"]), raw, parse_time(raw)), target,
                                    {"path": str(path.resolve()), "index": index}, visible.digest,
                                    history.corpus_id, options.get("history_prefix"),
                                    any(not s.time_known for s in visible.sessions)))
        _unique([e.id for e in examples], "question IDs")
        return examples

    def load_history(self, reference: dict):
        for index, item in enumerate(iter_dataset(Path(reference["path"]))):
            if index == reference["index"]:
                return _lme_history(item)
        raise ValueError("History reference no longer exists")


class LoCoMo:
    def scan(self, options: dict):
        from ..harness.benchmark import Example, Query

        if "split" in options.get("explicit", []):
            raise ValueError("--split is a LongMemEval option, not a LoCoMo split")
        path = Path(options.get("data_path") or settings.data_dir.parent / "locomo" / "locomo10.json")
        examples, corpus_ids = [], []
        for index, item in enumerate(iter_dataset(path)):
            history = _locomo_history(item, options["text_mode"])
            corpus_ids.append(history.corpus_id)
            visible = history.prefix(options.get("history_prefix"))
            known = {m.id: s.id for s in history.sessions for m in s.messages}
            for position, qa in enumerate(item["qa"]):
                category = int(qa["category"])
                if category not in range(1, 6):
                    raise ValueError(f"Unknown LoCoMo category: {category}")
                abstention = category == 5
                # 拒答题的目标是说明信息不足，不能把干扰答案当作正确答案。
                answer = "No information available" if abstention else qa["answer"]
                target = _target(answer, category, abstention, qa.get("evidence"),
                                 {s.id for s in history.sessions}, known)
                qid = str(qa.get("question_id") or f"{history.corpus_id}:{position}")
                examples.append(Example(qid, Query(str(qa["question"])), target,
                                        {"path": str(path.resolve()), "index": index,
                                         "text_mode": options["text_mode"]}, visible.digest,
                                        history.corpus_id, options.get("history_prefix"),
                                        any(not s.time_known for s in visible.sessions)))
        _unique(corpus_ids, "corpus IDs")
        _unique([e.id for e in examples], "question IDs")
        return examples

    def load_history(self, reference: dict):
        for index, item in enumerate(iter_dataset(Path(reference["path"]))):
            if index == reference["index"]:
                return _locomo_history(item, reference["text_mode"])
        raise ValueError("History reference no longer exists")
