"""SAG2 chunk retrieval with rule-based time clarification for the Reader."""

from __future__ import annotations

import re
from dataclasses import asdict, replace
from datetime import date, datetime
from pathlib import Path

from ..data.loader import parse_time
from ..harness import relative_time
from ..harness.benchmark import Query, RetrievalResult, file_digest
from .sag2_adapter import SAG2Backend


def parse_source_date(raw: str) -> date:
    """Use the source's written calendar date, without timezone conversion."""
    chinese = re.fullmatch(r"(\d{4})年(\d{1,2})月(\d{1,2})日(?:\s+.*)?", raw.strip())
    if chinese:
        return date(*map(int, chinese.groups()))
    parsed = parse_time(raw)
    if parsed is None:
        raise ValueError(f"Cannot parse source date: {raw!r}")
    return datetime.fromisoformat(parsed).date()


def format_message_date(raw: str) -> str:
    """Normalize the written date and clock time without inventing a time."""
    displayed = parse_source_date(raw).strftime("%Y/%m/%d")
    clock = re.search(r"(\d{1,2}):(\d{2})(?::\d{2})?(?:\s*([ap]m))?", raw, re.IGNORECASE)
    if clock:
        value = f"{clock[1]}:{clock[2]}"
        pattern = "%H:%M"
        if clock[3]:
            value += " " + clock[3].upper()
            pattern = "%I:%M %p"
        displayed += datetime.strptime(value, pattern).strftime(" %H:%M")
    return displayed


def annotate_chunk(item: dict) -> tuple[str, list[dict]]:
    """Format source message headers and resolve relative times in each body."""
    if item["kind"] != "chunk":
        raise ValueError("Expected saved SAG2 chunk evidence")
    remaining = item["content"]
    original, annotated, explanations = [], [], []
    for index, span in enumerate(item["message_spans"]):
        if not remaining:  # Identical indexed text can carry multiple sets of source spans.
            break
        header = re.match(r"([^\r\n]+?) \(date: ([^\r\n]+?)\): ", remaining)
        if header is None:
            raise ValueError("Saved chunk does not match the original dated-message format")
        length = span["end"] - span["start"]
        end = header.end() + length
        if length < 0 or end > len(remaining):
            raise ValueError("Source span does not match the saved message")
        body = remaining[header.end():end]
        source_date = parse_source_date(header[2])
        new_body, matches = relative_time.add_time_explanations(body, source_date)
        original.append(remaining[:end])
        annotated.append(f"{header[1]}: [Message date: {format_message_date(header[2])}]\n{new_body}")
        explanations.extend({**asdict(m), "source_date": source_date.isoformat(),
                             "message_index": index, "chunk_id": item["chunk_id"]} for m in matches)
        remaining = remaining[end:]
        if remaining:
            if not remaining.startswith("\n"):
                raise ValueError("Saved message boundary does not match chunk text")
            remaining = remaining[1:]
    if remaining or "\n".join(original) != item["content"]:
        raise ValueError("Cannot preserve the complete original chunk")
    return "\n".join(annotated), explanations


class SAG2ChunkTimeBackend(SAG2Backend):
    """Reuse the SAG2 index and retrieval; clarify time only after selection."""

    @staticmethod
    def _with_defaults(options: dict) -> dict:
        return {**options, "backend_options": {
            "candidate_k": 50, "max_hops": 1, **options.get("backend_options", {}),
        }}

    def __init__(self, options: dict) -> None:
        super().__init__(self._with_defaults(options))

    @classmethod
    def validate(cls, options: dict, examples: list) -> None:
        super().validate(cls._with_defaults(options), examples)

    @classmethod
    def dependencies(cls, options: dict) -> dict:
        return super().dependencies(cls._with_defaults(options))

    @classmethod
    def runtime_dependencies(cls, options: dict) -> dict:
        return {**super().runtime_dependencies(cls._with_defaults(options)), "reader_context": {
            "format": "source_date_relative_time_inline_v8",
            "implementation_sha256": file_digest(Path(__file__)),
            "rules_sha256": file_digest(Path(relative_time.__file__)),
            "header_format": "{speaker}: [Message date: YYYY/MM/DD HH:MM]\\n{body}",
            "day_display": "YYYY/MM/DD", "month_display": "YYYY/MM",
            "header_weekday": False,
        }}

    async def retrieve(self, query: Query, options: dict) -> RetrievalResult:
        result = await super().retrieve(query, options)
        chunks, explanations = [], []
        for item in result.items:
            text, matches = annotate_chunk(item)
            chunks.append(text)
            explanations.extend(matches)
        return replace(result, context="\n\n".join(chunks),
                       stats={**result.stats, "time_explanations": explanations})
