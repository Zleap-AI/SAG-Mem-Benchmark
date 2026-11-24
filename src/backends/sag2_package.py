"""Persistence glue for the pinned zleap-sag2 retrieval components.

The benchmark supplies complete chunks/events and its own source mapping. Keep
that build contract while delegating tokenization, BM25, expansion, DPP, and JEV
to the installed package. Its internal search API is covered by integration tests
and an exact dependency pin; no external checkout or copied algorithm is used.
"""

from __future__ import annotations

from pathlib import Path

import aiohttp
from zleap.sag2 import LanceIndex, LancePassage, SQLiteIndex  # noqa: F401
from zleap.sag2._algorithm import chunk_text, search_sag2, tokenize  # noqa: F401
from zleap.sag2._jev_score import compact_json
from zleap.sag2.config import JevConfig
from zleap.sag2.jev import JevClient
from zleap.sag2.lance import (
    LANCE_SCHEMA_VERSION,  # noqa: F401
    normalize_passage,
)

normalize_chunk = normalize_passage


def _split_jev_text(text: str) -> tuple[str, str]:
    middle = len(text) // 2
    boundaries = [position + 1 for position in range(max(0, middle - 200), min(len(text) - 1, middle + 200))
                  if text[position].isspace()]
    boundary = min(boundaries, key=lambda position: abs(position - middle)) if boundaries else middle
    return text[:boundary], text[boundary:]


class LongTextJevClient(JevClient):
    """Use short rule references and retain complete passages, including long inputs."""

    def _stateful_batches(self, state_data: dict, inline_questions: dict) -> list[tuple]:
        batches = super()._stateful_batches(state_data, inline_questions)
        shortened = []
        for state, questions, estimated in batches:
            compact_questions = {
                key: {**question, "instructions": f"Apply the shared evidence-scoring rule to candidate {number}."}
                for number, (key, question) in enumerate(questions.items(), 1)
            }
            # Preserve the SDK's batch boundaries and shared state. Recursive batches
            # can already contain short references; their token delta is then zero.
            saved = sum(
                len(self.encoding.encode(compact_json(questions[key]), disallowed_special=()))
                - len(self.encoding.encode(compact_json(question), disallowed_special=()))
                for key, question in compact_questions.items()
            )
            shortened.append((state, compact_questions, estimated - saved))
        return shortened

    async def score(self, query: str, candidates: list[dict], evidence: list[dict]) -> tuple[list[dict], dict]:
        try:
            return await super().score(query, candidates, evidence)
        except ValueError as error:
            if not (str(error).startswith("Jev candidate ") and "exceeds 32000-token context budget" in str(error)
                    or str(error).startswith("JEV candidate and evidence exceed 32000-token context budget")):
                raise
            longest = max(candidates, key=lambda candidate: len(candidate["content"]))
            evidence_size = sum(len(item["content"]) for item in evidence)
            if evidence_size > len(longest["content"]) and evidence_size > 1:
                if len(evidence) > 1:
                    middle = min(range(1, len(evidence)), key=lambda position: abs(
                        2 * sum(len(item["content"]) for item in evidence[:position]) - evidence_size))
                    windows = [evidence[:middle], evidence[middle:]]
                else:
                    windows = [[{**evidence[0], "content": part}]
                               for part in _split_jev_text(evidence[0]["content"])]
                results = await self._gather([self.score(query, candidates, window) for window in windows])
                usage = {key: sum(result[1][key] for result in results)
                         if all(result[1][key] is not None for result in results) else None
                         for key in ("requests", "judgments", "input_tokens", "output_tokens")}
                usage["batches"] = [batch for _, stats in results for batch in stats["batches"]]
                usage["batches"][0].setdefault("long_evidence_splits", []).append({
                    "window_characters": [sum(len(item["content"]) for item in window) for window in windows],
                    "score_aggregation": "max",
                })
                return [max(pair, key=lambda item: item["jev_score"])
                        for pair in zip(*(scored for scored, _ in results))], usage
            if len(longest["content"]) < 2:
                raise
            parts = [{**longest, "id": f"{longest['id']}__part{number}", "content": part}
                     for number, part in enumerate(_split_jev_text(longest["content"]))]
            expanded = [part for candidate in candidates for part in (parts if candidate is longest else [candidate])]
            scored, usage = await self.score(query, expanded, evidence)
            by_id = {item["id"]: item for item in scored}
            best = max((by_id[part["id"]] for part in parts), key=lambda item: item["jev_score"])
            by_id[longest["id"]] = {**longest, "jev_score": best["jev_score"], "jev_model": best["jev_model"]}
            usage["batches"][0].setdefault("long_candidate_splits", []).append({
                "candidate_id": longest["id"], "part_ids": [part["id"] for part in parts],
                "part_characters": [len(part["content"]) for part in parts],
                "part_scores": [by_id[part["id"]]["jev_score"] for part in parts], "score_aggregation": "max",
            })
            return [by_id[candidate["id"]] for candidate in candidates], usage


def lance_passages(corpus: list[dict], mapping: dict[str, dict]) -> list[LancePassage]:
    """Translate benchmark provenance into the package's public passage model."""
    return [LancePassage(title=document["title"], content=mapping[str(position)]["content"],
                         source_id=str(document.get("idx", position)),
                         metadata={key: value for key, value in mapping[str(position)].items()
                                   if key not in ("content", "indexed_content")})
            for position, document in enumerate(corpus, 1)]


def _prepare_rows(corpus: list[dict]) -> list[LancePassage]:
    return [LancePassage(title=document["title"].strip(), content=document["text"].strip(),
                         source_id=str(document.get("idx", position)))
            for position, document in enumerate(corpus, 1)]


class CorpusIndex:
    """Benchmark construction contract; all persistence and retrieval live in the SDK."""

    def __init__(self, path: Path, corpus: list[dict], history_digest: str) -> None:
        self._index = (SQLiteIndex.open(path, corpus_digest=history_digest) if path.exists()
                       else SQLiteIndex.create(path, _prepare_rows(corpus), corpus_digest=history_digest))

    def __getattr__(self, name):
        return getattr(self._index, name)


def make_jev_client(session: aiohttp.ClientSession | None, service: dict, key: str) -> JevClient:
    from src.config import settings

    return LongTextJevClient(
        JevConfig(
            endpoint=service["endpoint"],
            model=service["model"],
            api_key=key,
            timeout_seconds=service["timeout_seconds"],
            payload_style=service["payload_style"],
            max_concurrency=max(1, min(settings.jev_concurrency, 100)),
        ),
        session=session,
    )
