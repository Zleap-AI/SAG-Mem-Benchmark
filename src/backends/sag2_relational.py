"""Benchmark translation over the SDK's public relational DataEngine API."""

from __future__ import annotations

import os
from contextvars import ContextVar
from pathlib import Path

from sqlalchemy.engine import make_url
from zleap.sag2 import (
    ChunkItem,
    ChunkSet,
    ChunkStats,
    DataEngine,
    EngineConfig,
    RelationalConfig,
    SearchOptions,
    SearchScope,
    SourceDescriptor,
)

from ..harness.benchmark import atomic_json


def mysql_config() -> RelationalConfig:
    value = os.environ.get("SAG2_MYSQL_URL")
    if not value:
        raise ValueError("Configure SAG2_MYSQL_URL for the sag2 MySQL backend")
    return RelationalConfig(provider="mysql", url=value, pool_size=5, max_overflow=0)


def mysql_identity() -> dict:
    """Record the destination without serializing authentication credentials."""
    url = make_url(mysql_config().url.get_secret_value())
    return {"provider": "mysql", "host": url.host, "port": url.port or 3306, "database": url.database}


class RelationalCorpus:
    """Own no SQL or scoring implementation: translate to SDK contracts only."""

    def __init__(self, config: RelationalConfig, namespace: str) -> None:
        active_scorer = ContextVar("sag2_benchmark_scorer")
        self._active_scorer = active_scorer

        class Scorer:
            async def score(self, *args):
                return await active_scorer.get().score(*args)

        self.engine = DataEngine(EngineConfig(relational=config), namespace, scorer=Scorer())
        self.scope = SearchScope(data_source_ids=(namespace,))
        self.total = 0

    async def prepare(self, corpus: list[dict], history_digest: str, marker: Path, *, reused: bool) -> None:
        import json

        await self.engine.start()
        try:
            if reused:
                saved = json.loads(marker.read_text())
                state = await self.engine.source_state("corpus")
                if state is None or state.status != "ready" or (
                    state.generation_id != saved["version_id"]
                    or state.source_version != history_digest
                    or state.chunk_count != len(corpus)
                ):
                    raise ValueError("Verified SAG2 remote corpus is missing or changed")
                self.total = state.chunk_count
                return
            chunks = tuple(ChunkItem(
                client_key=str(i), rank=i, content=document["text"],
                metadata={"title": document["title"], "original_document_id": str(i),
                          "original_text": document["text"], "benchmark_position": i,
                          "benchmark_document_id": i},
            ) for i, document in enumerate(corpus, 1))
            result = await self.engine.index(ChunkSet(
                descriptor=SourceDescriptor(data_source_id=self.scope.data_source_ids[0],
                                            source_id="corpus", source_type="text", title="Conversation"),
                source_version=history_digest, chunk_version="benchmark-complete-passages-v1",
                chunks=chunks, stats=ChunkStats(chunk_count=len(chunks)),
            ))
            self.total = len(result.chunk_ids)
            atomic_json(marker, {"namespace": self.scope.data_source_ids[0], "source_id": "corpus",
                                 "history_digest": history_digest, "version_id": result.version_id,
                                 "chunk_count": self.total})
        except BaseException:
            await self.engine.aclose()
            raise

    async def search(self, query: str, *, scorer, **options) -> dict:
        token = self._active_scorer.set(scorer)
        try:
            result = await self.engine.search(query, self.scope,
                SearchOptions(top_k=options["candidate_k"], **options))
        finally:
            self._active_scorer.reset(token)
        return {"candidates": [{"id": int(hit.document_id), "document_id": int(hit.document_id),
                                "content": hit.content, "jev_score": hit.score, "hop": hit.hop}
                               for hit in result.chunks],
                "trace": list(result.trace), "jev_usage": result.usage,
                "elapsed_ms": result.elapsed_ms, "algorithm": result.algorithm}

    async def aclose(self) -> None:
        await self.engine.aclose()
