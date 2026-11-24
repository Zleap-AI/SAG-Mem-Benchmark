"""Packaged SAG2 adapter over the shared 512-token dialogue chunks."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path

import aiohttp

from ..config import settings
from ..harness.benchmark import (
    BuildSpec,
    History,
    Query,
    RetrievalResult,
    atomic_json,
    file_digest,
    fingerprint,
)
from .adapters import manifest_matches, publish, reject_options, version
from .embedding_tokenizer import EmbeddingTokenizer
from .raw_chunks import CHUNK_POLICY, CHUNK_TOKENIZER, CHUNK_TOKENS, chunk_history

_PACKAGE_VERSION = "0.4.0"
_DEFAULTS = {"candidate_k": 100, "seed_k": 10, "max_hops": 2, "expansion_term_limit": 20}
_ALLOWED = {*_DEFAULTS, "source_dir", "jev_base_url", "jev_model", "jev_timeout_seconds", "jev_payload_style",
            "storage_backend"}


def _config(options: dict) -> dict:
    given = options.get("backend_options", {})
    if "jev_env_file" in given:
        raise ValueError(
            "sag2 no longer reads jev_env_file; configure JEV_API_KEY, JEV_BASE_URL and JEV_MODEL "
            "in sag_mem/.env or the process environment"
        )
    if unknown := set(given) - _ALLOWED:
        raise ValueError(f"Unknown sag2 backend options: {sorted(unknown)}")
    # Accept legacy command lines, but never load their external source checkout.
    result = {**_DEFAULTS, "jev_payload_style": "stateful", "storage_backend": "sqlite", **given}
    result.pop("source_dir", None)
    if result["storage_backend"] not in ("sqlite", "lancedb", "mysql"):
        raise ValueError("storage_backend must be sqlite, lancedb or mysql")
    if result["jev_payload_style"] not in ("inline", "stateful"):
        raise ValueError("jev_payload_style must be inline or stateful")
    for key in _DEFAULTS:
        value = result[key]
        if type(value) is not int or value < (0 if key == "max_hops" else 1):
            raise ValueError(f"sag2 {key} must be an integer in its supported range")
    if result["max_hops"] > 2:
        raise ValueError("SAG2 supports at most two expansion rounds")
    if options.get("top_k", 20) > result["candidate_k"]:
        raise ValueError("sag2 candidate_k must be at least top_k")
    if "jev_timeout_seconds" in result:
        timeout = result["jev_timeout_seconds"]
        if type(timeout) not in (int, float) or not 0 < timeout <= 1800:
            raise ValueError("sag2 jev_timeout_seconds must be in (0, 1800]")
    return result


def _source(config: dict) -> dict:
    installed = version("zleap-sag2")
    if installed != _PACKAGE_VERSION:
        raise ValueError(f"Install SAG2 runtime with python -m pip install -r reproduction/requirements.txt; "
                         f"expected zleap-sag2=={_PACKAGE_VERSION}, got {installed}")
    return {"package": "zleap-sag2", "version": installed, "sha256": _package_digests()}


@lru_cache(maxsize=1)
def _package_digests() -> dict[str, str]:
    # The pinned install is immutable for a process; hash its ~4 MB once, not per build.
    distribution = importlib.metadata.distribution("zleap-sag2")
    files = [path for path in distribution.files or ()
             if str(path).startswith("zleap/sag2/") and path.suffix in (".py", ".json", ".tiktoken")]
    return {str(path): file_digest(Path(distribution.locate_file(path))) for path in files}


def _packages(storage: str) -> list[str]:
    packages = ["jieba", "pyuca", "regex", "tiktoken"]
    if storage == "lancedb":
        packages.extend(("lancedb", "pyarrow"))
    if storage == "mysql":
        packages.append("aiomysql")
    return packages


def _service(config: dict) -> tuple[dict, str]:
    # Explicit construction keeps credentials in sag_mem, not SDK environment defaults.
    endpoint = config.get("jev_base_url") or settings.jev_base_url
    model = config.get("jev_model") or settings.jev_model
    key = settings.jev_api_key
    if not endpoint or not key:
        raise ValueError("Configure JEV_BASE_URL and your own JEV_API_KEY in sag_mem/.env or process environment")
    endpoint = endpoint.rstrip("/")
    if not endpoint.endswith(("/decisions", "/systemone")):
        endpoint += "/systemone"
    return {"endpoint": endpoint, "model": model,
            "timeout_seconds": config.get("jev_timeout_seconds", 300),
            "payload_style": config.get("jev_payload_style", "stateful")}, key


def _load_core():
    from . import sag2_package

    return sag2_package


def _corpus(chunks: list[dict], normalize: Callable[[str], str] = str.strip) -> tuple[list[dict], dict[str, dict]]:
    """Coalesce identical core documents while preserving every original source."""
    corpus, mapping, by_content = [], {}, {}
    for chunk in chunks:
        content = normalize(chunk["content"])
        identifier = by_content.get(content)
        if identifier is None:
            identifier = str(len(corpus) + 1)
            by_content[content] = identifier
            # Source IDs stay in metadata, never in a title that affects lexical scoring.
            corpus.append({"title": "Conversation", "text": content, "idx": identifier})
            mapping[identifier] = {**chunk, "sources": list(chunk["sources"]),
                                   "message_spans": list(chunk["message_spans"]),
                                   "indexed_content": content,
                                   "source_chunk_ids": [chunk["chunk_id"]]}
        else:
            item = mapping[identifier]
            item["source_chunk_ids"].append(chunk["chunk_id"])
            item["message_spans"].extend(chunk["message_spans"])
            for source in chunk["sources"]:
                if source not in item["sources"]:
                    item["sources"].append(source)
    return corpus, mapping


def _chunk_counts(mapping: dict[str, dict]) -> dict:
    return {"chunk_count": sum(len(v["source_chunk_ids"]) for v in mapping.values()),
            "indexed_chunk_count": len(mapping),
            "chunk_max_tokens": max((v["token_count"] for v in mapping.values()), default=0),
            "chunk_oversized_count": sum(len(v["source_chunk_ids"]) for v in mapping.values()
                                         if v["token_count"] > CHUNK_TOKENS)}


def _index_digest(path: Path) -> str:
    """Verify every persisted Lance file, including names, without following symlinks."""
    if path.is_symlink() or not path.exists():
        raise ValueError("Verified SAG2 index is missing or corrupt")
    if path.is_file():
        return file_digest(path)
    files = sorted(path.rglob("*"))
    if any(file.is_symlink() for file in files):
        raise ValueError("SAG2 build paths cannot be symlinks")
    return fingerprint({str(file.relative_to(path)): file_digest(file) for file in files if file.is_file()})


class _LanceSources(Mapping):
    """Benchmark evidence mapping over package-owned, selectively fetched passages."""

    def __init__(self, index):
        self.index = index

    def get_many(self, identifiers):
        return {str(identifier): {**passage.metadata, "content": passage.content,
                                  "indexed_content": passage.indexed_content}
                for identifier, passage in self.index.get_passages(identifiers).items()}

    def __getitem__(self, key):
        return self.get_many([int(key)])[str(key)]

    def __iter__(self):
        return map(str, range(1, self.index.total + 1))

    def __len__(self):
        return self.index.total


class SAG2Backend:
    def __init__(self, options: dict) -> None:
        self.options = options
        self.config = _config(options)
        self.index = None
        self.session = None
        self.client = None

    @classmethod
    def validate(cls, options: dict, examples: list) -> None:
        reject_options(options, {"strategy", "return_type", "evidence_mode", "extraction_profile", "graph_limit"})
        if options.get("time_filter", "none") != "none":
            raise ValueError("sag2 does not implement query-time filtering")
        config = _config(options)
        _source(config)
        _service(config)
        EmbeddingTokenizer(CHUNK_TOKENIZER)
        if config["storage_backend"] == "mysql":
            from .sag2_relational import mysql_identity
            mysql_identity()
        missing = [name for name in _packages(config["storage_backend"]) if version(name) == "not-installed"]
        if missing:
            raise ValueError(f"Install SAG2 runtime dependencies: {missing}")

    @classmethod
    def dependencies(cls, options: dict) -> dict:
        config = _config(options)
        storage = config["storage_backend"]
        remote = {}
        if storage == "mysql":
            from .sag2_relational import mysql_identity
            remote = {"relational_destination": mysql_identity()}
        return {"family": "sag2", "version": 1, "source": _source(config),
                "chunking": {"tokens": CHUNK_TOKENS, "policy": CHUNK_POLICY, "budget": "soft",
                             "overlap_tokens": 0, "oversized_message": "standalone_full_text",
                             "tokenizer": EmbeddingTokenizer.dependencies()},
                "storage": f"{storage}_persistent_postings_package_bm25", "title": "Conversation", **remote,
                **({"storage_schema_version": 4, "source_mapping_storage": "lancedb_provenance_column"}
                   if storage == "lancedb" else {}),
                "duplicate_policy": "merge_sources",
                "packages": {name: version(name) for name in _packages(storage)},
                "implementation": {name: file_digest(Path(__file__).with_name(name))
                                   for name in ("sag2_adapter.py", "sag2_package.py", "sag2_relational.py", "raw_chunks.py")}}

    @classmethod
    def runtime_dependencies(cls, options: dict) -> dict:
        config = _config(options)
        service, _ = _service(config)
        return {"algorithm": "SAG2", "retrieval": {key: config[key] for key in _DEFAULTS},
                "jev": service, "source": _source(config),
                "jev_instruction_format": ("shared_rule_reference_v1" if config["jev_payload_style"] == "stateful"
                                           else "inline_full_v1")}

    async def prepare(self, history: History, build_spec: BuildSpec) -> None:
        if fingerprint(build_spec.dependencies) != fingerprint(self.dependencies(self.options)):
            raise ValueError("SAG2 build dependencies changed")
        self.core = _load_core()
        self.spec = build_spec
        build_spec.directory.mkdir(parents=True, exist_ok=True)
        if self.config["storage_backend"] == "mysql":
            await self._prepare_mysql(history, build_spec)
            return
        lance = self.config["storage_backend"] == "lancedb"
        path = build_spec.directory / ("sag2.lancedb" if lance else "sag2.sqlite")
        mapping_path = build_spec.directory / "sources.json"
        reused = manifest_matches(build_spec)
        # A verified persistent index reopens without tokenizing the history or
        # loading all text/token streams. Source records are fetched for hits.
        corpus = None
        counts = None
        if not (lance and reused):
            corpus, self.mapping = _corpus(chunk_history(history), self.core.normalize_chunk)
            counts = _chunk_counts(self.mapping)
        if reused:
            manifest = json.loads((build_spec.directory / "build.json").read_text())
            if manifest.get("history_digest") != history.digest:
                raise ValueError("SAG2 history digest mismatch")
            verified_files = [(path, "index_sha256")]
            if not lance:
                verified_files.append((mapping_path, "sources_sha256"))
            for file, key in verified_files:
                if _index_digest(file) != manifest.get(key):
                    raise ValueError("Verified SAG2 index or sources are missing or corrupt")
            if not lance and json.loads(mapping_path.read_text()) != self.mapping:
                raise ValueError("SAG2 source mapping mismatch")
        else:
            if path.is_symlink() or mapping_path.is_symlink():
                raise ValueError("SAG2 build paths cannot be symlinks")
            if lance:
                if path.exists():
                    if any(file.is_symlink() for file in path.rglob("*")):
                        raise ValueError("SAG2 build paths cannot be symlinks")
                    shutil.rmtree(path)
                mapping_path.unlink(missing_ok=True)
            else:
                for suffix in ("", "-wal", "-shm", "-journal"):
                    path.with_name(path.name + suffix).unlink(missing_ok=True)
        if lance:
            if reused:
                self.index = self.core.LanceIndex.open(path, corpus_digest=history.digest)
            else:
                self.index = self.core.LanceIndex.create(
                    path, self.core.lance_passages(corpus, self.mapping),
                    corpus_digest=history.digest, metadata=counts)
            if reused and manifest.get("mapping_fingerprint") != self.index.provenance_digest:
                self.index.close()
                self.index = None
                raise ValueError("SAG2 source mapping mismatch")
            self.mapping = _LanceSources(self.index)
            counts = self.index.metadata
        else:
            self.index = self.core.CorpusIndex(path, corpus, history.digest)
        expected_count = manifest["indexed_chunk_count"] if lance and reused else len(corpus)
        if self.index.total != expected_count or (
                not lance and not self.index.check_integrity()):
            raise ValueError("SAG2 index validation failed")
        if not reused:
            if not lance:
                atomic_json(mapping_path, self.mapping)
            publish(build_spec, history_digest=history.digest, index_sha256=_index_digest(path),
                    **({"mapping_fingerprint": self.index.provenance_digest,
                        "storage_schema_version": self.core.LANCE_SCHEMA_VERSION,
                        "source_mapping_storage": "lancedb_provenance_column"} if lance else {
                            "sources_sha256": file_digest(mapping_path)}),
                    chunk_count=counts["chunk_count"],
                    indexed_chunk_count=expected_count)
        self.build_stats = {"build_reused": reused, "chunk_size_tokens": CHUNK_TOKENS,
                            "chunk_budget": "soft", "chunk_tokenizer": CHUNK_TOKENIZER,
                            **counts, "storage_backend": build_spec.dependencies["storage"],
                            "source_mapping_storage": "lancedb_provenance_column" if lance else "sources.json",
                            "embedding_build_requests": 0,
                            "extraction_build_requests": 0}

    async def _prepare_mysql(self, history: History, build_spec: BuildSpec) -> None:
        from .sag2_relational import RelationalCorpus, mysql_config

        corpus, self.mapping = _corpus(chunk_history(history), self.core.normalize_chunk)
        marker, mapping_path = build_spec.directory / "remote.json", build_spec.directory / "sources.json"
        reused = manifest_matches(build_spec)
        if reused:
            manifest = json.loads((build_spec.directory / "build.json").read_text())
            if manifest.get("history_digest") != history.digest:
                raise ValueError("SAG2 history digest mismatch")
            for path, field in ((marker, "index_sha256"), (mapping_path, "sources_sha256")):
                if _index_digest(path) != manifest.get(field):
                    raise ValueError("Verified SAG2 remote metadata or sources are corrupt")
            if json.loads(mapping_path.read_text()) != self.mapping:
                raise ValueError("SAG2 source mapping mismatch")
        from uuid import NAMESPACE_URL, uuid5
        namespace = str(uuid5(NAMESPACE_URL, "sag2-benchmark:" + build_spec.id))
        self.index = RelationalCorpus(mysql_config(), namespace)
        await self.index.prepare(corpus, history.digest, marker, reused=reused)
        counts = _chunk_counts(self.mapping)
        if not reused:
            atomic_json(mapping_path, self.mapping)
            publish(build_spec, history_digest=history.digest, index_sha256=file_digest(marker),
                    sources_sha256=file_digest(mapping_path), **counts)
        self.build_stats = {"build_reused": reused, "chunk_size_tokens": CHUNK_TOKENS,
                            "chunk_budget": "soft", "chunk_tokenizer": CHUNK_TOKENIZER, **counts,
                            "storage_backend": build_spec.dependencies["storage"],
                            "source_mapping_storage": "sources.json", "package_relational_api": "DataEngine",
                            "embedding_build_requests": 0, "extraction_build_requests": 0}

    async def retrieve(self, query: Query, options: dict) -> RetrievalResult:
        if not self.mapping or not query.text.strip():
            return RetrievalResult("", [], dict(self.build_stats), "message", "chunk")
        if self.client is None:
            service, key = _service(self.config)
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=service["timeout_seconds"]))
            self.client = self.core.make_jev_client(self.session, service, key)
        search_options = {key: self.config[key] for key in _DEFAULTS}
        if self.config["storage_backend"] in ("lancedb", "mysql"):
            result = await self.index.search(query.text, scorer=self.client, **search_options)
        else:
            result = await self.core.search_sag2(self.index.query_view(), self.client, query.text, **search_options)
        items = []
        hits = result["candidates"][:options["top_k"]]
        sources = (self.mapping.get_many(hit["document_id"] for hit in hits)
                   if self.config["storage_backend"] == "lancedb" else self.mapping)
        for hit in hits:
            source = sources[str(hit["document_id"])]
            if hit["content"] != source["indexed_content"]:
                raise ValueError("SAG2 core changed a pre-chunked dialogue passage")
            items.append({**{key: value for key, value in source.items() if key != "indexed_content"},
                          "score": hit["jev_score"], "kind": "chunk",
                          "sag2_chunk_id": hit["id"], "hop": hit["hop"]})
        stats = {**self.build_stats, "sag2_trace": result["trace"], "jev_usage": result["jev_usage"],
                 "retrieval_elapsed_ms": result["elapsed_ms"], "algorithm": result["algorithm"],
                 "jev_payload_style": self.config["jev_payload_style"]}
        return RetrievalResult("\n\n".join(item["content"] for item in items), items, stats, "message", "chunk")

    async def aclose(self) -> None:
        try:
            if self.index is not None:
                if self.config["storage_backend"] == "mysql":
                    await self.index.aclose()
                else:
                    self.index.close()
                self.index = None
        finally:
            if self.session is not None:
                await self.session.close()
                self.session = None
                self.client = None
