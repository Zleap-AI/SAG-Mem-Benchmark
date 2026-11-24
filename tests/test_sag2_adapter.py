import json
from types import SimpleNamespace

import pytest

from src.backends import sag2_adapter as sa
from src.backends.raw_chunks import chunk_history
from src.harness import benchmark as b


@pytest.fixture
def package_core():
    pytest.importorskip("zleap.sag2")



def options(**extra):
    return {"backend_options": {"max_hops": 1},
            "explicit": [], "time_filter": "none", "top_k": 2, **extra}


def history():
    return b.History("c", (
        b.Session("s1", (b.Message("m1", "Alice", "I adopted two cats.\r\n\r\n\r\nFrom Cedar Shelter.  "),), "2023-05-08"),
        b.Session("s2", (b.Message("m2", "Bob", "Cedar Shelter is located in Boston."),), "2023-05-09"),
        b.Session("s3", (b.Message("m3", "Alice", "I grow apples."),), "2023-05-10"),
    ))


def spec(tmp_path, backend, source):
    dependencies = backend.dependencies(backend.options)
    return b.BuildSpec(b.fingerprint({"history": source.digest, "dependencies": dependencies}),
                       "sag2", str(tmp_path), dependencies)


class FakeJev:
    def __init__(self):
        self.calls = []

    async def score(self, query, candidates, evidence):
        self.calls.append((query, candidates, evidence))
        return ([{**item, "jev_score": 0.9 if "Shelter" in item["content"] else 0.1}
                 for item in candidates],
                {"requests": 1, "judgments": len(candidates), "input_tokens": 100,
                 "output_tokens": 10, "batches": []})


def test_service_uses_project_credentials_without_serializing_key(monkeypatch):
    monkeypatch.setattr(sa, "settings", SimpleNamespace(
        jev_api_key="own-project-key", jev_base_url="https://jev.example/v1/", jev_model="own-model"))
    service, key = sa._service({})
    assert key == "own-project-key"
    assert service == {"endpoint": "https://jev.example/v1/systemone",
                       "model": "own-model", "timeout_seconds": 300, "payload_style": "stateful"}
    assert key not in json.dumps(service)


def test_missing_project_key_never_uses_external_or_generic_credentials(tmp_path, monkeypatch):
    external = tmp_path / ".env"
    external.write_text("JEV_API_KEY=external-project-key\nJEV_BASE_URL=https://external.example/v1\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "generic-router-key")
    monkeypatch.setenv("OPENAI_API_KEY", "generic-openai-key")
    monkeypatch.setattr(sa, "settings", SimpleNamespace(
        jev_api_key="", jev_base_url="https://jev.example/v1", jev_model="own-model"))
    with pytest.raises(ValueError, match=r"your own JEV_API_KEY in sag_mem/\.env"):
        sa._service({"source_dir": str(tmp_path), "jev_env_file": str(external)})


def test_config_rejects_external_env_file(tmp_path):
    opts = options(backend_options={"source_dir": str(tmp_path),
                                                 "jev_env_file": str(tmp_path / ".env")})
    with pytest.raises(ValueError, match="no longer reads jev_env_file"):
        sa._config(opts)


def test_installed_package_ignores_legacy_source_locations(package_core, monkeypatch):
    monkeypatch.setenv("SAG2_SOURCE_DIR", "/missing/old/checkout")
    plain = options()
    legacy = options(backend_options={"source_dir": "/another/missing/path", "max_hops": 1})
    assert sa._config(plain) == sa._config(legacy)
    assert sa.SAG2Backend.dependencies(plain) == sa.SAG2Backend.dependencies(legacy)


def test_incompatible_package_version_fails_before_build(monkeypatch):
    monkeypatch.setattr(sa, "version", lambda _: "0.3.0")
    with pytest.raises(ValueError, match="expected zleap-sag2==0.4.0"):
        sa._source({})


def test_payload_style_changes_runtime_identity_without_rebuilding(package_core, monkeypatch):
    monkeypatch.setattr(sa, "settings", SimpleNamespace(
        jev_api_key="private", jev_base_url="https://unused.invalid", jev_model="fake"))
    stateful = options()
    inline = options(backend_options={"max_hops": 1, "jev_payload_style": "inline"})
    assert sa.SAG2Backend.dependencies(stateful) == sa.SAG2Backend.dependencies(inline)
    assert sa.SAG2Backend.runtime_dependencies(stateful) != sa.SAG2Backend.runtime_dependencies(inline)
    assert "private" not in json.dumps(sa.SAG2Backend.runtime_dependencies(stateful))


@pytest.mark.parametrize("endpoint", ["https://jev.example/v1", "https://jev.example/v1/systemone",
                                      "https://jev.example/v1/decisions"])
def test_nonsecret_service_overrides_keep_project_key(monkeypatch, endpoint):
    monkeypatch.setattr(sa, "settings", SimpleNamespace(
        jev_api_key="own-project-key", jev_base_url="https://default.example/v1", jev_model="default"))
    service, key = sa._service({"jev_base_url": endpoint, "jev_model": "override",
                               "jev_timeout_seconds": 60})
    expected = endpoint if endpoint.endswith(("/systemone", "/decisions")) else endpoint + "/systemone"
    assert service == {"endpoint": expected, "model": "override", "timeout_seconds": 60, "payload_style": "stateful"}
    assert key == "own-project-key"


def test_duplicate_mapping_preserves_all_sources_and_raw_text():
    source = b.History("c", (
        b.Session("s1", (b.Message("m1", "A", "same  "),)),
        b.Session("s2", (b.Message("m2", "A", "same"),)),
    ))
    chunks = chunk_history(source)
    corpus, mapping = sa._corpus(chunks)
    assert len(corpus) == 1
    assert corpus[0]["text"] == chunks[0]["content"].strip()
    assert mapping["1"]["content"] == chunks[0]["content"]
    assert mapping["1"]["sources"] == [{"session_id": "s1", "message_id": "m1"},
                                         {"session_id": "s2", "message_id": "m2"}]
    assert len(mapping["1"]["source_chunk_ids"]) == 2


def test_source_prefix_does_not_expose_future_messages():
    corpus, mapping = sa._corpus(chunk_history(history().prefix(1)))
    assert len(corpus) == 1
    assert "Boston" not in json.dumps([corpus, mapping])


def test_registry_and_validation(package_core, monkeypatch):
    monkeypatch.setattr(sa, "_service", lambda config: ({"model": "fake", "endpoint": "local"}, "secret"))
    opts = options()
    assert b.resolve("backend", "sag2") is sa.SAG2Backend
    sa.SAG2Backend.validate(opts, [])
    for invalid in ({"max_hops": 3}, {"candidate_k": True}, {"seed_k": 0},
                    {"unknown": 1}, {"jev_timeout_seconds": float("nan")}):
        with pytest.raises(ValueError):
            sa.SAG2Backend.validate({**opts, "backend_options": {**opts["backend_options"], **invalid}}, [])
    with pytest.raises(ValueError):
        sa.SAG2Backend.validate({**opts, "explicit": ["strategy"]}, [])
    with pytest.raises(ValueError):
        sa.SAG2Backend.validate({**opts, "time_filter": "query_time"}, [])


def test_fingerprints_separate_index_and_retrieval(package_core, tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "_service", lambda config: ({"model": "fake", "endpoint": "local"}, "secret"))
    opts = options()
    deps = sa.SAG2Backend.dependencies(opts)
    other = {**opts, "top_k": 1, "reader_model": "another", "backend_options": {
        **opts["backend_options"], "max_hops": 2}}
    assert deps == sa.SAG2Backend.dependencies(other)
    assert sa.SAG2Backend.runtime_dependencies(opts) != sa.SAG2Backend.runtime_dependencies(other)
    assert "secret" not in json.dumps([deps, sa.SAG2Backend.runtime_dependencies(opts)])
    assert deps["chunking"]["tokens"] == 512
    assert deps["source"]["version"] == "0.4.0"
    assert "zleap/sag2/_algorithm.py" in deps["source"]["sha256"]


@pytest.mark.parametrize("storage", ["sqlite", "lancedb"])
@pytest.mark.asyncio
async def test_original_core_retrieval_reuse_and_message_sources(package_core, tmp_path, monkeypatch, storage):
    opts = options(backend_options={"max_hops": 1, "storage_backend": storage})
    backend = sa.SAG2Backend(opts)
    source = history()
    build = spec(tmp_path, backend, source)
    await backend.prepare(source, build)
    fake = FakeJev()
    backend.client = fake
    try:
        result = await backend.retrieve(b.Query("Where were the adopted cats from?"), backend.options)
        assert result.stats["algorithm"] == "SAG2"
        assert result.stats["jev_usage"]["judgments"] > 0
        assert any(row["stage"] == "expansion" for row in result.stats["sag2_trace"])
        assert result.source_granularity == "message"
        assert len(result.items) == 2
        assert any(item["sources"] == [{"session_id": "s1", "message_id": "m1"}] for item in result.items)
        assert all(item["content"] in [c["content"] for c in chunk_history(source)] for item in result.items)
        path = build.directory / ("sag2.sqlite" if storage == "sqlite" else "sag2.lancedb")
        checksum = sa._index_digest(path)
    finally:
        await backend.aclose()
    originals = {name: getattr(backend.core, name) for name in ("_prepare_rows", "lance_passages")}

    def forbid_rebuild(*args):
        raise AssertionError("verified index was rebuilt")

    for name in originals:
        monkeypatch.setattr(backend.core, name, forbid_rebuild)
    reopened = sa.SAG2Backend(opts)
    await reopened.prepare(source, build)
    assert reopened.build_stats["build_reused"] is True
    await reopened.aclose()
    assert sa._index_digest(path) == checksum
    for name, original in originals.items():
        monkeypatch.setattr(backend.core, name, original)
    if storage == "sqlite":
        (build.directory / "sources.json").write_text("{}")
    else:
        assert not (build.directory / "sources.json").exists()
        data = next(file for file in path.rglob("*.lance") if file.is_file())
        with data.open("ab") as stream:
            stream.write(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        await sa.SAG2Backend(opts).prepare(source, build)


@pytest.mark.asyncio
async def test_large_message_remains_one_chunk_and_failed_build_not_published(package_core, tmp_path, monkeypatch):
    source = b.History("c", (b.Session("s", (b.Message("m", "Alice", "word " * 2000),)),))
    backend = sa.SAG2Backend(options())
    build = spec(tmp_path, backend, source)
    await backend.prepare(source, build)
    assert backend.index.total == 1
    assert backend.build_stats["chunk_oversized_count"] == 1
    assert backend.index.get_passages([1])[1].indexed_content == chunk_history(source)[0]["content"].strip()
    await backend.aclose()
    other = spec(tmp_path / "failure", backend, source)
    monkeypatch.setattr(backend.core, "CorpusIndex", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fail")))
    with pytest.raises(RuntimeError, match="fail"):
        await sa.SAG2Backend(options()).prepare(source, other)
    assert not (other.directory / "build.json").exists()


@pytest.mark.parametrize("storage", ["sqlite", "lancedb"])
@pytest.mark.asyncio
async def test_empty_history_avoids_model_calls(package_core, tmp_path, storage):
    source = b.History("empty", ())
    backend = sa.SAG2Backend(options(backend_options={"storage_backend": storage}))
    await backend.prepare(source, spec(tmp_path, backend, source))
    try:
        result = await backend.retrieve(b.Query("anything"), backend.options)
        assert result.context == ""
        assert result.items == []
        assert backend.client is None
    finally:
        await backend.aclose()


@pytest.mark.asyncio
async def test_lance_and_sqlite_retrieval_parity_and_tamper_detection(package_core, tmp_path):
    source = history()
    results = []
    for storage in ("sqlite", "lancedb"):
        backend = sa.SAG2Backend(options(backend_options={"storage_backend": storage}))
        build = spec(tmp_path / storage, backend, source)
        await backend.prepare(source, build)
        backend.client = FakeJev()
        try:
            result = await backend.retrieve(b.Query("Where were the adopted cats from?"), backend.options)
            results.append((result.context, result.items, result.stats["sag2_trace"], backend.client.calls))
        finally:
            await backend.aclose()
    assert results[0] == results[1]
    data = next(file for file in (build.directory / "sag2.lancedb").rglob("*.lance") if file.is_file())
    with data.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        await sa.SAG2Backend(backend.options).prepare(source, build)


def test_storage_build_identity_and_invalid_storage(package_core):
    sqlite = sa.SAG2Backend.dependencies(options())
    lance = sa.SAG2Backend.dependencies(options(backend_options={"storage_backend": "lancedb"}))
    assert sqlite != lance
    assert lance["packages"]["lancedb"] != "not-installed"
    with pytest.raises(ValueError, match="storage_backend"):
        sa._config(options(backend_options={"storage_backend": "unknown"}))


@pytest.mark.asyncio
async def test_compact_lance_keeps_one_body_and_lossless_duplicate_sources(package_core, tmp_path):
    source = b.History("compact", (
        b.Session("s1", (b.Message("m1", "Alice", "Café 猫\r\n\r\n\r\nShelter.  "),), "2023-05-08"),
        b.Session("s2", (b.Message("m2", "Alice", "Café 猫\n\nShelter."),), "2023-05-08"),
        b.Session("s3", (b.Message("m3", "Alice", "word " * 2000),), "2023-05-09"),
    ))
    backend = sa.SAG2Backend(options(backend_options={"storage_backend": "lancedb"}))
    build = spec(tmp_path, backend, source)
    await backend.prepare(source, build)
    expected = dict(backend.mapping)
    try:
        assert len(expected) == 2
        assert len(expected["1"]["source_chunk_ids"]) == 2
        assert expected["1"]["content"] != expected["1"]["indexed_content"]
        assert backend.build_stats["chunk_oversized_count"] == 1
        table = backend.index.db.open_table("chunks")
        assert table.schema.names == ["id", "title", "content", "source_id", "tokens", "provenance", "length"]
        persisted = table.to_arrow().to_pylist()
        for row in persisted:
            original = expected[str(row["id"])]
            assert row["content"] == original["content"]
            provenance = json.loads(row["provenance"])
            assert "content" not in provenance and "indexed_content" not in provenance
            assert provenance == {key: value for key, value in original.items()
                                  if key not in ("content", "indexed_content")}
        assert not (build.directory / "sources.json").exists()
    finally:
        await backend.aclose()
    reopened = sa.SAG2Backend(backend.options)
    await reopened.prepare(source, build)
    try:
        assert reopened.mapping == expected
        assert reopened.build_stats["build_reused"] is True
    finally:
        await reopened.aclose()


@pytest.mark.asyncio
async def test_compact_lance_mapping_validation_and_failed_build_not_published(package_core, tmp_path, monkeypatch):
    backend = sa.SAG2Backend(options(backend_options={"storage_backend": "lancedb"}))
    build = spec(tmp_path, backend, history())
    await backend.prepare(history(), build)
    await backend.aclose()
    manifest = json.loads((build.directory / "build.json").read_text())
    manifest["mapping_fingerprint"] = "incorrect"
    b.atomic_json(build.directory / "build.json", manifest)
    with pytest.raises(ValueError, match="source mapping mismatch"):
        await sa.SAG2Backend(backend.options).prepare(history(), build)

    def fail(*args):
        raise RuntimeError("failed ingest")

    monkeypatch.setattr(backend.core, "lance_passages", fail)
    failed = spec(tmp_path / "failed", backend, history())
    with pytest.raises(RuntimeError, match="failed ingest"):
        await sa.SAG2Backend(backend.options).prepare(history(), failed)
    assert not (failed.directory / "build.json").exists()
