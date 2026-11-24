import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from reproduction.experiments.lite.backend import LiteBackend
from reproduction.experiments.lite.jsonl_io import read_jsonl
from reproduction.experiments.shared.chunking import chunk_history
from reproduction.experiments.shared.profiles import PROFILES
from reproduction.experiments.shared.runtime import audit_chunks
from reproduction.run import ROOT, configure, parse_cli
from src.harness import benchmark
from src.harness.benchmark import History, Message, Session


def test_imports_use_repository_runtime():
    assert Path(benchmark.__file__).resolve() == Path(__file__).resolve().parents[2] / "src/harness/benchmark.py"


def test_cli_defaults_and_exact_model_profiles():
    args = parse_cli(["--method", "sag-mem-lite", "--dataset", "longmemeval"])
    assert (args.candidate_k, args.max_hops, args.top_k, args.concurrency) == (100, 2, 20, 20)
    assert args.messages_per_chunk == 3
    assert str(args.output).endswith("experiments/lite/results/longmemeval/qwen/run.jsonl")
    assert args.data_path == (ROOT / "data/longmemeval_s_cleaned.json").resolve()
    args = parse_cli(["--method", "sag-mem", "--dataset", "locomo", "--model", "deepseek"])
    assert (args.candidate_k, args.max_hops) == (50, 1)
    assert args.messages_per_chunk == 3
    assert str(args.output).endswith("experiments/main/results_deepseek/locomo/full/run.jsonl")
    assert PROFILES["deepseek"]["model"] == "deepseek/deepseek-v4.1-flash-1017"
    assert PROFILES["deepseek"]["extra_body"]["provider"]["only"] == ["DeepSeek"]
    assert all(not p["extra_body"]["reasoning"]["enabled"] for p in PROFILES.values())


def test_lite_configures_without_jev_and_preserves_encoder_options(monkeypatch):
    from src import config
    monkeypatch.setattr(config, "settings", replace(config.settings, jev_api_key="", jev_base_url=""))
    args = parse_cli(["--method", "sag-mem-lite", "--dataset", "locomo",
                      "--embedding-profile", "bge", "--embedding-overflow", "prefix"])
    options = configure(args)
    assert options["embedding_profile"] == "bge"
    assert options["embedding_overflow"] == "prefix"


def test_atomic_oversized_messages_and_sessions():
    long_text = "complete message " * 600
    history = History("h", (Session("s", (
        Message("1", "A", "before"), Message("2", "B", long_text), Message("3", "A", "after")
    ), "2026-11-05"), Session("other", (Message("4", "B", "next session"),), "2026-11-06")))
    chunks = chunk_history(history)
    assert [len(c["sources"]) for c in chunks] == [1, 1, 1, 1]
    assert chunks[1]["content"].endswith(long_text)
    assert audit_chunks(history, chunks)["messages"] == 4


def test_jsonl_unicode_and_incomplete_tail(tmp_path):
    path = tmp_path / "records.jsonl"
    record = {"text": "one\u2028two\u2029three"}
    path.write_bytes((json.dumps(record, ensure_ascii=False) + "\n{").encode())
    assert read_jsonl(path, allow_incomplete_tail=True) == [record]
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)
    path.write_bytes(b'{broken}\n')
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path, allow_incomplete_tail=True)


def test_embedding_cache_reuse_and_corruption_detection(tmp_path):
    options = {"backend_options": {}, "cache_root": str(tmp_path)}
    calls = []

    async def scenario():
        backend = LiteBackend(options)

        async def create(**kwargs):
            calls.append(kwargs)
            usage = SimpleNamespace(prompt_tokens=sum(len(backend.tokenizer.encode(t).ids)
                                                       for t in kwargs["input"]),
                                    model_dump=lambda: {"cost": 0.01})
            return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[1.0] + [0.0] * 4095)
                                         for i in range(len(kwargs["input"]))], usage=usage)

        backend.client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
        first, _ = await backend.embed(["complete original source"])
        second, usage = await backend.embed(["complete original source"])
        assert np.array_equal(first, second)
        assert usage["requests"] == 0 and len(calls) == 1
        backend.cache.execute("UPDATE vectors SET vector_hash = 'changed'")
        backend.cache.commit()
        with pytest.raises(ValueError, match="hash mismatch"):
            await backend.embed(["complete original source"])
        backend.cache.close()

    asyncio.run(scenario())


def test_embedding_window_fails_before_any_api_call(tmp_path):
    async def scenario():
        backend = LiteBackend({"backend_options": {}, "cache_root": str(tmp_path)})
        with pytest.raises(ValueError, match="32768"):
            await backend.embed(["word " * 33000])
        await backend.aclose()

    asyncio.run(scenario())
