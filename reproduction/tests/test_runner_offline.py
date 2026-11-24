import asyncio
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from reproduction import run
from reproduction.experiments.lite.backend import LiteBackend
from reproduction.experiments.lite.jsonl_io import read_jsonl
from reproduction.experiments.main.paper_backend import PaperSAGMem
from reproduction.experiments.shared import runtime
from src.harness import benchmark
from src.harness.reader_grader import GraderCall, ReaderCall


@pytest.mark.parametrize("method", ["sag-mem", "sag-mem-lite"])
@pytest.mark.parametrize("dataset", ["locomo", "longmemeval"])
@pytest.mark.parametrize("message_cap", [None, 0, 3])
def test_complete_runner_and_cached_rerun_without_network(tmp_path, monkeypatch, method, dataset, message_cap):
    data = tmp_path / "input.json"
    sample = {"sample_id": "fixture", "conversation": {
        "session_1_date_time": "2026-11-05", "session_1": [
            {"dia_id": "m1", "speaker": "Alice", "text": "I moved to Paris."},
            {"dia_id": "m2", "speaker": "Bob", "text": "Paris is in France."}]},
        "qa": [{"question": "Where did Alice move?", "answer": "Paris", "category": 4,
                "evidence": ["m1"]}]}
    if dataset == "longmemeval":
        sample = {"question_id": "fixture", "question_type": "single-session-user",
                  "question": "Where did Alice move?", "answer": "Paris", "question_date": "2026-11-06",
                  "answer_session_ids": ["session-1"], "haystack_session_ids": ["session-1"],
                  "haystack_dates": ["2026-11-05"], "haystack_sessions": [[
                      {"role": "user", "content": "I moved to Paris."},
                      {"role": "assistant", "content": "Paris is in France."}]]}
    for i in range(3, 7):
        if dataset == "locomo":
            sample["conversation"]["session_1"].append(
                {"dia_id": f"m{i}", "speaker": "Alice", "text": f"Paris detail {i}."})
        else:
            sample["haystack_sessions"][0].append({"role": "user", "content": f"Paris detail {i}."})
    data.write_text(json.dumps([sample]))
    args = run.parse_cli(["--method", method, "--dataset", dataset, "--data-path", str(data),
        *(["--messages-per-chunk", str(message_cap)] if message_cap is not None else []),
        "--output", str(tmp_path / "first.jsonl"), "--index-root", str(tmp_path / "indexes")])
    options = run.configure(args)
    options["retained_sources_root"] = str(tmp_path / "retained")
    assert run.preflight(options)["model_calls"] == 0
    embedding_calls = []
    scoring_calls = []

    async def embed(self, texts):
        embedding_calls.extend(texts)
        return np.array([[1.0] + [0.0] * 4095] * len(texts), dtype=np.float32), {
            "requests": 0, "texts": len(texts), "prompt_tokens": 0, "cost_usd": 0}

    class Scorer:
        async def score(self, query, candidates, evidence):
            scoring_calls.append(query)
            return ([{**item, "jev_score": 0.9} for item in candidates],
                    {"requests": 0, "judgments": len(candidates), "input_tokens": 0,
                     "output_tokens": 0, "batches": []})

    original_prepare = PaperSAGMem.prepare

    async def prepare(self, history, spec):
        self.options["retained_sources_root"] = str(tmp_path / "retained")
        await original_prepare(self, history, spec)
        self.client = Scorer()

    class Reader:
        def __init__(self, *args):
            self.reader_model = "qwen/qwen3.6-35b-a3b"

        async def respond(self, context, question, **kwargs):
            assert "I moved to Paris." in context
            return ReaderCall("Paris", 0.001, 20, 1)

        async def grade(self, *args, **kwargs):
            return GraderCall(True, 20, 1)

        async def aclose(self):
            pass

    monkeypatch.setattr(LiteBackend, "embed", embed)
    monkeypatch.setattr(PaperSAGMem, "prepare", prepare)
    monkeypatch.setattr(runtime, "PaperReader", Reader)
    monkeypatch.setattr(benchmark, "budget_context", lambda text, budget: (text, {"context_truncated": False}))
    prepared = benchmark.prepare_run(options)
    assert asyncio.run(run.execute(args, options, prepared)) == 0
    build = benchmark.make_build(prepared[0][0], options, prepared[1])
    sources = json.loads((build.directory / "sources.json").read_text())
    assert len(sources) == (1 if message_cap == 0 else 2)
    assert sum(len(item["sources"]) for item in sources.values()) == 6
    from reproduction.experiments.ablations import prepare as controls
    from reproduction.experiments.ablations.verify import verify as verify_controls
    monkeypatch.setitem(controls.EXPECTED["all"], dataset, 1)
    controls_dir = tmp_path / "controls"
    report = controls.prepare(SimpleNamespace(
        source=[args.output], output=controls_dir, method=method, dataset=dataset,
        scope="all", limit=None, no_expansion=method == "sag-mem", only_no_expansion=False,
        sources_root=tmp_path / "retained", data_root=tmp_path))
    assert report["questions"] == 1
    assert verify_controls([controls_dir])["offline_validation"] == "passed"
    count = len(embedding_calls)
    scored = len(scoring_calls)
    assert (count > 0) == (method == "sag-mem-lite")
    assert (scored > 0) == (method == "sag-mem")
    args.output = tmp_path / "second.jsonl"
    options["output"] = str(args.output)
    assert asyncio.run(run.execute(args, options, benchmark.prepare_run(options))) == 0
    assert len(embedding_calls) == count
    assert len(scoring_calls) == scored
    row = read_jsonl(args.output)[0]
    assert row["status"] == "succeeded" and row["grade"] is True
    assert row["retrieval_cache_reused"] is True
    assert row["context"] == row["reader_context"]
    replay = importlib.import_module("reproduction.experiments.lite.replay")
    monkeypatch.setattr(replay, "PaperReader", Reader)
    target = tmp_path / "deepseek.jsonl"
    assert asyncio.run(replay.replay(args.output, target, index_root=args.index_root,
                                     cache_root=args.cache_root)) == 0
    replayed = read_jsonl(target)[0]
    assert replayed["evidence"] == row["evidence"]
    assert replayed["retrieval_duration"] == 0
    from src.config import settings
    identity = json.loads(target.with_suffix(".jsonl.manifest.json").read_text())["identity"]
    assert identity["config"]["reader_model"] == settings.search_llm.model
    assert identity["reader_service"]["model"] == settings.search_llm.model
    assert identity["judge_service"]["extra_body"] == settings.judge_llm.extra_body
    assert len(embedding_calls) == count
    assert len(scoring_calls) == scored
    from reproduction.analysis.aggregate import verify_manifest
    verify_manifest(identity_manifest := json.loads(target.with_suffix(".jsonl.manifest.json").read_text()),
                    target, "SAG-Mem" if method == "sag-mem" else "SAG-Mem-Lite", "qwen")
    assert identity_manifest["identity"]["reproduction"]["runtime"] == "src"

    if method == "sag-mem-lite":
        from reproduction.experiments.latency import run as latency
        from reproduction.experiments.lite.canonical_summary import canonical_summary
        summary = canonical_summary(args.output)
        assert summary["questions"] == 1
        assert summary["embedding_documents"] > 0
        output = tmp_path / "latency"
        output.mkdir()
        data.rename(tmp_path / ("locomo10.json" if dataset == "locomo" else "longmemeval_s_cleaned.json"))
        monkeypatch.setattr(latency, "DATA_ROOT", tmp_path)
        monkeypatch.setattr(latency, "OUTPUT_ROOT", output)
        monkeypatch.setattr(latency, "MAIN_INDEX_ROOT", tmp_path / "latency-main-indexes")
        monkeypatch.setattr(latency, "LITE_INDEX_ROOT", tmp_path / "latency-lite-indexes")
        monkeypatch.setattr(latency, "LITE_CACHE_ROOT", tmp_path / "latency-cache")
        monkeypatch.setattr(latency, "PaperReader", Reader)
        monkeypatch.setattr(latency, "select", lambda examples, _: examples[:1])
        asyncio.run(latency.main(SimpleNamespace(dataset=dataset, model_family="qwen",
                     local_env_file=None, prepare_missing_lite_index=True)))
        trials = read_jsonl(output / "trials.jsonl")
        assert {trial["method"] for trial in trials} == {"SAG-Mem", "SAG-Mem-Lite"}
        assert all(not trial["judge_called"] and not trial["application_retrieval_cache"] for trial in trials)
