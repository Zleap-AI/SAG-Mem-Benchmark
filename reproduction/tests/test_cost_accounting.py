import json
from pathlib import Path

import pytest

from reproduction.experiments.lite.canonical_summary import canonical_summary
from reproduction.experiments.lite.summarize import summarize


def saved_run(tmp_path):
    index = tmp_path / "indexes/lite/build"
    index.mkdir(parents=True)
    (index / "sag2.sqlite").write_bytes(b"fixture")
    (index / "sources.json").write_text(json.dumps({"1": {"content": "A short passage."}}))
    usage = {"prompt_tokens": 100, "completion_tokens": 10,
             "prompt_tokens_details": {"cached_tokens": 80}}
    model = "deepseek/deepseek-v4.1-flash-1017"
    calls = [{"question_id": "q", "stage": role, "status": "succeeded", "model": model,
              "method": role + ".create", "usage": usage, "cost_usd": 0.00001848,
              "elapsed_seconds": 1} for role in ["reader", "judge"]]
    manifest = {"run_id": "fixture", "identity": {
        "questions": [{"id": "q"}],
        "config": {"namespace": "lite", "index_root": str(tmp_path / "indexes"),
                   "reader_model": model, "judge_model": model},
        "build_dependencies": {"embedding": {"profile": "qwen3", "model": "qwen/qwen3-embedding-8b",
                                                 "max_input_tokens": 32768, "overflow": "error"}},
        "backend_runtime": {"query_instruction": "Retrieve relevant passages"}}}
    path = tmp_path / "run.jsonl"
    path.with_suffix(".jsonl.manifest.json").write_text(json.dumps(manifest))
    row = {"question_id": "q", "question": "Where?", "run_id": "fixture", "source_history_sha256": "h",
           "build_id": "build", "status": "succeeded", "grade": True,
           "retrieval_cache_reused": False, "retrieval_duration": 1, "question_type": "single-session-user",
           "reader_prompt_tokens": 100, "reader_completion_tokens": 10,
           "grader_prompt_tokens": 100, "grader_completion_tokens": 10, "api_calls": calls}
    path.write_text(json.dumps(row) + "\n")
    (tmp_path / "requests.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    return path


def test_operational_cost_excludes_judge_and_prices_cached_input(tmp_path):
    path = saved_run(tmp_path)
    result = canonical_summary(path)
    reader = (100 - 80) * 0.0000003 + 80 * 0.000000006 + 10 * 0.0000012
    assert result["canonical_reader_standardized_usd"] == pytest.approx(reader)
    assert result["canonical_judge_standardized_usd"] == pytest.approx(reader)
    expected = (result["logical_embedding_build_standardized_usd"]
                + result["canonical_query_embedding_standardized_usd"] + reader)
    assert result["canonical_total_standardized_usd"] == pytest.approx(expected)
    assert summarize(path)["standardized_cost_usd"] == pytest.approx(2 * reader)


def test_embedding_accounting_follows_recorded_encoder(tmp_path, monkeypatch):
    path = saved_run(tmp_path)
    manifest_path = path.with_suffix(".jsonl.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["identity"]["build_dependencies"]["embedding"].update(
        profile="bge", model="test-bge", max_input_tokens=512, overflow="prefix")
    manifest_path.write_text(json.dumps(manifest))
    from reproduction.experiments.lite import summarize as module
    prices = json.loads(Path(module.__file__).with_name("prices.json").read_text())
    prices["models"]["test-bge"] = {"input_per_token": 0.0002, "output_per_token": 0}
    price_path = tmp_path / "prices.json"
    price_path.write_text(json.dumps(prices))
    monkeypatch.setattr(module, "PRICE_PATH", price_path, raising=False)
    result = canonical_summary(path)
    assert result["embedding_input_window_tokens"] == 512
    assert result["logical_embedding_build_standardized_usd"] == pytest.approx(
        result["logical_embedding_build_tokens"] * 0.0002)


def test_publication_summary_uses_cache_discount_and_excludes_judge(tmp_path, monkeypatch):
    from reproduction.analysis import aggregate

    path = saved_run(tmp_path)
    saved = json.loads(path.read_text())
    usage = saved["api_calls"][0]["usage"]
    row = {"grade": True, "status": "succeeded", "truncated": False,
           "history": "h", "category": "single-session-user",
           "reader_input_tokens": 100, "reader_output_tokens": 10,
           "grader_input_tokens": 100, "grader_output_tokens": 10,
           "reader_usage": usage, "grader_usage": usage, "time_annotations": 0,
           "jev_input_tokens": 0, "jev_output_tokens": 0, "jev_requests": 0, "jev_judgments": 0}
    monkeypatch.setattr(aggregate, "read_run", lambda *args: ({"q": row}, [], True))
    _, result, _ = aggregate.summarize("SAG-Mem", "deepseek", "longmemeval", [path])
    expected = 20 * 0.0000003 + 80 * 0.000000006 + 10 * 0.0000012
    assert result["reader_cost_usd"] == pytest.approx(expected)
    assert result["offline_grader_cost_usd"] == pytest.approx(expected)
    assert result["total_cost_usd"] == pytest.approx(expected)
