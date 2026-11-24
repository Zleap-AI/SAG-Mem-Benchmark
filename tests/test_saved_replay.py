import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_benchmark import parse_args
from src import config
from src.harness import benchmark as b
from src.harness import reader_grader


def legacy_rows():
    return [
        {"question_id": "q_abs", "condition": "sag", "question_type": "single-session-user",
         "question": "(date: 2023/05/30 (Tue) 14:24) How many plants?",
         "gold_answer": "Only five tomatoes were mentioned, not chili peppers.",
         "is_abstention": True, "answer_session_ids": ["s1"],
         "context": "second chunk\nfirst chunk", "hypothesis": "old answer", "grade": True},
        {"question_id": "q_missing", "condition": "sag", "question_type": "single-session-user",
         "question": "Missing evidence?", "gold_answer": "unknown", "grade": False,
         "hypothesis": "", "error": "Old retrieval failed"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["regrade", "replay"])
async def test_legacy_evaluation_preserves_inputs_and_compares_same_subset(tmp_path, monkeypatch, mode):
    source = tmp_path / "legacy.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in legacy_rows()))
    original = source.read_bytes()
    calls = []

    class Reader:
        def __init__(self, *args):
            pass

        async def respond(self, context, question, **kwargs):
            assert mode == "replay"
            assert context == legacy_rows()[0]["context"]
            assert question == legacy_rows()[0]["question"]
            calls.append("reader")
            return SimpleNamespace(hypothesis="new answer", duration=0, prompt_tokens=2, completion_tokens=1)

        async def grade(self, question, gold, answer, category, **kwargs):
            assert question == legacy_rows()[0]["question"]
            assert gold in kwargs["prompt_override"]
            assert answer == ("new answer" if mode == "replay" else "old answer")
            calls.append("judge")
            return SimpleNamespace(is_correct=False, prompt_tokens=2, completion_tokens=1)

        async def aclose(self):
            pass

    def no_backend(*args, **kwargs):
        raise AssertionError("Saved-result evaluation must not open any search backend")

    monkeypatch.setattr(b, "BackendProcess", no_backend)
    monkeypatch.setattr(reader_grader, "ReaderGrader", Reader)
    output = tmp_path / "result.jsonl"
    opts = parse_args([f"--{mode}-from", str(source), "--output", str(output)])
    summary = await b.regrade(opts)
    assert calls == (["reader", "judge"] if mode == "replay" else ["judge"])
    assert source.read_bytes() == original
    assert summary["total"] == 2 and summary["valid_grades"] == 1 and summary["errors"] == 1
    assert summary["source_comparison"] == {
        "paired_records": 1, "source_correct": 1, "current_correct": 0, "lost": 1, "gained": 0,
    }
    rows = b.read_records(output)
    assert rows[0]["source_grade"] is True
    assert rows[1]["grade"] is None and rows[1]["status"] == "failed"
    manifest = json.loads(output.with_suffix(".jsonl.manifest.json").read_text())
    assert manifest["source_records"] == 2
    assert manifest["identity"]["source_sha256"] == b.file_digest(source)
    assert manifest["source_format"] == "historical_longmemeval"
    if mode == "replay":
        assert rows[0]["source_hypothesis"] == "old answer"
        assert manifest["identity"]["evaluation_mode"] == "reader_judge_replay"
    assert "extract_service" not in manifest["identity"]

    # Saved evidence never uses the current extractor. Changing every extractor
    # field must allow resume and must not repeat successful model calls.
    monkeypatch.setattr(config, "settings", replace(
        config.settings, llm_model="another-extractor", llm_base_url="https://extract.test/v1",
        llm_api_key="another-key", llm_extra_body={"another_option": True},
    ))
    resumed = await b.regrade({**opts, "resume": True})
    assert resumed["valid_grades"] == 1
    assert calls == (["reader", "judge"] if mode == "replay" else ["judge"])
    assert json.loads(output.with_suffix(".jsonl.manifest.json").read_text())["run_id"] == manifest["run_id"]
    with pytest.raises(ValueError, match="identity mismatch"):
        await b.regrade({**opts, "resume": True, "judge_model": "another-judge",
                         "explicit": [*opts["explicit"], "judge_model"]})
    with pytest.raises(ValueError, match="identity mismatch"):
        await b.regrade(parse_args([f"--{'regrade' if mode == 'replay' else 'replay'}-from", str(source),
                                   "--output", str(output), "--resume"]))


@pytest.mark.asyncio
async def test_manifest_replay_uses_exact_saved_reader_context(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    row = {"question_id": "q", "run_id": "original", "query": {"text": "question"},
           "target": {"answer": "gold", "category": "multi-session"}, "question_type": "multi-session",
           "context": "untruncated retrieval", "reader_context": "actual reader input", "hypothesis": "old"}
    source.write_text(json.dumps(row) + "\n")
    opts = parse_args(["--replay-from", str(source), "--output", str(tmp_path / "result.jsonl")])
    source.with_suffix(".jsonl.manifest.json").write_text(json.dumps({
        "run_id": "original", "identity": {"config": {
            k: opts[k] for k in ("protocol", "reader_model", "judge_model", "plugins", "model_timeout",
                                "question_date_text", "context_tokens")}}, "question_order": ["q"],
    }))

    class Reader:
        def __init__(self, *args):
            pass

        async def respond(self, context, question, **kwargs):
            assert context == "actual reader input"
            return SimpleNamespace(hypothesis="new", duration=0, prompt_tokens=1, completion_tokens=1)

        async def grade(self, *args, **kwargs):
            return SimpleNamespace(is_correct=True, prompt_tokens=1, completion_tokens=1)

        async def aclose(self):
            pass

    monkeypatch.setattr(reader_grader, "ReaderGrader", Reader)
    summary = await b.regrade(opts)
    assert summary["valid_grades"] == 1
    assert b.read_records(Path(opts["output"]))[0]["source_run_id"] == "original"


@pytest.mark.asyncio
async def test_replay_parallelizes_reader_calls_and_serializes_result_submission(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "result.jsonl"
    opts = parse_args(["--replay-from", str(source), "--output", str(output),
                       "--concurrency", "3"])
    rows = [{"question_id": f"q{i}", "run_id": "original", "question_type": "single-session-user",
             "query": {"text": f"question {i}"},
             "target": {"answer": "gold", "category": "single-session-user"},
             "reader_context": f"context {i}", "hypothesis": "old"} for i in range(4)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    source.with_suffix(".jsonl.manifest.json").write_text(json.dumps({
        "run_id": "original", "identity": {"config": {
            k: opts[k] for k in ("protocol", "reader_model", "judge_model", "plugins", "model_timeout",
                                "question_date_text", "context_tokens")}},
        "question_order": [row["question_id"] for row in rows],
    }))
    active = 0
    peak = 0
    submitted = []

    class Reader:
        def __init__(self, *args):
            pass

        async def respond(self, context, question, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return SimpleNamespace(hypothesis="gold", duration=0, prompt_tokens=1, completion_tokens=1)

        async def grade(self, *args, **kwargs):
            return SimpleNamespace(is_correct=True, prompt_tokens=1, completion_tokens=1)

        async def aclose(self):
            pass

    class Tracking:
        def bind(self, manifest):
            pass

        async def record(self, record, *, extra_metrics=None):
            await asyncio.sleep(0)
            # The next result must not be committed while this one is being reported.
            assert b.read_records(output)[-1]["question_id"] == record["question_id"]
            submitted.append(record["question_id"])

    monkeypatch.setattr(reader_grader, "ReaderGrader", Reader)
    summary = await b.regrade(opts, tracking=Tracking())
    assert peak == 3
    assert summary["valid_grades"] == 4
    assert {row["question_id"] for row in b.read_records(output)} == {"q0", "q1", "q2", "q3"}
    assert submitted == [row["question_id"] for row in b.read_records(output)]


@pytest.mark.asyncio
async def test_replay_cancels_inflight_calls_before_closing_resources_on_write_failure(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    rows = [{**legacy_rows()[0], "question_id": qid, "context": qid} for qid in ("fail", "wait")]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    waiting = asyncio.Event()
    events = []

    class Reader:
        def __init__(self, *args):
            pass

        async def respond(self, context, question, **kwargs):
            if context == "wait":
                waiting.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    events.append("inflight_stopped")
            await waiting.wait()
            return SimpleNamespace(hypothesis="new", duration=0, prompt_tokens=1, completion_tokens=1)

        async def grade(self, *args, **kwargs):
            return SimpleNamespace(is_correct=True, prompt_tokens=1, completion_tokens=1)

        async def aclose(self):
            events.append("reader_closed")

    class Store(b.ResultStore):
        def append(self, record):
            events.append("write_failed")
            raise OSError("Result write failed")

        def close(self):
            events.append("store_closed")
            super().close()

    monkeypatch.setattr(reader_grader, "ReaderGrader", Reader)
    monkeypatch.setattr(b, "ResultStore", Store)
    opts = parse_args(["--replay-from", str(source), "--output", str(tmp_path / "result.jsonl"),
                       "--concurrency", "2"])
    with pytest.raises(ExceptionGroup) as failure:
        await asyncio.wait_for(b.regrade(opts), timeout=5)
    assert len(failure.value.exceptions) == 1
    assert isinstance(failure.value.exceptions[0], OSError)
    assert str(failure.value.exceptions[0]) == "Result write failed"
    assert events == ["write_failed", "inflight_stopped", "store_closed", "reader_closed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [1, 2])
async def test_locomo_retrieval_only_replay_reports_scores_and_resumes(tmp_path, monkeypatch, concurrency):
    source = tmp_path / "retrieval.jsonl"
    opts = parse_args(["--replay-from", str(source), "--output", str(tmp_path / "replay.jsonl"),
                       "--concurrency", str(concurrency)])
    rows = [{"question_id": f"q{category}", "run_id": "retrieval", "question_type": category,
             "query": {"text": f"question {category}"},
             "target": {"answer": "blue", "category": category, "abstention": category == "5"},
             "context": "blue", "reader_context": "blue", "retrieval_only": True,
             "grade": None, "status": "succeeded"} for category in ("4", "5")]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    saved_config = {k: opts[k] for k in ("reader_model", "judge_model", "plugins", "model_timeout",
                                       "question_date_text", "context_tokens")}
    saved_manifest = {"run_id": "retrieval", "identity": {"config": {
        **saved_config, "protocol": "locomo", "retrieval_only": True}},
        "question_order": [r["question_id"] for r in rows]}
    source_manifest_path = source.with_suffix(".jsonl.manifest.json")
    source_manifest_path.write_text(json.dumps(saved_manifest))
    source_bytes, manifest_bytes = source.read_bytes(), source_manifest_path.read_bytes()
    calls = []
    reader_calls = []
    live_metrics = []

    class Reader:
        def __init__(self, *args):
            pass

        async def respond(self, context, question, **kwargs):
            reader_calls.append(question)
            assert context == "blue"
            answer = "No information available" if question.endswith("5") else "blue"
            return SimpleNamespace(hypothesis=answer, duration=0, prompt_tokens=1, completion_tokens=1)

        async def grade(self, question, gold, answer, category, **kwargs):
            calls.append(category)
            if category == "5" and calls.count("5") == 1:
                raise RuntimeError("Transient judge failure")
            return SimpleNamespace(is_correct=True, prompt_tokens=1, completion_tokens=1)

        async def aclose(self):
            pass

    class Tracking:
        def bind(self, manifest):
            self.manifest = manifest

        async def record(self, record, *, extra_metrics=None):
            live_metrics.append(extra_metrics)

    monkeypatch.setattr(reader_grader, "ReaderGrader", Reader)
    monkeypatch.setattr(b, "BackendProcess", lambda *args: pytest.fail("Replay must not open a backend"))
    tracking = Tracking()
    partial = await b.regrade(opts, tracking=tracking)
    report = partial["locomo_evaluation"]
    assert partial["valid_grades"] == report["overall"]["graded"] == 1
    assert report["overall"]["errors"] == 1 and not report["evaluation_complete"]
    assert report["bleu1"] == 1 and report["bleu1_valid_count"] == 1
    assert live_metrics[-1]["judge_accuracy"] == live_metrics[-1]["mem0_judge_accuracy"] == 1
    assert tracking.manifest["identity"]["config"]["retrieval_only"] is False

    result = await b.regrade({**opts, "resume": True}, tracking=tracking)
    report = result["locomo_evaluation"]
    assert calls == ["4", "5", "5"]  # Resume skips the successful factual answer.
    assert result["valid_grades"] == report["overall"]["graded"] == 2
    assert result["accuracy"] == report["overall"]["judge_accuracy"] == 1
    assert report["evaluation_complete"] and report["overall"]["errors"] == 0
    assert report["mem0"]["graded"] == report["by_category"]["adversarial"]["graded"] == 1
    assert live_metrics[-1]["judge_accuracy_adversarial"] == 1
    assert all(r["retrieval_only"] is False for r in b.read_records(Path(opts["output"])))

    regraded = await b.regrade(parse_args([
        "--regrade-from", opts["output"], "--output", str(tmp_path / "regraded.jsonl")]))
    assert regraded["locomo_evaluation"]["overall"] == report["overall"]
    assert len(reader_calls) == 3  # Judge-only regrading preserves the replayed answers.
    assert source.read_bytes() == source_bytes
    assert source_manifest_path.read_bytes() == manifest_bytes


@pytest.mark.parametrize("extra", [["--retrieval-only"], ["--strategy", "vector"],
                                 ["--rerun-incorrect"], ["--regrade-from", "other"]])
def test_replay_rejects_conflicting_options(tmp_path, extra):
    with pytest.raises(SystemExit):
        parse_args(["--replay-from", "source", "--output", str(tmp_path / "out"), *extra])


def test_missing_native_manifest_is_not_silently_treated_as_legacy(tmp_path):
    source = tmp_path / "native.jsonl"
    source.write_text(json.dumps({**legacy_rows()[0], "target": {"answer": "x"}}) + "\n")
    opts = parse_args(["--regrade-from", str(source), "--output", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="Source manifest missing"):
        b.saved_evaluation_input(source, opts)


@pytest.mark.asyncio
async def test_replay_honors_explicit_concurrency_and_preserves_saved_evidence(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    originals = [{**legacy_rows()[0], "question_id": f"q{i}", "context": f"chunk {i}\nfirst chunk"}
                 for i in range(25)]
    source.write_text("".join(json.dumps(row) + "\n" for row in originals))
    source_bytes = source.read_bytes()
    active = peak = 0
    calls = []

    class Reader:
        def __init__(self, *args):
            pass

        async def respond(self, context, question, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            calls.append(context)
            await asyncio.sleep(0.01)
            active -= 1
            return SimpleNamespace(hypothesis="new answer", duration=0, prompt_tokens=1, completion_tokens=1)

        async def grade(self, *args, **kwargs):
            return SimpleNamespace(is_correct=True, prompt_tokens=1, completion_tokens=1)

        async def aclose(self):
            assert active == 0

    monkeypatch.setattr(reader_grader, "ReaderGrader", Reader)
    monkeypatch.setattr(b, "BackendProcess", lambda *args: pytest.fail("Replay must not open a backend"))
    opts = parse_args(["--replay-from", str(source), "--output", str(tmp_path / "replay.jsonl"),
                       "--concurrency", "20"])
    summary = await b.regrade(opts)
    assert peak == 20
    assert summary["valid_grades"] == 25 and summary["errors"] == 0
    records = b.latest_records(b.read_records(Path(opts["output"])))
    assert set(records) == {row["question_id"] for row in originals}
    for original in originals:
        assert records[original["question_id"]]["reader_context"] == original["context"]
    await b.regrade({**opts, "resume": True})
    assert len(calls) == 25
    assert source.read_bytes() == source_bytes
