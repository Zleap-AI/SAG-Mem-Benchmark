"""Exercise Reader/Judge routing through real SDK clients and a local mock transport."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from scripts.run_benchmark import parse_args
from src import config
from src.harness import benchmark, reader_grader
from src.llm_client import ConfiguredAsyncOpenAI


def test_sag_event_chunk_context_pairs_each_event_with_its_source() -> None:
    hits = (
        SimpleNamespace(memory_id="event-1", chunk_id="chunk-1", content="Ranked event one."),
        SimpleNamespace(memory_id="event-2", chunk_id="chunk-2", content="Ranked event two."),
    )

    context = reader_grader.format_sag_event_chunk_evidence(
        hits,
        {"chunk-1": "Source text one.", "chunk-2": "Source text two."},
    )

    assert context.index("[event]\nRanked event one.") < context.index("[source chunk]\nSource text one.")
    assert context.index("[source chunk]\nSource text one.") < context.index("[event]\nRanked event two.")
    assert "[source chunk]\nSource text two." in context


@pytest.mark.asyncio
@pytest.mark.parametrize("structured_judge", [True, False])
@pytest.mark.parametrize("overrides, endpoints", [
    ({}, [("shared.test", "shared-key"), ("shared.test", "shared-key")]),
    ({"openai_api_key": "", "openai_base_url": None,
      "reader_base_url": "https://reader.test/v1", "reader_api_key": "reader-key",
      "judge_base_url": "https://judge.test/v1", "judge_api_key": "judge-key"},
     [("reader.test", "reader-key"), ("judge.test", "judge-key")]),
    ({"reader_base_url": "https://reader.test/v1", "judge_api_key": "judge-key"},
     [("reader.test", "shared-key"), ("shared.test", "judge-key")]),
])
async def test_reader_and_judge_route_models_and_credentials(monkeypatch, overrides, endpoints, structured_judge):
    settings = replace(config.settings, openai_api_key="shared-key", openai_base_url="https://shared.test/v1",
                       reader_base_url=None, reader_api_key="", judge_base_url=None, judge_api_key="",
                       openai_extra_body={"vendor_option": {"enabled": True}},
                       reader_extra_body=None, judge_extra_body=None)
    monkeypatch.setattr(reader_grader, "settings", replace(settings, **overrides))
    requests = []
    clients = []

    def handler(request):
        body = json.loads(request.content)
        requests.append((request.url.host, request.headers["authorization"], body["model"]))
        assert request.url.path == "/v1/chat/completions"
        assert body["vendor_option"] == {"enabled": True}
        if "response_format" in body and not structured_judge:
            return httpx.Response(400, json={"error": {"message": "Structured output unsupported"}})
        content = "blue" if body["model"] == "reader-model" else (
            '{"is_correct":"yes"}' if "response_format" in body else "yes"
        )
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
        })

    def client_factory(**kwargs):
        client = ConfiguredAsyncOpenAI(
            **kwargs, max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(reader_grader, "AsyncOpenAI", client_factory)
    evaluator = reader_grader.ReaderGrader("reader-model", "judge-model")
    try:
        answer = await evaluator.respond("My color is blue.", "What color?")
        grade = await evaluator.grade("What color?", "blue", answer.hypothesis, "single-session-user")
    finally:
        await evaluator.aclose()

    expected = [(host, f"Bearer {key}", model)
                for (host, key), model in zip(endpoints, ("reader-model", "judge-model"), strict=True)]
    if not structured_judge:
        expected.append(expected[-1])
    assert requests == expected
    assert answer.hypothesis == "blue" and grade.is_correct
    assert len(clients) == 2 and all(client.is_closed() for client in clients)


@pytest.mark.asyncio
async def test_saved_answer_regrade_requires_only_judge_credentials(tmp_path, monkeypatch):
    settings = replace(config.settings, openai_api_key="", openai_base_url=None,
                       reader_api_key="", reader_base_url=None, judge_api_key="judge-only-key",
                       judge_base_url="https://judge.test/v1", openai_extra_body={})
    monkeypatch.setattr(config, "settings", settings)
    monkeypatch.setattr(reader_grader, "settings", settings)
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({
        "question_id": "q1", "condition": "sag", "question_type": "single-session-user",
        "question": "What color?", "gold_answer": "blue", "hypothesis": "blue", "grade": True,
    }) + "\n")
    requests = []
    clients = []

    def handler(request):
        requests.append(request)
        assert request.url.host == "judge.test"
        assert request.headers["authorization"] == "Bearer judge-only-key"
        assert json.loads(request.content)["model"] == "judge-model"
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0, "model": "judge-model",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": '{"is_correct":"yes"}'}}],
        })

    def client_factory(**kwargs):
        assert kwargs["api_key"] == "judge-only-key"
        client = ConfiguredAsyncOpenAI(
            **kwargs, max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(reader_grader, "AsyncOpenAI", client_factory)
    output = tmp_path / "regraded.jsonl"
    options = parse_args(["--regrade-from", str(source), "--output", str(output), "--judge-model", "judge-model"])
    summary = await benchmark.regrade(options)

    assert summary["accuracy"] == 1 and summary["errors"] == 0
    assert len(requests) == len(clients) == 1 and clients[0].is_closed()
    manifest = json.loads(output.with_suffix(".jsonl.manifest.json").read_text())
    assert manifest["identity"]["reader_service"]["base_url"] is None
    assert manifest["identity"]["judge_service"]["base_url"] == "https://judge.test/v1"
    assert manifest["identity"]["judge_service"]["model"] == "judge-model"
    assert "judge-only-key" not in json.dumps(manifest)


@pytest.mark.asyncio
async def test_close_releases_both_clients_even_if_reader_close_fails():
    evaluator = reader_grader.ReaderGrader()
    reader_close = AsyncMock(side_effect=RuntimeError("close failed"))
    judge_close = AsyncMock()
    evaluator._reader_client = SimpleNamespace(close=reader_close)
    evaluator._judge_client = SimpleNamespace(close=judge_close)

    await evaluator.aclose()
    await evaluator.aclose()

    reader_close.assert_awaited_once()
    judge_close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides, expected_reader, expected_judge", [
    ({"reader_extra_body": {"reader_only": True}, "judge_extra_body": {"judge_only": True}},
     {"reader_only": True}, {"judge_only": True}),
    ({}, {"shared": True}, {"shared": True}),
    ({"judge_extra_body": {}}, {"shared": True}, {}),
    ({"reader_extra_body": {"reader_only": True}}, {"reader_only": True}, {"shared": True}),
])
async def test_reader_and_judge_request_bodies_are_independent(monkeypatch, overrides,
                                                              expected_reader, expected_judge):
    """Each eval role carries its own request body, falling back to the shared one."""
    settings = replace(config.settings, openai_api_key="shared", openai_base_url="https://shared.test/v1",
                       openai_extra_body={"shared": True}, reader_extra_body=None, judge_extra_body=None)
    monkeypatch.setattr(reader_grader, "settings", replace(settings, **overrides))
    evaluator = reader_grader.ReaderGrader()
    try:
        assert evaluator._reader_client._request_extra_body == expected_reader
        assert evaluator._judge_client._request_extra_body == expected_judge
    finally:
        await evaluator.aclose()


@pytest.mark.asyncio
async def test_role_models_apply_without_cli_overrides(monkeypatch):
    settings = replace(config.settings, openai_api_key="shared", openai_base_url="https://shared.test/v1",
                       response_model="from-search-role", grader_model="from-judge-role")
    monkeypatch.setattr(reader_grader, "settings", settings)
    evaluator = reader_grader.ReaderGrader()
    try:
        assert evaluator.reader_model == "from-search-role"
        assert evaluator.judge_model == "from-judge-role"
    finally:
        await evaluator.aclose()
