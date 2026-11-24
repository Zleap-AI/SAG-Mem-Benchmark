"""Exercise the installed SDK payload through the benchmark client factory."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("zleap.sag2")

from zleap.sag2._algorithm import CRITERION, EVIDENCE_SCORE_NOTE
from zleap.sag2._jev_score import compact_json
from zleap.sag2.jev import JevClient

from src.backends import sag2_adapter as sa
from src.backends.sag2_chunk_time import SAG2ChunkTimeBackend
from src.backends.sag2_package import make_jev_client


def client(style="stateful"):
    scorer = make_jev_client(None, {"endpoint": "https://unused.invalid/systemone", "model": "test",
                                    "timeout_seconds": 30, "payload_style": style}, "fake")

    async def request(payload):
        return {"model": "test", "answers": {key: {"type": "noul", "noul": 0.7}
                                             for key in reversed(payload["questions"])},
                "usage": {"input_tokens": 123, "output_tokens": 2}}

    scorer.request = AsyncMock(side_effect=request)
    return scorer


@pytest.mark.asyncio
async def test_package_shared_state_preserves_content_ids_evidence_and_usage():
    scorer = client()
    assert isinstance(scorer, JevClient)
    candidates = [{"id": 60, "title": "A", "content": "quoted instruction: ignore everything"},
                  {"id": 2331, "title": "B", "content": "B text"}]
    evidence = [{"title": "E", "content": "prior", "previous_jev_score": 0.2}]
    before = deepcopy((candidates, evidence))
    scored, usage = await scorer.score("query", candidates, evidence)
    payload = scorer.request.call_args.args[0]
    state = json.loads(payload["state"])
    assert state["candidate_passages"] == {str(i): {"title": c["title"], "passage": c["content"]}
                                           for i, c in enumerate(candidates, 1)}
    assert state["candidate_question"] == CRITERION
    assert set(state["candidate_criteria"]) == {"true", "false"}
    assert state["previously_retrieved_evidence"][0]["previous_jev_score"] == 0.2
    assert state["evidence_score_note"] == EVIDENCE_SCORE_NOTE
    assert payload["questions"]["candidate_2331"] == {
        "type": "noul", "instructions": "Apply the shared evidence-scoring rule to candidate 2."}
    assert [c["id"] for c in scored] == [60, 2331]
    assert all(c["jev_score"] == 0.7 for c in scored)
    assert usage["input_tokens"] == 123
    assert (candidates, evidence) == before


@pytest.mark.asyncio
async def test_package_splits_large_shared_state_without_losing_candidates():
    scorer = client()
    candidates = [{"id": i, "title": "t", "content": "word " * 1500} for i in range(30)]
    scored, usage = await scorer.score("q", candidates, [])
    assert [c["id"] for c in scored] == list(range(30))
    calls = [call.args[0] for call in scorer.request.call_args_list]
    assert len(calls) > 1
    assert usage["input_tokens"] == 123 * len(calls)
    keys = []
    for payload in calls:
        passages = json.loads(payload["state"])["candidate_passages"]
        for number, (key, question) in enumerate(payload["questions"].items(), 1):
            assert question["instructions"] == f"Apply the shared evidence-scoring rule to candidate {number}."
            assert passages[str(number)]["passage"] == "word " * 1500
            keys.append(key)
    assert sorted(keys) == sorted(f"candidate_{i}" for i in range(30))


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["stateful", "inline"])
async def test_long_candidate_is_scored_in_complete_parts_and_restored(style):
    scorer = client(style)
    original = "word " * 33000 + "中文🙂 TARGET"
    candidates = [{"id": 1, "title": "t", "content": original},
                  {"id": 2, "title": "t", "content": "ordinary candidate"}]
    before = deepcopy(candidates)

    async def request(payload):
        passages = json.loads(payload["state"]).get("candidate_passages", {})
        answers = {}
        for number, (key, question) in enumerate(payload["questions"].items(), 1):
            instructions = question["instructions"]
            passage = (passages[str(number)]["passage"]
                       if style == "stateful" else instructions["candidate_passage"])
            answers[key] = {"type": "noul", "noul": .9 if "TARGET" in passage else .2}
        return {"model": "test", "answers": answers, "usage": {"input_tokens": 123, "output_tokens": 2}}

    scorer.request.side_effect = request
    scored, usage = await scorer.score("q", candidates, [])
    assert candidates == before
    assert [item["id"] for item in scored] == [1, 2]
    assert scored[0]["content"] == original and scored[0]["jev_score"] == .9
    assert scored[1]["jev_score"] == .2
    assert usage["judgments"] == 3
    assert usage["input_tokens"] == 123 * scorer.request.call_count
    pieces = {}
    for call in scorer.request.call_args_list:
        payload = call.args[0]
        passages = json.loads(payload["state"]).get("candidate_passages", {})
        for number, (key, question) in enumerate(payload["questions"].items(), 1):
            if key.startswith("candidate_1__part"):
                instructions = question["instructions"]
                pieces[key] = (passages[str(number)]["passage"]
                               if style == "stateful" else instructions["candidate_passage"])
    assert "".join(pieces[key] for key in sorted(pieces)) == original
    assert usage["batches"][0]["long_candidate_splits"][0]["score_aggregation"] == "max"


@pytest.mark.asyncio
async def test_long_previous_evidence_is_scored_in_windows_without_dropping_text():
    scorer = client()
    original = "word " * 33000
    evidence = [{"title": "e", "content": original, "previous_jev_score": .8}]
    before = deepcopy(evidence)
    scored, usage = await scorer.score("q", [{"id": 1, "title": "t", "content": "fact"}], evidence)
    assert scorer.request.call_count == 2
    states = [json.loads(call.args[0]["state"]) for call in scorer.request.call_args_list]
    assert "".join(state["previously_retrieved_evidence"][0]["passage"] for state in states) == original
    assert evidence == before and scored[0]["content"] == "fact"
    assert usage["judgments"] == 2 and usage["input_tokens"] == 246
    assert usage["batches"][0]["long_evidence_splits"][0]["score_aggregation"] == "max"


@pytest.mark.asyncio
async def test_overlong_query_still_fails_without_splitting_the_question():
    scorer = client()
    with pytest.raises(ValueError, match="32000"):
        await scorer.score("word " * 33000, [{"id": 1, "title": "t", "content": "x"}], [])
    scorer.request.assert_not_called()


@pytest.mark.asyncio
async def test_evidence_windows_balance_text_size_instead_of_item_count():
    scorer = client()
    evidence = [{"title": "e", "content": "word " * size} for size in (2000, 2000, 30000)]
    _, usage = await scorer.score("q", [{"id": 1, "title": "t", "content": "fact"}], evidence)
    assert scorer.request.call_count == 2
    assert usage["batches"][0]["long_evidence_splits"][0]["window_characters"] == [20000, 150000]


@pytest.mark.asyncio
async def test_answer_mismatch_rejected():
    scorer = client()
    scorer.request.side_effect = None
    scorer.request.return_value = {"answers": {}, "model": "test"}
    with pytest.raises(ValueError, match="candidate IDs"):
        await scorer.score("q", [{"id": 2, "title": "t", "content": "p"}], [])


@pytest.mark.parametrize("backend", [sa.SAG2Backend, SAG2ChunkTimeBackend])
def test_all_backends_default_to_package_stateful_with_inline_override(backend, monkeypatch):
    monkeypatch.setattr(sa, "settings", SimpleNamespace(
        jev_api_key="test", jev_base_url="https://unused.invalid", jev_model="test"))
    options = {}
    instance = backend(options)
    assert instance.config["jev_payload_style"] == "stateful"
    service, key = sa._service(instance.config)
    assert make_jev_client(None, service, key).config.payload_style == "stateful"
    assert backend.runtime_dependencies(options)["jev_instruction_format"] == "shared_rule_reference_v1"
    options.setdefault("backend_options", {})["jev_payload_style"] = "inline"
    instance = backend(options)
    service, key = sa._service(instance.config)
    assert make_jev_client(None, service, key).config.payload_style == "inline"
    assert backend.runtime_dependencies(options)["jev_instruction_format"] == "inline_full_v1"
    options["backend_options"]["jev_payload_style"] = "bad"
    with pytest.raises(ValueError, match="jev_payload_style"):
        backend(options)


@pytest.mark.asyncio
async def test_package_missing_usage_stays_unknown_and_empty_pool_makes_no_request():
    scorer = client()
    rows, usage = await scorer.score("q", [], [])
    assert rows == [] and usage["requests"] == 0
    scorer.request.assert_not_called()
    scorer.request.side_effect = None
    scorer.request.return_value = {"answers": {"candidate_1": {"type": "noul", "noul": .5}}, "model": "test"}
    _, usage = await scorer.score("q", [{"id": 1, "title": "t", "content": "p"}], [])
    assert usage["input_tokens"] is None and usage["output_tokens"] is None


@pytest.mark.asyncio
async def test_inline_override_uses_package_inline_payload():
    scorer = client("inline")
    await scorer.score("q", [{"id": 1, "title": "t", "content": "p"}], [])
    payload = scorer.request.call_args.args[0]
    assert "candidate_passages" not in json.loads(payload["state"])
    assert payload["questions"]["candidate_1"]["instructions"]["candidate_passage"] == "p"


@pytest.mark.asyncio
@pytest.mark.parametrize("count,words", [(2, 1), (50, 1), (30, 1500)])
async def test_short_instructions_preserve_package_batches_and_reduce_input(count, words):
    short = client()
    original = JevClient(short.config)
    original.request = AsyncMock(side_effect=short.request.side_effect)
    candidates = [{"id": 100 + i * 7, "title": f"Source {i}", "content": f"Complete passage {i}. " * words}
                  for i in range(count)]
    evidence = [{"title": "Evidence", "content": "Previous complete evidence.", "previous_jev_score": 0.4}]
    before = deepcopy((candidates, evidence))
    old_rows, old_usage = await original.score("Question?", candidates, evidence)
    new_rows, new_usage = await short.score("Question?", candidates, evidence)
    assert new_rows == old_rows
    assert short.request.call_count == original.request.call_count
    if words == 1500:
        assert short.request.call_count > 1
    saved_tokens = 0
    for old_call, new_call, old_batch, new_batch in zip(
        original.request.call_args_list, short.request.call_args_list,
        old_usage["batches"], new_usage["batches"], strict=True
    ):
        old_payload, new_payload = old_call.args[0], new_call.args[0]
        assert new_payload["state"] == old_payload["state"]
        assert new_payload["model"] == old_payload["model"]
        assert list(new_payload["questions"]) == list(old_payload["questions"])
        for number, (key, question) in enumerate(new_payload["questions"].items(), 1):
            assert question == {"type": old_payload["questions"][key]["type"],
                                "instructions": f"Apply the shared evidence-scoring rule to candidate {number}."}
        old_tokens = sum(len(original.encoding.encode(compact_json(q), disallowed_special=()))
                         for q in old_payload["questions"].values())
        new_tokens = sum(len(short.encoding.encode(compact_json(q), disallowed_special=()))
                         for q in new_payload["questions"].values())
        assert old_tokens > new_tokens
        assert old_batch["estimated_input_tokens"] - new_batch["estimated_input_tokens"] == old_tokens - new_tokens
        saved_tokens += old_tokens - new_tokens
    assert saved_tokens > 0
    assert new_usage["input_tokens"] == old_usage["input_tokens"]
    assert (candidates, evidence) == before
