"""Pin historical prompt compatibility independently of the current templates."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.harness import prompts
from src.harness.benchmark import Target
from src.harness.reader_grader import Grade, LongMemEvalProtocol, ReaderGrader

HISTORICAL = json.loads(
    (Path(__file__).parent / "fixtures" / "longmemeval_judge_03aa095.json").read_text()
)


def test_reader_and_judge_constants_match_historical_run():
    for name in ("READER_SYSTEM_PROMPT", "READER_USER_PROMPT", "GRADER_SYSTEM_PROMPT", "GRADING_PROMPTS"):
        assert getattr(prompts, name) == HISTORICAL[name]


@pytest.mark.asyncio
async def test_reader_request_hides_session_labels_without_changing_evidence():
    context = "Evidence:\n[session session_123] I have five plants.\n[session abc-456] Two are peppers."
    reader = ReaderGrader.__new__(ReaderGrader)
    reader.reader_model = "test-model"
    create = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Five plants."))], usage=None,
    ))
    reader._reader_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    await reader.respond(context, "How many plants?")

    assert create.call_args.kwargs["messages"] == [
        {"role": "system", "content": prompts.READER_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.READER_USER_PROMPT.format(
            context="Evidence:\nI have five plants.\nTwo are peppers.", question="How many plants?",
        )},
    ]
    assert "[session session_123]" in context


@pytest.mark.asyncio
@pytest.mark.parametrize("abstention", [False, True])
@pytest.mark.parametrize("category", [
    "single-session-user", "single-session-assistant", "single-session-preference",
    "multi-session", "temporal-reasoning", "knowledge-update",
])
async def test_longmemeval_judge_request_matches_historical_run(category, abstention):
    question = "How many tomato and chili pepper plants did I initially plant?"
    gold = "You mentioned five tomato plants, but did not mention chili peppers."
    answer = "You initially planted 5 tomato plants. Chili pepper plants were not mentioned."
    target = Target(gold, category, abstention)
    protocol = LongMemEvalProtocol()
    grader = ReaderGrader.__new__(ReaderGrader)
    grader.judge_model = "test-model"
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=Grade(is_correct="yes")))],
        usage=None,
    )
    parse = AsyncMock(return_value=completion)
    grader._judge_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))

    await grader.grade(question, gold, answer, category,
                       prompt_override=protocol.judge_prompt(question, answer, target))

    template = HISTORICAL["GRADING_PROMPTS"].get(category, HISTORICAL["GRADING_PROMPTS"]["default"])
    assert parse.call_args.kwargs["messages"] == [
        {"role": "system", "content": HISTORICAL["GRADER_SYSTEM_PROMPT"]},
        {"role": "user", "content": template.format(question=question, gold_answer=gold, response=answer)},
    ]
    assert parse.call_args.kwargs["temperature"] == 0
