from copy import deepcopy
from datetime import date

import pytest

from src.backends.sag2_chunk_time import annotate_chunk, format_message_date, parse_source_date


def chunk():
    bodies = ["I left yesterday.\nQuoted (date: 2099-01-01): keep this text.", "Last month was busy."]
    dates = ["1:56 pm on 21 May, 2023", "2023/06/01 (Thu) 08:00"]
    return {"kind": "chunk", "chunk_id": "s0:c0", "content": "\n".join(
        f"{speaker} (date: {raw}): {body}"
        for speaker, raw, body in zip(("Alice", "Bob"), dates, bodies, strict=True)),
        "message_spans": [{"start": 0, "end": len(body)} for body in bodies]}


@pytest.mark.parametrize("raw", ["2023-05-21", "2023/05/21 (Sun) 18:59",
                                "6:59 pm on 21 May, 2023", "2023年5月21日", "2023年5月21日 18:59"])
def test_source_date_formats(raw):
    assert parse_source_date(raw) == date(2023, 5, 21)


@pytest.mark.parametrize(("raw", "expected"), [
    ("2023-05-21", "2023/05/21"),
    ("2023/05/21 (Sun) 18:59", "2023/05/21 18:59"),
    ("1:56 pm on 21 May, 2023", "2023/05/21 13:56"),
    ("12:05 am on 21 May, 2023", "2023/05/21 00:05"),
    ("12:05 pm on 21 May, 2023", "2023/05/21 12:05"),
    ("2023年5月21日 8:09", "2023/05/21 08:09"),
    ("2023-05-21T23:59:12-07:00", "2023/05/21 23:59"),
])
def test_message_date_preserves_available_local_time_without_weekday(raw, expected):
    assert format_message_date(raw) == expected


def test_each_message_uses_its_own_date_and_normalizes_its_header():
    item = chunk()
    original = deepcopy(item)
    item["message_spans"] *= 2
    text, matches = annotate_chunk(item)
    assert [m["value"] for m in matches] == ["2023-05-20", "2023-05"]
    assert "Alice: [Message date: 2023/05/21 13:56]\nI left yesterday（2023/05/20）." in text
    assert "Bob: [Message date: 2023/06/01 08:00]\nLast month（2023/05） was busy." in text
    assert "Quoted (date: 2099-01-01): keep this text." in text
    assert item["content"] == original["content"]


def test_unknown_source_date_is_rejected_before_time_annotation():
    body = "The agent tried this yesterday."
    item = {"kind": "chunk", "chunk_id": "s0:t0", "content": f"Trajectory (date: unknown): {body}",
            "message_spans": [{"start": 0, "end": len(body)}]}
    with pytest.raises(ValueError, match="Cannot parse source date"):
        annotate_chunk(item)
