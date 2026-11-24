import json

import pytest

from src.data import loader


def _write_dataset(tmp_path, monkeypatch, items: list[dict]) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    monkeypatch.setattr(loader, "_dataset_path", lambda _name: path)
    # Exercise boundaries inside strings and nested JSON instead of relying on
    # the production 1 MiB read size.
    monkeypatch.setattr(loader, "_READ_CHUNK_SIZE", 17)


def _item(question_id: str, text: str) -> dict:
    return {
        "question_id": question_id,
        "question_type": "single-session-user",
        "question": text,
        "answer": "答案",
        "question_date": "2026/09/03",
        "answer_session_ids": ["session-1"],
        "haystack_sessions": [[{"role": "user", "content": text}]],
        "haystack_dates": ["2026/09/02"],
        "haystack_session_ids": ["session-1"],
    }


def test_streaming_loaders_preserve_items_and_drop_haystack_from_metadata(
    tmp_path, monkeypatch
) -> None:
    items = [_item("q1", "braces { inside } a string"), _item("q2", "第二条消息")]
    _write_dataset(tmp_path, monkeypatch, items)

    assert loader.load_dataset("s") == items
    assert loader.load_instance("s", 1) == items[1]

    metadata = loader.load_dataset_metadata("s")
    assert [item["question_id"] for item in metadata] == ["q1", "q2"]
    assert metadata[0]["answer_session_ids"] == ["session-1"]
    assert "haystack_sessions" not in metadata[0]
    assert "haystack_dates" not in metadata[0]


def test_load_instance_validates_index(tmp_path, monkeypatch) -> None:
    _write_dataset(tmp_path, monkeypatch, [_item("q1", "text")])

    with pytest.raises(IndexError, match="non-negative"):
        loader.load_instance("s", -1)
    with pytest.raises(IndexError, match="out of range"):
        loader.load_instance("s", 1)


def test_metadata_requires_scheduler_fields(tmp_path, monkeypatch) -> None:
    item = _item("q1", "text")
    del item["answer"]
    _write_dataset(tmp_path, monkeypatch, [item])

    with pytest.raises(ValueError, match="missing required field 'answer'"):
        loader.load_dataset_metadata("s")


def _write_raw(tmp_path, monkeypatch, text: str) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(loader, "_dataset_path", lambda _name: path)
    monkeypatch.setattr(loader, "_READ_CHUNK_SIZE", 7)


@pytest.mark.parametrize(
    "raw",
    [
        '[{"a": 1}] GARBAGE',
        '[{"a": 1},{"b": 2}][{"c": 3}]',  # dataset file written twice
        '[{"a": 1}]]',
    ],
)
def test_trailing_content_after_array_is_rejected(tmp_path, monkeypatch, raw) -> None:
    # json.load used to catch these; a silently short run would have its
    # accuracy attributed to the full 500-instance split.
    _write_raw(tmp_path, monkeypatch, raw)
    with pytest.raises(ValueError, match="Trailing content"):
        loader.load_dataset("s")


def test_trailing_comma_is_rejected(tmp_path, monkeypatch) -> None:
    _write_raw(tmp_path, monkeypatch, '[{"a": 1},]')
    with pytest.raises(ValueError, match="Trailing comma"):
        loader.load_dataset("s")


def test_trailing_whitespace_is_accepted(tmp_path, monkeypatch) -> None:
    _write_raw(tmp_path, monkeypatch, '[{"a": 1}]\n\n  \n')
    assert loader.load_dataset("s") == [{"a": 1}]


def test_metadata_load_rejects_truncated_file(tmp_path, monkeypatch) -> None:
    # The scheduler derives its index range from this; a short read must raise.
    _write_raw(tmp_path, monkeypatch, '[{"question_id": "q1"}] extra')
    with pytest.raises(ValueError):
        loader.load_dataset_metadata("s")
