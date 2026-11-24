from reproduction.experiments.shared.chunk_width import chunk_history as width_chunk
from reproduction.experiments.shared.chunking import chunk_history as published_chunk
from src.harness.benchmark import History, Message, Session


def _history():
    short = "hello"
    long = "token " * 400
    return History("c", (Session("s", (
        Message("a", "User", short),
        Message("b", "Assistant", short),
        Message("c", "User", short),
        Message("d", "Assistant", long),
        Message("e", "User", short),
    ), date_raw="2023/01/10"),))


def test_soft_budget_has_no_three_message_cap():
    history = History("c", (Session("s", tuple(
        Message(str(i), "User", "hello") for i in range(6)
    ), date_raw="2023/01/10"),))
    chunks = published_chunk(history)
    assert len(chunks) == 1
    assert len(chunks[0]["sources"]) == 6


def test_oversized_message_stays_whole():
    long = " ".join(f"word{i}" for i in range(800))
    history = History("c", (Session("s", (
        Message("a", "User", "hello"),
        Message("b", "Assistant", long),
        Message("c", "User", "hello"),
    ), date_raw="2023/01/10"),))
    chunks = published_chunk(history)
    oversized = [chunk for chunk in chunks if chunk["token_count"] > 512]
    assert len(oversized) == 1
    assert len(oversized[0]["sources"]) == 1
    assert chunks[0]["sources"][0]["message_id"] == "a"
    assert chunks[-1]["sources"][0]["message_id"] == "c"
    assert all(chunk["token_count"] <= 512 or len(chunk["sources"]) == 1 for chunk in chunks)


def test_width_one_keeps_every_message_separate():
    history = _history()
    chunks = width_chunk(history, 1)
    assert len(chunks) == 5
    assert all(len(chunk["sources"]) == 1 for chunk in chunks)
