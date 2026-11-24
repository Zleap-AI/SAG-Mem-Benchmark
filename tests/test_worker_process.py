"""Worker shutdown contracts retained after index-maintenance removal."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from scripts.run_benchmark import parse_args
from src.harness import benchmark as b


@pytest.mark.asyncio
async def test_worker_close_uses_separate_maintenance_timeout(monkeypatch):
    waited = []
    worker = b.BackendProcess({"worker_close_timeout": 123})
    worker.process = SimpleNamespace(returncode=None, stdin=SimpleNamespace(write=lambda _: None,
                                      drain=AsyncMock()), wait=AsyncMock(), kill=lambda: pytest.fail("killed"))

    async def wait_for(awaitable, timeout):
        waited.append(timeout)
        return await awaitable

    monkeypatch.setattr(asyncio, "wait_for", wait_for)
    await worker.close()
    assert waited == [123]


def test_close_timeout_does_not_change_run_identity(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps([{"question_id": "q", "question": "color?", "answer": "blue",
                                "question_type": "single-session-user", "haystack_session_ids": ["s"],
                                "haystack_dates": ["2023/01/01 10:00"],
                                "haystack_sessions": [[{"role": "user", "content": "blue"}]]}]))
    opts = parse_args(["--backend", "full_context", "--data-path", str(path)])
    assert b.prepare_run(opts)[2]["run_id"] == b.prepare_run({**opts, "worker_close_timeout": 999})[2]["run_id"]
    with pytest.raises(SystemExit):
        parse_args(["--worker-close-timeout", "0"])
