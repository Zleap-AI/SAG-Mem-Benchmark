import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel

from src.llm_client import ConfiguredAsyncOpenAI


class Answer(BaseModel):
    ok: bool


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat_create", "chat_parse", "responses_create", "responses_parse"])
@pytest.mark.parametrize("extra", [None, {
    "enable_thinking": True, "chat_template_kwargs": {"enable_thinking": True}, "custom": 7,
}])
async def test_default_policy_preserves_native_body(api, extra):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(429, json={"error": {"message": "retry"}}, headers={"retry-after-ms": "1"})
        return httpx.Response(200, json=response_for(request))

    async with ConfiguredAsyncOpenAI(
        api_key="test", max_retries=1,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as client:
        if api.startswith("chat"):
            kwargs = {"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}],
                      "extra_body": extra}
            if api.endswith("parse"):
                await client.chat.completions.parse(**kwargs, response_format=Answer)
            else:
                await client.chat.completions.create(**kwargs)
        elif api.endswith("parse"):
            await client.responses.parse(model="gpt-4o", input="test", text_format=Answer, extra_body=extra)
        else:
            await client.responses.create(model="gpt-4o", input="test", extra_body=extra)

    assert len(bodies) == 2
    for body in bodies:
        if extra is None:
            assert "enable_thinking" not in body
            assert "chat_template_kwargs" not in body
        else:
            assert all(body[key] == value for key, value in extra.items())


def response_for(request):
    if request.url.path.endswith("/responses"):
        return {
            "id": "resp_test", "object": "response", "created_at": 0,
            "model": "auto", "status": "completed", "error": None,
            "incomplete_details": None, "parallel_tool_calls": True,
            "tools": [], "tool_choice": "auto",
            "output": [{"type": "message", "id": "msg_test", "role": "assistant",
                        "status": "completed", "content": [
                            {"type": "output_text", "text": '{"ok":true}', "annotations": []}
                        ]}],
        }
    if request.url.path.endswith("/embeddings"):
        return {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                "model": "embedding"}
    if request.method == "GET":
        return {"object": "list", "data": []}
    return {
        "id": "chat_test", "object": "chat.completion", "created": 0, "model": "auto",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": '{"ok":true}'}}],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat_create", "chat_parse", "responses_create", "responses_parse"])
@pytest.mark.parametrize("model", ["auto", "gpt-4o", "qwen-test"])
async def test_configured_json_on_wire_and_sdk_retry(api, model):
    requests = []
    extra = {"enable_thinking": True, "chat_template_kwargs": {"enable_thinking": True, "custom": 7},
             "other_option": "keep"}

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(429, json={"error": {"message": "retry"}}, headers={"retry-after-ms": "1"})
        return httpx.Response(200, json=response_for(request))

    async with ConfiguredAsyncOpenAI(
        request_extra_body={"enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False},
                            "temperature": 0.3, "vendor_option": [1, True, None]},
        api_key="test", base_url="http://test/v1", max_retries=1,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as client:
        if api.startswith("chat"):
            kwargs = {"model": model, "messages": [{"role": "user", "content": "test"}], "extra_body": extra}
            if api.endswith("parse"):
                result = await client.chat.completions.parse(**kwargs, response_format=Answer)
                assert result.choices[0].message.parsed.ok
            else:
                await client.chat.completions.create(**kwargs)
        elif api.endswith("parse"):
            result = await client.responses.parse(model=model, input="test", text_format=Answer, extra_body=extra)
            assert result.output_parsed.ok
        else:
            await client.responses.create(model=model, input="test", extra_body=extra)

    assert len(requests) == 2
    for body in requests:
        assert body["enable_thinking"] is False
        assert body["chat_template_kwargs"] == {"enable_thinking": False, "custom": 7}
        assert body["other_option"] == "keep"
        assert body["temperature"] == 0.3
        assert body["vendor_option"] == [1, True, None]
    assert extra["enable_thinking"] is True
    assert extra["chat_template_kwargs"]["enable_thinking"] is True


@pytest.mark.asyncio
async def test_non_llm_requests_unchanged():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=response_for(request))

    async with ConfiguredAsyncOpenAI(
        request_extra_body={"enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False},
                            "temperature": 0.3, "vendor_option": [1, True, None]},
        api_key="test", base_url="http://test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as client:
        await client.models.list()
        await client.embeddings.create(model="embedding", input="test")
    assert requests[0].content == b""
    assert "enable_thinking" not in json.loads(requests[1].content)
    assert "chat_template_kwargs" not in json.loads(requests[1].content)
    assert "vendor_option" not in json.loads(requests[1].content)




def test_all_direct_llm_entrypoints_use_shared_client():
    from src.harness import reader_grader

    assert reader_grader.AsyncOpenAI is ConfiguredAsyncOpenAI


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [
    {"enable_thinking": False},
    {"chat_template_kwargs": {"enable_thinking": False}},
    {"vendor_option": {"mode": "custom", "budget": 512}},
])
async def test_only_configured_parameters_are_added(configured):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=response_for(request))

    async with ConfiguredAsyncOpenAI(
        api_key="test", request_extra_body=configured,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as client:
        await client.chat.completions.create(model="any-model", messages=[])
    assert bodies == [{"model": "any-model", "messages": [], **configured}]


@pytest.mark.asyncio
async def test_recursive_override_of_native_fields_and_input_isolation():
    from copy import deepcopy

    configured = {"temperature": 0.3, "reasoning": {"effort": "low"},
                  "custom": {"items": [3], "nullable": None}}
    original = deepcopy(configured)
    extra = {"custom": {"items": [1, 2], "nullable": "old", "keep": True}}
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=response_for(request))

    async with ConfiguredAsyncOpenAI(
        api_key="test", request_extra_body=configured,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as client:
        # Client configuration is captured at construction, not shared with callers.
        configured["reasoning"]["effort"] = "high"
        await client.responses.create(
            model="any-model", input="test", temperature=0,
            reasoning={"effort": "high", "summary": "auto"}, extra_body=extra,
        )
    assert bodies[0]["temperature"] == 0.3
    assert bodies[0]["reasoning"] == {"effort": "low", "summary": "auto"}
    assert bodies[0]["custom"] == {"items": [3], "nullable": None, "keep": True}
    assert extra == {"custom": {"items": [1, 2], "nullable": "old", "keep": True}}
    assert client._request_extra_body == original
