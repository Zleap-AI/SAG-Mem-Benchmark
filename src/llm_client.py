"""Configurable JSON request parameters for OpenAI-compatible LLM endpoints."""

from copy import deepcopy
from typing import Any

from openai import AsyncOpenAI
from openai._models import FinalRequestOptions


def _merge_objects(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge objects; configured scalars, lists and null replace values."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_objects(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


class ConfiguredAsyncOpenAI(AsyncOpenAI):
    """Apply configured extra body fields to Chat/Responses, including SDK retries."""

    def __init__(self, *, request_extra_body: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._request_extra_body = deepcopy(request_extra_body or {})

    async def _prepare_options(self, options: FinalRequestOptions) -> FinalRequestOptions:
        options = await super()._prepare_options(options)
        if not self._request_extra_body:
            return options
        path = options.url.split("?", 1)[0].rstrip("/")
        if options.method.upper() != "POST" or not path.endswith(("/chat/completions", "/responses")):
            return options

        # Environment configuration wins over caller/SDK values without mutating
        # either input. First reproduce the SDK's shallow extra_body merge, then
        # apply our recursive overrides, including nested native SDK parameters.
        options = options.model_copy(deep=True)
        body = {**(options.json_data or {}), **(options.extra_json or {})}
        options.extra_json = _merge_objects(body, self._request_extra_body)
        return options
