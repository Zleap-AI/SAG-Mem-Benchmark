"""Model identifiers used in the paper; live endpoints and keys come from the repository root .env.

Reader / Judge (§4.1): ``PROFILES``. JEV evidence evaluator (Table 2): ``_PAPER_JEV``.
SAG-Mem-Lite encoder (§3.5): ``EMBEDDING_PROFILES``.
"""
import os

from reproduction.load_env import load_repo_env

PROFILES = {
    "qwen": {"model": "qwen/qwen3.6-35b-a3b", "extra_body": {
        "reasoning": {"enabled": False},
        "provider": {"only": ["DeepInfra"], "allow_fallbacks": False, "quantizations": ["fp8"]}}},
    "deepseek": {"model": "deepseek/deepseek-v4.1-flash-1017", "extra_body": {
        "reasoning": {"enabled": False},
        "provider": {"only": ["DeepSeek"], "allow_fallbacks": False}}},
}

# Encoders for SAG-Mem-Lite. ``max_input_tokens`` is the encoder window; the Lite backend refuses
# to send a longer chunk unless an overflow rule is selected explicitly (see lite/backend.py).
EMBEDDING_PROFILES = {
    # The encoder named in §3.5 of the technical report. Set EMBEDDING_MODEL_NAME when your
    # endpoint serves it under another identifier (e.g. text-embedding-bge-large-en-v1.5).
    "bge": {"model": "BAAI/bge-large-en-v1.5", "dimensions": 1024, "max_input_tokens": 512,
            "send_dimensions": False, "tokenizer": "bge", "extra_body": {}},
    # The encoder used by the Lite runs recorded in experiments/lite/README.md.
    "qwen3": {"model": "qwen/qwen3-embedding-8b", "dimensions": 4096, "max_input_tokens": 32768,
              "send_dimensions": True, "tokenizer": "qwen3",
              "extra_body": {"provider": {"only": ["DeepInfra"], "allow_fallbacks": False}}},
}
DEFAULT_EMBEDDING_PROFILE = "qwen3"
EMBEDDING_OVERFLOW_RULES = ("error", "prefix")

_PAPER_JEV = ("typesafe/jev-1.14-20261014", "https://openrouter.ai/api/alpha/decisions")


def embedding_profile(options: dict) -> dict:
    """Resolve the Lite encoder for a run; ``options`` is the harness option dict."""
    name = options.get("embedding_profile") or DEFAULT_EMBEDDING_PROFILE
    if name not in EMBEDDING_PROFILES:
        raise ValueError(f"Unknown embedding profile {name!r}; choose from {sorted(EMBEDDING_PROFILES)}")
    overflow = options.get("embedding_overflow") or "error"
    if overflow not in EMBEDDING_OVERFLOW_RULES:
        raise ValueError(f"Unknown embedding overflow rule {overflow!r}; choose from {EMBEDDING_OVERFLOW_RULES}")
    profile = {**EMBEDDING_PROFILES[name], "name": name, "overflow": overflow}
    if name == "bge":
        profile["model"] = os.environ.get("EMBEDDING_MODEL_NAME", "").strip() or profile["model"]
    return profile


def apply_paper_defaults_only_when_unset() -> None:
    """OpenRouter paper pins apply only if the repository .env left a slot empty."""
    load_repo_env()
    model, base = _PAPER_JEV
    os.environ.setdefault("JEV_MODEL", model)
    os.environ.setdefault("JEV_BASE_URL", base)


def llm_settings(require_jev: bool = True):
    """Return Settings after repository .env is loaded; SAG-Mem-Lite never calls JEV."""
    load_repo_env()
    from src.config import settings

    missing = []
    if not settings.search_llm.api_key:
        missing.append("SEARCH_LLM_API_KEY")
    if not settings.search_llm.base_url:
        missing.append("SEARCH_LLM_BASE_URL")
    if not settings.search_llm.model:
        missing.append("SEARCH_LLM_MODEL")
    if not settings.judge_llm.api_key:
        missing.append("JUDGE_LLM_API_KEY")
    if not settings.judge_llm.base_url:
        missing.append("JUDGE_LLM_BASE_URL")
    if not settings.judge_llm.model:
        missing.append("JUDGE_LLM_MODEL")
    if require_jev:
        if not settings.jev_api_key:
            missing.append("JEV_API_KEY")
        if not settings.jev_base_url:
            missing.append("JEV_BASE_URL")
        if not settings.jev_model:
            missing.append("JEV_MODEL")
    if missing:
        raise ValueError("Set in repository .env: " + ", ".join(missing))
    return settings


def reader_judge_models(profile: dict, require_jev: bool = True) -> tuple[str, str]:
    settings = llm_settings(require_jev)
    reader = settings.search_llm.model or profile["model"]
    judge = settings.judge_llm.model or profile["model"]
    return reader, judge


def api_settings(require_jev: bool = True):
    endpoint = llm_settings(require_jev).search_llm
    return {"api_key": endpoint.api_key, "base_url": endpoint.base_url or ""}
