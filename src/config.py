"""Central settings loaded from .env at the project root."""
import json
import os
import re
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / '.env')

def _get(name: str, default: str='') -> str:
    val = os.getenv(name, default).strip()
    return val if val else default

def _get_first(*names: str, default: str='') -> str:
    """Return the first non-blank value among `names`, in precedence order.

    Role settings list the explicit variable first and the legacy name second
    (e.g. EXTRACT_LLM_MODEL, LLM_MODEL), so each unset or blank override falls
    back item by item instead of switching the whole role at once.
    """
    for name in names:
        value = _get(name)
        if value:
            return value
    return default

def _get_json_object(name: str) -> dict[str, Any]:

    def reject_constant(_value: str) -> None:
        raise ValueError('Non-finite numbers are not JSON values')
    try:
        value = json.loads(_get(name, '{}'), parse_constant=reject_constant)
    except ValueError:
        raise ValueError(f'{name} must be a valid JSON object') from None
    if not isinstance(value, dict):
        raise TypeError(f'{name} must be a JSON object')
    return value

def _get_first_json_object(*names: str) -> dict[str, Any]:
    """First non-blank JSON object among `names`, reported under its own name."""
    for name in names:
        if _get(name):
            return _get_json_object(name)
    return {}

def _detect_host_label() -> str:
    """Return a stable per-machine label derived from the hostname.

    Unlike the local IP, the hostname does not change with DHCP leases, VPNs,
    or WiFi/Ethernet switches, so runs from the same machine stay in one MLflow
    experiment. Reads a local string only -- no network traffic, no subprocess.
    The domain suffix (e.g. ".local") is dropped and any character outside
    [A-Za-z0-9._-] is replaced with "-" so the result is a safe experiment name.
    """
    try:
        host = socket.gethostname()
    except OSError:
        host = ''
    host = host.split('.', 1)[0].strip()
    host = re.sub('[^A-Za-z0-9._-]+', '-', host).strip('-')
    return host or 'unknown-host'

def _default_mlflow_experiment_name() -> str:
    return f'{_detect_host_label()}_sag_mem'

@dataclass(frozen=True)
class LLMEndpoint:
    """One resolved LLM role: endpoint, credentials, model and request extras."""
    api_key: str = field(repr=False)
    base_url: str | None
    model: str
    extra_body: dict[str, Any]

@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(default_factory=lambda : _get('OPENAI_API_KEY'))
    openai_base_url: str | None = field(default_factory=lambda : _get('OPENAI_BASE_URL') or None)
    response_model: str = field(default_factory=lambda : _get_first('SEARCH_LLM_MODEL', 'RESPONSE_MODEL', default='gpt-4o'))
    grader_model: str = field(default_factory=lambda : _get_first('JUDGE_LLM_MODEL', 'GRADER_MODEL', default='gpt-4o'))
    openai_extra_body: dict[str, Any] = field(default_factory=lambda : _get_json_object('OPENAI_EXTRA_BODY'))
    reader_api_key: str = field(default_factory=lambda : _get_first('SEARCH_LLM_API_KEY', 'READER_API_KEY'))
    reader_base_url: str | None = field(default_factory=lambda : _get_first('SEARCH_LLM_BASE_URL', 'READER_BASE_URL') or None)
    judge_api_key: str = field(default_factory=lambda : _get_first('JUDGE_LLM_API_KEY', 'JUDGE_API_KEY'))
    judge_base_url: str | None = field(default_factory=lambda : _get_first('JUDGE_LLM_BASE_URL', 'JUDGE_BASE_URL') or None)
    reader_extra_body: dict[str, Any] | None = field(default_factory=lambda : _get_json_object('SEARCH_LLM_EXTRA_BODY') if _get('SEARCH_LLM_EXTRA_BODY') else None)
    judge_extra_body: dict[str, Any] | None = field(default_factory=lambda : _get_json_object('JUDGE_LLM_EXTRA_BODY') if _get('JUDGE_LLM_EXTRA_BODY') else None)
    llm_api_key: str = field(default_factory=lambda : _get_first('EXTRACT_LLM_API_KEY', 'LLM_API_KEY'))
    llm_base_url: str = field(default_factory=lambda : _get_first('EXTRACT_LLM_BASE_URL', 'LLM_BASE_URL'))
    llm_model: str = field(default_factory=lambda : _get_first('EXTRACT_LLM_MODEL', 'LLM_MODEL', default='qwen3.6-flash'))
    llm_extra_body: dict[str, Any] = field(default_factory=lambda : _get_first_json_object('EXTRACT_LLM_EXTRA_BODY', 'LLM_EXTRA_BODY'))
    jev_api_key: str = field(default_factory=lambda : _get('JEV_API_KEY'))
    jev_base_url: str = field(default_factory=lambda : _get('JEV_BASE_URL'))
    jev_model: str = field(default_factory=lambda : _get('JEV_MODEL', 'jev-latest'))
    jev_timeout_seconds: int = field(default_factory=lambda : int(_get('JEV_TIMEOUT_SECONDS', '120')))
    jev_concurrency: int = field(default_factory=lambda : int(_get('JEV_CONCURRENCY', '2')))
    embedding_api_key: str = field(default_factory=lambda : _get('EMBEDDING_API_KEY'))
    embedding_base_url: str = field(default_factory=lambda : _get('EMBEDDING_BASE_URL'))
    embedding_model: str = field(default_factory=lambda : _get('EMBEDDING_MODEL_NAME'))
    embedding_dimensions: int = field(default_factory=lambda : int(_get('EMBEDDING_DIMENSIONS', '1024') or 1024))
    mlflow_url: str = field(default_factory=lambda : _get('MLFLOW_URL', 'http://localhost:5000'))
    mlflow_experiment_name: str = field(default_factory=lambda : _get('MLFLOW_EXPERIMENT_NAME') or _default_mlflow_experiment_name())

    @property
    def effective_reader_api_key(self) -> str:
        return self.reader_api_key or self.openai_api_key

    @property
    def effective_reader_base_url(self) -> str | None:
        return self.reader_base_url or self.openai_base_url

    @property
    def effective_judge_api_key(self) -> str:
        return self.judge_api_key or self.openai_api_key

    @property
    def effective_judge_base_url(self) -> str | None:
        return self.judge_base_url or self.openai_base_url

    @property
    def extract_llm(self) -> LLMEndpoint:
        """Resolve the legacy extraction endpoint for shared configuration compatibility."""
        return LLMEndpoint(api_key=self.llm_api_key, base_url=self.llm_base_url or None, model=self.llm_model, extra_body=self.llm_extra_body)

    @property
    def search_llm(self) -> LLMEndpoint:
        """Resolve the Reader endpoint that answers from retrieved evidence."""
        return LLMEndpoint(api_key=self.effective_reader_api_key, base_url=self.effective_reader_base_url, model=self.response_model, extra_body=self.openai_extra_body if self.reader_extra_body is None else self.reader_extra_body)

    @property
    def judge_llm(self) -> LLMEndpoint:
        """Role: the Judge that grades a Reader answer."""
        return LLMEndpoint(api_key=self.effective_judge_api_key, base_url=self.effective_judge_base_url, model=self.grader_model, extra_body=self.openai_extra_body if self.judge_extra_body is None else self.judge_extra_body)

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / 'data' / 'longmemeval'

    @property
    def results_dir(self) -> Path:
        return PROJECT_ROOT / 'results'
settings = Settings()

def require_keys(*names: str) -> None:
    """Raise with a clear message if required settings are missing."""
    missing = [n for n in names if not getattr(settings, n)]
    if missing:
        raise SystemExit(f"Missing required settings: {', '.join(missing)}. Please fill .env at {PROJECT_ROOT / '.env'}")
