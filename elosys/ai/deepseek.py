"""Thin chat client for OpenAI-compatible LLM APIs (DeepSeek by default, Gemini optional).

The provider is picked with LLM_PROVIDER ("deepseek" | "gemini"); the API key is read from the
provider's own variable (DEEPSEEK_API_KEY / GEMINI_API_KEY) and never stored in the project.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import NamedTuple

from curl_cffi import requests

from ..log import get_logger

log = get_logger("elosys.ai.llm")


class ModelQuota(NamedTuple):
    name: str
    min_interval_seconds: float  # spacing between calls, to respect the per-minute quota


class Provider(NamedTuple):
    api_url: str
    key_env: str
    models: tuple[ModelQuota, ...]  # fallback order: first is the default

    @property
    def default_model(self) -> str:
        return self.models[0].name


def _per_minute(requests_per_minute: int) -> float:
    return 60 / requests_per_minute + 0.1


PROVIDERS = {
    "deepseek": Provider(
        "https://api.deepseek.com/chat/completions", "DEEPSEEK_API_KEY",
        (ModelQuota("deepseek-chat", 0.0),),
    ),
    # Gemini free tier, one quota per model (RPM / requests per day), most daily requests first.
    "gemini": Provider(
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "GEMINI_API_KEY",
        (
            ModelQuota("gemini-3.1-flash-lite", _per_minute(15)),  # 15 RPM / 500 RPD
            ModelQuota("gemini-3.5-flash-lite", _per_minute(15)),  # 15 RPM / 500 RPD
            ModelQuota("gemini-3.6-flash", _per_minute(5)),        # 5 RPM / 20 RPD
            ModelQuota("gemini-3.7-flash", _per_minute(5)),        # 5 RPM / 20 RPD
            ModelQuota("gemini-3.8-flash", _per_minute(5)),        # 5 RPM / 20 RPD
        ),
    ),
}
DEFAULT_PROVIDER = "deepseek"

DEFAULT_MODEL = PROVIDERS.get(
    os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER).lower(), PROVIDERS[DEFAULT_PROVIDER]
).default_model

_state_lock = threading.Lock()
_last_call_at: dict[str, float] = {}
_exhausted: set[str] = set()  # models that answered 429 in this process


TRANSIENT_STATUSES = frozenset({500, 502, 503, 504})  # server-side hiccups, usually short-lived
TRANSIENT_RETRIES = 1
TRANSIENT_BACKOFF_SECONDS = 5.0


class DeepSeekError(RuntimeError):
    """Any LLM API failure (name kept for the callers; it covers every provider)."""


class _QuotaError(DeepSeekError):
    """HTTP 429 from the API."""


class _TransientError(DeepSeekError):
    """HTTP 5xx that is worth retrying."""


def _provider() -> Provider:
    name = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER).lower()
    if name not in PROVIDERS:
        raise DeepSeekError(f"LLM_PROVIDER inválido: {name!r} (use: {', '.join(PROVIDERS)})")
    return PROVIDERS[name]


def has_key() -> bool:
    return bool(os.environ.get(_provider().key_env))


def model_chain(model: str) -> list[str]:
    """`model` first, then the provider's other models in fallback order."""
    others = [m.name for m in _provider().models if m.name != model]
    return [model, *others]


def _wait_for_slot(model: str, min_interval: float) -> None:
    """Block until `min_interval` seconds passed since the previous call to `model` (thread-safe)."""
    if min_interval <= 0:
        return
    with _state_lock:
        wait = _last_call_at.get(model, 0.0) + min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call_at[model] = time.monotonic()


def _interval_for(provider: Provider, model: str) -> float:
    known = {m.name: m.min_interval_seconds for m in provider.models}
    return known.get(model, max(known.values(), default=0.0))


def _strip_code_fence(text: str) -> str:
    """Some models wrap JSON in ```json fences even when asked for a JSON object."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def chat_json(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    timeout: int = 120,
) -> dict:
    """Ask for a JSON answer. On HTTP 429 (quota) falls through to the next model of the chain;
    the result's "model" is the one that actually answered."""
    provider = _provider()
    key = os.environ.get(provider.key_env)
    if not key:
        raise DeepSeekError(
            f"{provider.key_env} não definida no ambiente (no PowerShell: $env:{provider.key_env} = \"...\")"
        )

    chain = model_chain(model)
    for position, candidate in enumerate(chain):
        with _state_lock:
            if candidate in _exhausted:
                continue
            has_alternative = any(m not in _exhausted for m in chain[position + 1:])
        try:
            out = _call_with_retry(provider, key, candidate, system_prompt, user_prompt, temperature, timeout)
        except _TransientError as e:
            if not has_alternative:
                raise DeepSeekError(str(e)) from e
            log.warning("%s indisponível (%s) — usando o próximo modelo", candidate, e)
            continue
        except _QuotaError as e:
            if not has_alternative:
                raise DeepSeekError(str(e)) from e
            with _state_lock:
                _exhausted.add(candidate)
            log.warning("cota esgotada em %s — usando o próximo modelo", candidate)
            continue
        out["model"] = candidate
        return out
    raise DeepSeekError(f"cota esgotada em todos os modelos: {', '.join(chain)}")


def _call_with_retry(*call_args) -> dict:
    """`_call`, retried after a short pause when the server reports a transient failure."""
    model = call_args[2]
    for attempt in range(TRANSIENT_RETRIES + 1):
        try:
            return _call(*call_args)
        except _TransientError:
            if attempt == TRANSIENT_RETRIES:
                raise
            log.warning("%s: erro temporário do servidor, nova tentativa em %.0fs",
                        model, TRANSIENT_BACKOFF_SECONDS)
            time.sleep(TRANSIENT_BACKOFF_SECONDS)
    raise AssertionError("unreachable")  # pragma: no cover


def _call(provider: Provider, key: str, model: str, system_prompt: str, user_prompt: str,
          temperature: float, timeout: int) -> dict:
    _wait_for_slot(model, _interval_for(provider, model))
    try:
        r = requests.post(
            provider.api_url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": temperature,
                "stream": False,
            },
            timeout=timeout,
        )
    except Exception as e:  # noqa: BLE001
        raise DeepSeekError(f"falha de rede: {e}") from e

    if r.status_code == 429:
        raise _QuotaError(f"HTTP 429: {r.text[:400]}")
    if r.status_code in TRANSIENT_STATUSES:
        raise _TransientError(f"HTTP {r.status_code}: {r.text[:400]}")
    if r.status_code != 200:
        raise DeepSeekError(f"HTTP {r.status_code}: {r.text[:400]}")

    try:
        body = r.json()
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage") or {}
    except (KeyError, IndexError, ValueError) as e:
        raise DeepSeekError(f"resposta inesperada da API: {r.text[:400]}") from e
    if not isinstance(content, str) or not content.strip():
        raise DeepSeekError(f"modelo devolveu resposta vazia (bloqueio de segurança?): {r.text[:400]}")

    try:
        data = json.loads(_strip_code_fence(content))
    except json.JSONDecodeError as e:
        raise DeepSeekError(f"modelo não devolveu JSON válido: {content[:400]}") from e

    return {
        "data": data,
        "raw": content,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }
