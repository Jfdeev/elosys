"""LLM client: provider selection, rate limiting and tolerant response parsing (no network)."""

from __future__ import annotations

import json

import pytest

from elosys.ai import deepseek as llm


class _Resp:
    def __init__(self, content='{"ok": 1}', status=200):
        self.status_code = status
        self._body = {"choices": [{"message": {"content": content}}],
                      "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


@pytest.fixture
def fake_post(monkeypatch):
    calls = []

    def _install(content='{"ok": 1}', status=200):
        def _post(url, headers, json, timeout):  # noqa: A002
            calls.append({"url": url, "headers": headers, "json": json})
            return _Resp(content, status)
        monkeypatch.setattr(llm.requests, "post", _post)
        return calls
    return _install


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.setattr(llm, "_last_call_at", {})
    monkeypatch.setattr(llm, "_exhausted", set())
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)


def test_defaults_to_deepseek(monkeypatch, fake_post):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    calls = fake_post()

    out = llm.chat_json("sys", "user")

    assert calls[0]["url"] == "https://api.deepseek.com/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer ds-key"
    assert out["data"] == {"ok": 1}
    assert (out["prompt_tokens"], out["completion_tokens"]) == (7, 3)


def test_gemini_uses_its_url_key_and_model(monkeypatch, fake_post):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "gm-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-be-used")
    calls = fake_post()

    llm.chat_json("sys", "user", model=llm.PROVIDERS["gemini"].default_model)

    assert calls[0]["url"].startswith("https://generativelanguage.googleapis.com/v1beta/openai/")
    assert calls[0]["headers"]["Authorization"] == "Bearer gm-key"
    assert calls[0]["json"]["model"] == "gemini-3.1-flash-lite"


def test_missing_key_error_names_the_providers_variable(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(llm.DeepSeekError, match="GEMINI_API_KEY"):
        llm.chat_json("s", "u")


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(llm.DeepSeekError, match="LLM_PROVIDER"):
        llm.chat_json("s", "u")


def test_throttle_spaces_calls_by_the_providers_interval(monkeypatch):
    sleeps = []
    clock = iter([100.0, 100.0, 101.0, 101.0])
    monkeypatch.setattr(llm.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    llm._wait_for_slot("m", 4.0)   # first call to this model: no wait
    llm._wait_for_slot("m", 4.0)   # 1s later: must wait the remaining 3s

    assert sleeps == [pytest.approx(3.0)]


def test_throttle_is_tracked_per_model(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    llm._wait_for_slot("a", 4.0)
    llm._wait_for_slot("b", 4.0)   # another model has its own quota: no wait

    assert sleeps == []


def test_no_throttle_when_interval_is_zero(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    llm._wait_for_slot("m", 0.0)
    llm._wait_for_slot("m", 0.0)
    assert sleeps == []


def test_json_wrapped_in_code_fence_is_accepted(monkeypatch, fake_post):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    fake_post('```json\n{"verdict": "bizarro"}\n```')
    assert llm.chat_json("s", "u")["data"] == {"verdict": "bizarro"}


@pytest.mark.parametrize("content", [None, "", "   "])
def test_empty_content_is_a_clean_error(monkeypatch, fake_post, content):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    fake_post(content)
    with pytest.raises(llm.DeepSeekError, match="vazia"):
        llm.chat_json("s", "u")


def test_non_json_content_is_a_clean_error(monkeypatch, fake_post):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    fake_post("desculpe, não posso ajudar")
    with pytest.raises(llm.DeepSeekError, match="JSON válido"):
        llm.chat_json("s", "u")


def test_http_error_reports_status(monkeypatch, fake_post):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    fake_post(status=429)
    with pytest.raises(llm.DeepSeekError, match="HTTP 429"):
        llm.chat_json("s", "u")


def test_result_reports_the_model_that_answered(monkeypatch, fake_post):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    fake_post()
    assert llm.chat_json("s", "u")["model"] == "deepseek-chat"


def _gemini(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "k")


def _scripted_post(monkeypatch, statuses_by_model):
    """Each call pops the next status for the requested model; returns the list of models called."""
    called = []

    def _post(url, headers, json, timeout):  # noqa: A002
        called.append(json["model"])
        return _Resp(status=statuses_by_model[json["model"]].pop(0))
    monkeypatch.setattr(llm.requests, "post", _post)
    return called


def test_model_chain_starts_with_requested_then_the_others(monkeypatch):
    _gemini(monkeypatch)
    chain = llm.model_chain("gemini-3.5-flash-lite")
    assert chain[0] == "gemini-3.5-flash-lite"
    assert sorted(chain) == sorted(m.name for m in llm.PROVIDERS["gemini"].models)
    assert len(chain) == len(set(chain))


def test_falls_back_to_next_model_on_429(monkeypatch):
    _gemini(monkeypatch)
    first, second = "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"
    called = _scripted_post(monkeypatch, {first: [429], second: [200]})

    out = llm.chat_json("s", "u", model=first)

    assert called == [first, second]
    assert out["model"] == second


def test_exhausted_model_is_not_retried_on_later_calls(monkeypatch):
    _gemini(monkeypatch)
    first, second = "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"
    called = _scripted_post(monkeypatch, {first: [429], second: [200, 200]})

    llm.chat_json("s", "u", model=first)
    llm.chat_json("s", "u", model=first)

    assert called == [first, second, second]


def test_non_quota_errors_do_not_trigger_fallback(monkeypatch):
    _gemini(monkeypatch)
    first = "gemini-3.1-flash-lite"
    called = _scripted_post(monkeypatch, {first: [400]})

    with pytest.raises(llm.DeepSeekError, match="HTTP 400"):
        llm.chat_json("s", "u", model=first)
    assert called == [first]


def test_when_every_model_is_out_of_quota_the_last_one_is_retried_next_call(monkeypatch):
    _gemini(monkeypatch)
    names = [m.name for m in llm.PROVIDERS["gemini"].models]
    statuses = {n: [429] for n in names}
    statuses[names[-1]] = [429, 429]
    called = _scripted_post(monkeypatch, statuses)

    with pytest.raises(llm.DeepSeekError, match="HTTP 429"):
        llm.chat_json("s", "u", model=names[0])
    assert called == names

    # the others stay skipped; only the last model (never marked exhausted) is tried again
    with pytest.raises(llm.DeepSeekError, match="HTTP 429"):
        llm.chat_json("s", "u", model=names[0])
    assert called == [*names, names[-1]]


def test_chain_with_every_model_marked_exhausted_fails_without_calling_the_api(monkeypatch):
    _gemini(monkeypatch)
    names = [m.name for m in llm.PROVIDERS["gemini"].models]
    monkeypatch.setattr(llm, "_exhausted", set(names))
    called = _scripted_post(monkeypatch, {})

    with pytest.raises(llm.DeepSeekError, match="esgotada em todos"):
        llm.chat_json("s", "u", model=names[0])
    assert called == []


def test_single_model_provider_429_is_not_marked_exhausted(monkeypatch, fake_post):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    fake_post(status=429)
    with pytest.raises(llm.DeepSeekError, match="HTTP 429"):
        llm.chat_json("s", "u")
    assert llm._exhausted == set()


def test_transient_503_is_retried_on_the_same_model_after_a_pause(monkeypatch):
    _gemini(monkeypatch)
    first = "gemini-3.1-flash-lite"
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    called = _scripted_post(monkeypatch, {first: [503, 200]})

    out = llm.chat_json("s", "u", model=first)

    assert called == [first, first]
    assert out["model"] == first
    assert llm.TRANSIENT_BACKOFF_SECONDS in sleeps


def test_persistent_503_falls_back_without_marking_the_model_exhausted(monkeypatch):
    _gemini(monkeypatch)
    first, second = "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"
    called = _scripted_post(monkeypatch, {first: [503, 503, 200], second: [200]})

    out = llm.chat_json("s", "u", model=first)
    assert (called, out["model"]) == ([first, first, second], second)
    assert first not in llm._exhausted

    # a later call tries the first model again (the outage may be over)
    assert llm.chat_json("s", "u", model=first)["model"] == first
    assert called[-1] == first


@pytest.mark.parametrize("status", sorted(llm.TRANSIENT_STATUSES))
def test_every_transient_status_is_retried(monkeypatch, status):
    _gemini(monkeypatch)
    first = "gemini-3.1-flash-lite"
    called = _scripted_post(monkeypatch, {first: [status, 200]})
    assert llm.chat_json("s", "u", model=first)["model"] == first
    assert called == [first, first]


def test_persistent_503_on_the_last_model_raises_the_http_error(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    called = _scripted_post(monkeypatch, {"deepseek-chat": [503, 503]})

    with pytest.raises(llm.DeepSeekError, match="HTTP 503"):
        llm.chat_json("s", "u")
    assert len(called) == 1 + llm.TRANSIENT_RETRIES
