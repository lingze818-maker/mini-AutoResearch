from __future__ import annotations

import json

import pytest

from agent.limits import MAX_MODEL_CALLS, MAX_MODEL_RETRIES, MAX_MODEL_TOKENS, MODEL_TIMEOUT_S, OFFLINE_BANNER
from agent.model import (
    HTTPResult,
    LiveConfigError,
    LiveDeepSeekProvider,
    ModelError,
    _RefuseRedirect,
    build_provider,
    official_completions_url,
    proposal_tool_specs,
)
from agent.offline import OfflineFixtureProvider
from tests.conftest import ROOT

_KEY = "sk-test-secret-value"


def _summaries():
    return {
        "510300": {"symbol": "510300", "period_return": 0.1, "in_sample_end": "2023-12-31"},
        "510500": {"symbol": "510500", "period_return": -0.2, "in_sample_end": "2023-12-31"},
    }


def _candidates():
    source = "def generate(bars):\n    return {'entry': False, 'exit': False}\n"
    return [
        {"id": "flat_a", "name": "flat a", "rationale": "preset", "source": source},
        {"id": "flat_b", "name": "flat b", "rationale": "preset", "source": source},
    ]


def _reply(arguments: dict) -> HTTPResult:
    body = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "submit_candidates",
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ]
                }
            }
        ]
    }
    return HTTPResult(200, json.dumps(body).encode("utf-8"))


def test_offline_provider_is_deterministic_and_marked():
    provider = OfflineFixtureProvider()
    first = provider.propose(_summaries())
    assert provider.banner == OFFLINE_BANNER
    assert provider.calls == 1
    with pytest.raises(RuntimeError):
        provider.propose(_summaries())
    other = OfflineFixtureProvider().propose(
        {"510300": {"symbol": "510300", "period_return": -0.5, "in_sample_end": "2023-12-31"}}
    )
    assert [item.source for item in first] == [item.source for item in other]
    assert len(first) == 3
    assert all(OFFLINE_BANNER in item.rationale for item in first)


def test_offline_build_does_not_require_a_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    provider = build_provider(False)
    assert isinstance(provider, OfflineFixtureProvider)
    assert provider.banner == OFFLINE_BANNER


def test_live_mode_missing_key_or_model_does_not_fall_back(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_MODEL", "configured-by-operator")
    with pytest.raises(LiveConfigError, match="DEEPSEEK_API_KEY"):
        build_provider(True)
    monkeypatch.setenv("DEEPSEEK_API_KEY", _KEY)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    with pytest.raises(LiveConfigError, match="DEEPSEEK_MODEL"):
        build_provider(True)


def test_live_request_uses_the_configured_model_and_bounds():
    seen = {}

    def transport(url, headers, body, timeout_s):
        seen["url"] = url
        seen["headers"] = dict(headers)
        seen["body"] = json.loads(body.decode("utf-8"))
        seen["timeout"] = timeout_s
        return _reply({"candidates": _candidates()})

    provider = LiveDeepSeekProvider(api_key=_KEY, model="configured-by-operator", transport=transport)
    proposed = provider.propose(_summaries())
    assert [item.id for item in proposed] == ["flat_a", "flat_b"]
    assert seen["body"]["model"] == "configured-by-operator"
    assert seen["body"]["max_tokens"] == MAX_MODEL_TOKENS
    assert seen["body"]["max_tokens"] <= MAX_MODEL_TOKENS
    assert seen["timeout"] == MODEL_TIMEOUT_S
    assert seen["headers"]["Authorization"] == f"Bearer {_KEY}"
    assert _KEY.encode("utf-8") not in json.dumps(seen["body"]).encode("utf-8")
    names = [item["function"]["name"] for item in seen["body"]["tools"]]
    assert names == ["summarize_in_sample", "backtest_candidate", "write_report", "submit_candidates"]
    assert provider.http_attempts == 1


def test_tool_schemas_have_no_path_or_shell_arguments():
    for spec in proposal_tool_specs():
        parameters = spec["function"]["parameters"]
        assert "path" not in parameters["properties"]
        assert "shell" not in parameters["properties"]
        assert parameters["additionalProperties"] is False


def test_malformed_tool_arguments_and_verdicts_are_rejected():
    def transport(_url, _headers, _body, _timeout):
        return _reply({"candidates": _candidates(), "path": "/tmp/x"})

    provider = LiveDeepSeekProvider(api_key=_KEY, model="configured-by-operator", transport=transport)
    with pytest.raises(ModelError, match="paths"):
        provider.propose(_summaries())

    def verdict_transport(_url, _headers, _body, _timeout):
        rows = _candidates()
        rows[0]["decision"] = "pass"
        return _reply({"candidates": rows})

    provider = LiveDeepSeekProvider(api_key=_KEY, model="configured-by-operator", transport=verdict_transport)
    with pytest.raises(ModelError, match="verdict"):
        provider.propose(_summaries())


def test_unknown_tool_and_host_only_tools_do_not_run_a_backtest():
    calls = {"n": 0}

    def transport(_url, _headers, _body, _timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            body = {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_backtest",
                                    "type": "function",
                                    "function": {
                                        "name": "backtest_candidate",
                                        "arguments": json.dumps({"candidate_id": "flat_a"}),
                                    },
                                },
                                {
                                    "id": "call_shell",
                                    "type": "function",
                                    "function": {"name": "shell", "arguments": "{}"},
                                },
                            ]
                        }
                    }
                ]
            }
            return HTTPResult(200, json.dumps(body).encode("utf-8"))
        return _reply({"candidates": _candidates()})

    provider = LiveDeepSeekProvider(api_key=_KEY, model="configured-by-operator", transport=transport)
    proposed = provider.propose(_summaries())
    assert [item.id for item in proposed] == ["flat_a", "flat_b"]
    assert calls["n"] == 2
    assert provider.calls == 1


def test_retries_stop_after_the_budget_and_errors_hide_the_key():
    attempts = {"n": 0}

    def timeout_transport(*_args):
        attempts["n"] += 1
        raise TimeoutError(_KEY)

    provider = LiveDeepSeekProvider(api_key=_KEY, model="configured-by-operator", transport=timeout_transport)
    with pytest.raises(ModelError, match="timed out") as caught:
        provider.propose(_summaries())
    assert attempts["n"] == MAX_MODEL_RETRIES + 1
    assert _KEY not in str(caught.value)

    def crash_transport(*_args):
        raise RuntimeError(f"Authorization: Bearer {_KEY}")

    provider = LiveDeepSeekProvider(api_key=_KEY, model="configured-by-operator", transport=crash_transport)
    with pytest.raises(ModelError, match="model request failed") as caught:
        provider.propose(_summaries())
    assert _KEY not in str(caught.value)


def test_http_400_is_not_retried_and_call_budget_is_two_rounds():
    attempts = {"n": 0}

    def reject_transport(*_args):
        attempts["n"] += 1
        return HTTPResult(400, b'{"error":"bad"}')

    provider = LiveDeepSeekProvider(api_key=_KEY, model="configured-by-operator", transport=reject_transport)
    with pytest.raises(ModelError, match="status 400"):
        provider.propose(_summaries())
    assert attempts["n"] == 1

    def spin_transport(*_args):
        attempts["n"] += 1
        body = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": f"call_{attempts['n']}",
                                "type": "function",
                                "function": {
                                    "name": "summarize_in_sample",
                                    "arguments": json.dumps({"symbol": "510300"}),
                                },
                            }
                        ]
                    }
                }
            ]
        }
        return HTTPResult(200, json.dumps(body).encode("utf-8"))

    attempts["n"] = 0
    provider = LiveDeepSeekProvider(
        api_key=_KEY,
        model="configured-by-operator",
        transport=spin_transport,
        max_calls=MAX_MODEL_CALLS,
    )
    with pytest.raises(ModelError, match="budget"):
        provider.propose(_summaries())
    assert attempts["n"] == MAX_MODEL_CALLS


def test_summary_tool_does_not_accept_an_out_of_sample_payload():
    def transport(_url, _headers, _body, _timeout):
        return _reply({"candidates": _candidates()})

    provider = LiveDeepSeekProvider(api_key=_KEY, model="configured-by-operator", transport=transport)
    poisoned = _summaries()
    poisoned["510300"] = {**poisoned["510300"], "out_of_sample": {"period_return": 1}}
    with pytest.raises(ModelError, match="out-of-sample"):
        provider.propose(poisoned)


def test_only_the_official_https_origin_can_receive_the_credential():
    assert official_completions_url("https://api.deepseek.com") == "https://api.deepseek.com/chat/completions"
    assert official_completions_url("https://API.DEEPSEEK.COM") == "https://api.deepseek.com/chat/completions"
    rejected = [
        "http://api.deepseek.com",
        "http://127.0.0.1",
        "https://127.0.0.1",
        "https://api.deepseek.com.evil.com",
        "https://evil.com",
        "https://user:pass@api.deepseek.com",
        "https://api.deepseek.com@evil.com",
        "https://api.deepseek.com:8443",
        "https://api.deepseek.com/other",
        "https://api.deepseek.com?x=1",
        " https://api.deepseek.com",
    ]
    for base in rejected:
        with pytest.raises(ModelError):
            official_completions_url(base)

    def transport(*_args):
        raise AssertionError("credential was sent")

    with pytest.raises(ModelError):
        LiveDeepSeekProvider(api_key=_KEY, model="configured-by-operator", base_url="http://127.0.0.1", transport=transport)


def test_redirects_are_not_followed_with_the_credential():
    handler = _RefuseRedirect()
    with pytest.raises(ModelError, match="redirect"):
        handler.redirect_request(None, None, 302, "Found", {}, "http://127.0.0.1/steal")


def _choice(arguments, *, finish_reason="tool_calls", usage=None, name="submit_candidates"):
    body = {
        "usage": usage
        or {"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33, "ignored": "no"},
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ]
                },
            }
        ],
    }
    return HTTPResult(200, json.dumps(body).encode("utf-8"))


def test_object_arguments_are_accepted_without_a_second_decode():
    def transport(_url, _headers, _body, _timeout):
        return _choice({"candidates": _candidates()})

    provider = LiveDeepSeekProvider(api_key=_KEY, model="configured-by-operator", transport=transport)
    proposed = provider.propose(_summaries())
    assert [item.id for item in proposed] == ["flat_a", "flat_b"]
    assert provider.http_attempts == 1


def test_truncated_and_malformed_arguments_fail_closed_without_retry():
    calls = {"n": 0}

    def transport(_url, _headers, _body, _timeout):
        calls["n"] += 1
        return _choice('{"candidates":[{"id":"flat_a"', finish_reason="length")

    provider = LiveDeepSeekProvider(api_key=_KEY, model="configured-by-operator", transport=transport)
    with pytest.raises(ModelError, match="tool arguments are not valid json") as caught:
        provider.propose(_summaries())
    assert calls["n"] == 1
    assert provider.http_attempts == 1
    diagnostics = caught.value.diagnostics
    assert diagnostics["finish_reason"] == "length"
    assert diagnostics["tool_name"] == "submit_candidates"
    assert diagnostics["argument_type"] == "str"
    assert diagnostics["argument_length"] == len('{"candidates":[{"id":"flat_a"')
    assert diagnostics["json_error"] == "JSONDecodeError"
    assert diagnostics["usage"] == {"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33}
    assert "ignored" not in json.dumps(diagnostics)
    blob = json.dumps(diagnostics)
    assert "flat_a" not in blob
    assert _KEY not in blob

    def garbage(_url, _headers, _body, _timeout):
        calls["n"] += 1
        return _choice("{not json")

    calls["n"] = 0
    provider = LiveDeepSeekProvider(api_key=_KEY, model="configured-by-operator", transport=garbage)
    with pytest.raises(ModelError, match="tool arguments are not valid json"):
        provider.propose(_summaries())
    assert calls["n"] == 1


def test_a_later_well_formed_call_does_not_rescue_bad_arguments():
    def transport(_url, _headers, _body, _timeout):
        body = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_bad",
                                "type": "function",
                                "function": {"name": "submit_candidates", "arguments": '{"candidates":'},
                            },
                            {
                                "id": "call_ok",
                                "type": "function",
                                "function": {
                                    "name": "submit_candidates",
                                    "arguments": json.dumps({"candidates": _candidates()}),
                                },
                            },
                        ]
                    },
                }
            ]
        }
        return HTTPResult(200, json.dumps(body).encode("utf-8"))

    provider = LiveDeepSeekProvider(api_key=_KEY, model="configured-by-operator", transport=transport)
    with pytest.raises(ModelError, match="tool arguments are not valid json"):
        provider.propose(_summaries())


def test_source_does_not_hardcode_a_model_id():
    text = (ROOT / "src" / "agent" / "model.py").read_text(encoding="utf-8")
    for banned in ("deepseek-chat", "deepseek-reasoner", "deepseek-v4", "gpt-4"):
        assert banned not in text
    assert "https://api.deepseek.com" in (ROOT / "src" / "agent" / "limits.py").read_text(encoding="utf-8")
