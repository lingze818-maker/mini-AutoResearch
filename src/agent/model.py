"""Model client for a planned DeepSeek tool-calling request.

The HTTP shape follows a common chat-completions tool-calling layout. It has
not been verified against a live account. The model id is always supplied by
the caller. The offline provider is the default and is not this client.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from agent.limits import (
    DEFAULT_BASE_URL,
    MAX_MODEL_CALLS,
    MAX_MODEL_RETRIES,
    MAX_MODEL_TOKENS,
    MAX_SOURCE_CHARS,
    MODEL_TIMEOUT_S,
    OFFICIAL_API_HOST,
    OFFLINE_BANNER,
)
from agent.offline import Candidate, OfflineFixtureProvider

_ID = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_FORBIDDEN_KEYS = {"path", "paths", "cmd", "command", "shell", "filename", "directory", "file"}
_VERDICT_KEYS = {"decision", "rule_id", "verdict"}


class ModelError(Exception):
    """The live client rejected an input or an untrusted reply."""

    def __init__(self, message: str, *, diagnostics: dict | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class LiveConfigError(ModelError):
    """Live mode was requested without the configuration it requires."""


@dataclass(frozen=True, slots=True)
class HTTPResult:
    status: int
    body: bytes


Transport = Callable[[str, Mapping[str, str], bytes, float], HTTPResult]


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    """Do not send the bearer token to a second origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ModelError("refusing to follow a redirect with the API credential")


def official_completions_url(base_url: str) -> str:
    """Accept only https://api.deepseek.com with no credentials, query, or port override."""
    if not isinstance(base_url, str):
        raise ModelError("DEEPSEEK_BASE_URL must be https://api.deepseek.com")
    raw = base_url.strip()
    if raw != base_url or any(char in raw for char in "\r\n\t "):
        raise ModelError("DEEPSEEK_BASE_URL must be https://api.deepseek.com")
    parts = urllib.parse.urlsplit(raw)
    if parts.scheme != "https" or parts.hostname != OFFICIAL_API_HOST:
        raise ModelError("DEEPSEEK_BASE_URL must be https://api.deepseek.com")
    if parts.username or parts.password:
        raise ModelError("DEEPSEEK_BASE_URL must not include credentials")
    if parts.port not in (None, 443):
        raise ModelError("DEEPSEEK_BASE_URL must be https://api.deepseek.com")
    if parts.query or parts.fragment or parts.path not in ("", "/"):
        raise ModelError("DEEPSEEK_BASE_URL must be https://api.deepseek.com")
    return "https://api.deepseek.com/chat/completions"


def urllib_transport(url: str, headers: Mapping[str, str], body: bytes, timeout_s: float) -> HTTPResult:
    import urllib.error
    import urllib.request

    if url != official_completions_url(DEFAULT_BASE_URL):
        raise ModelError("refusing to send the API credential to a non-official endpoint")
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    opener = urllib.request.build_opener(_RefuseRedirect)
    try:
        with opener.open(request, timeout=timeout_s) as response:
            return HTTPResult(status=getattr(response, "status", 200), body=response.read(1_000_000))
    except ModelError:
        raise
    except urllib.error.HTTPError as exc:
        if exc.code in {301, 302, 303, 307, 308}:
            raise ModelError("refusing to follow a redirect with the API credential") from None
        return HTTPResult(status=exc.code, body=exc.read(200_000))


def build_provider(live: bool, *, transport: Transport | None = None):
    """Offline unless ``live`` is set. Missing live settings do not fall back."""
    if not live:
        return OfflineFixtureProvider()
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    model = os.environ.get("DEEPSEEK_MODEL", "")
    if not key:
        raise LiveConfigError(
            "DEEPSEEK_API_KEY is missing; live mode does not fall back to the offline fixture"
        )
    if not model:
        raise LiveConfigError(
            "DEEPSEEK_MODEL is missing; this program does not select a model id"
        )
    base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
    return LiveDeepSeekProvider(
        api_key=key,
        model=model,
        base_url=base_url,
        transport=transport or urllib_transport,
    )


def proposal_tool_specs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "summarize_in_sample",
                "description": "Return the already computed in-sample summary for one configured symbol.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["symbol"],
                    "properties": {"symbol": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "backtest_candidate",
                "description": "Host-only. This bounded run does not return backtest or out-of-sample results to the model.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["candidate_id"],
                    "properties": {"candidate_id": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_report",
                "description": "Host-only. Reports are written by trusted code. The model cannot set a path or a verdict.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"note": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_candidates",
                "description": (
                    "Submit 2 or 3 causal strategies. Each source defines generate(bars) "
                    "and will see only the prefix up to the bar it decides."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["candidates"],
                    "properties": {
                        "candidates": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 3,
                        }
                    },
                },
            },
        },
    ]


def _reject_keys(payload: Mapping[str, object]) -> None:
    found = _FORBIDDEN_KEYS.intersection(payload)
    if found:
        raise ModelError("tool arguments cannot include paths or shell commands")
    if _VERDICT_KEYS.intersection(payload):
        raise ModelError("the model cannot set a verdict")


def validate_candidates(candidates: list[Candidate]) -> list[Candidate]:
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 3:
        raise ModelError("a proposal must contain 2 or 3 candidates")
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Candidate):
            raise ModelError("candidate record is invalid")
        if not _ID.fullmatch(candidate.id):
            raise ModelError("candidate id is invalid")
        if candidate.id in seen:
            raise ModelError("candidate ids must be unique")
        seen.add(candidate.id)
        if not isinstance(candidate.name, str) or not candidate.name.strip() or len(candidate.name) > 80:
            raise ModelError("candidate name is invalid")
        if not isinstance(candidate.rationale, str) or len(candidate.rationale) > 2000:
            raise ModelError("candidate rationale is invalid")
        if not isinstance(candidate.source, str) or "\x00" in candidate.source:
            raise ModelError("candidate source is invalid")
        if not candidate.source.strip() or len(candidate.source) > MAX_SOURCE_CHARS:
            raise ModelError("candidate source is invalid")
        if "def generate" not in candidate.source:
            raise ModelError("candidate source must define generate")
    return candidates


def _parse_candidates(arguments: Mapping[str, object]) -> list[Candidate]:
    _reject_keys(arguments)
    raw = arguments.get("candidates")
    if not isinstance(raw, list):
        raise ModelError("candidates must be a list")
    parsed: list[Candidate] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ModelError("each candidate must be an object")
        _reject_keys(item)
        parsed.append(
            Candidate(
                id=str(item.get("id", "")),
                name=str(item.get("name", "")),
                rationale=str(item.get("rationale", "")),
                source=str(item.get("source", "")),
            )
        )
    return validate_candidates(parsed)


class LiveDeepSeekProvider:
    """Unverified tool-calling client. It does not see out-of-sample results."""

    name = "live-deepseek-unverified"
    banner = "live model request, unverified integration"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        transport: Transport = urllib_transport,
        max_calls: int = MAX_MODEL_CALLS,
        max_tokens: int = MAX_MODEL_TOKENS,
        timeout_s: float = MODEL_TIMEOUT_S,
        max_retries: int = MAX_MODEL_RETRIES,
    ) -> None:
        if not api_key:
            raise LiveConfigError(
                "DEEPSEEK_API_KEY is missing; live mode does not fall back to the offline fixture"
            )
        if not model or not str(model).strip():
            raise LiveConfigError("DEEPSEEK_MODEL is missing; this program does not select a model id")
        if max_calls < 1 or max_calls > MAX_MODEL_CALLS:
            raise ModelError("model call budget is outside the allowed range")
        if max_tokens < 1 or max_tokens > MAX_MODEL_TOKENS:
            raise ModelError("token budget is outside the allowed range")
        if timeout_s <= 0 or timeout_s > MODEL_TIMEOUT_S:
            raise ModelError("model timeout is outside the allowed range")
        if max_retries < 0 or max_retries > MAX_MODEL_RETRIES:
            raise ModelError("retry budget is outside the allowed range")
        self._api_key = api_key
        self._model = model
        self._url = official_completions_url(base_url)
        self._transport = transport
        self._max_calls = max_calls
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self.calls = 0
        self.http_attempts = 0
        self.http_responses = 0

    def propose(self, summaries: Mapping[str, Mapping[str, object]]) -> list[Candidate]:
        self.calls += 1
        _assert_summaries(summaries)
        messages: list[dict] = [
            {
                "role": "system",
                "content": (
                    "Propose 2 or 3 causal daily strategies by calling submit_candidates. "
                    "generate(bars) receives only the prefix up to the current bar. "
                    "Do not request paths, shell commands, or a verdict. "
                    "Out-of-sample results are not available."
                ),
            },
            {"role": "user", "content": json.dumps({"in_sample_summaries": list(summaries.values())}, ensure_ascii=False)},
        ]
        for _round in range(self._max_calls):
            message, reply_meta = self._complete(messages)
            try:
                tool_calls = message.get("tool_calls") or []
                if not isinstance(tool_calls, list):
                    raise ModelError("tool_calls must be a list")
                submission = None
                results = []
                for call in tool_calls:
                    name, arguments, call_id = _parse_call(call)
                    if name == "submit_candidates":
                        submission = _parse_candidates(arguments)
                        continue
                    if name == "summarize_in_sample":
                        content = _summary_result(arguments, summaries)
                    elif name == "backtest_candidate":
                        _reject_keys(arguments)
                        content = {
                            "error": "backtest_candidate is host-only in this bounded run and does not return out-of-sample results"
                        }
                    elif name == "write_report":
                        _reject_keys(arguments)
                        content = {"error": "write_report is host-only; the model cannot set a path or a verdict"}
                    else:
                        content = {"error": "tool is not available to the model in this bounded run"}
                    results.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(content)})
                if submission is not None:
                    return submission
                if not results:
                    raise ModelError("model reply did not submit candidates")
            except ModelError as exc:
                exc.diagnostics = _public_diagnostics(reply_meta, exc.diagnostics)
                raise
            messages.append(message)
            messages.extend(results)
        raise ModelError("model call budget exhausted before candidates were submitted")

    def _complete(self, messages: list[dict]) -> dict:
        body = {
            "model": self._model,
            "temperature": 0,
            "max_tokens": self._max_tokens,
            "messages": messages,
            "tools": proposal_tool_specs(),
        }
        payload = json.dumps(body).encode("utf-8")
        if self._api_key and self._api_key.encode("utf-8") in payload:
            raise ModelError("refusing to send the API key in the request body")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        last_error = "model request failed"
        for attempt in range(self._max_retries + 1):
            self.http_attempts += 1
            try:
                result = self._transport(self._url, headers, payload, self._timeout_s)
            except TimeoutError:
                last_error = "model request timed out"
                if attempt < self._max_retries:
                    continue
                raise ModelError(last_error) from None
            except Exception:
                raise ModelError("model request failed") from None
            if result.status in {500, 502, 503, 504}:
                last_error = f"model request failed with status {result.status}"
                if attempt < self._max_retries:
                    continue
                raise ModelError(last_error)
            if result.status != 200:
                raise ModelError(f"model request rejected with status {result.status}")
            self.http_responses += 1
            return _message_from_body(result.body)
        raise ModelError(last_error)


def _message_from_body(body: bytes) -> tuple[dict, dict]:
    if len(body) > 1_000_000:
        raise ModelError("model response is too large")
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise ModelError("model response is not valid json") from None
    if not isinstance(parsed, dict):
        raise ModelError("model response is not an object")
    choices = parsed.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ModelError("model response did not contain one choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ModelError("model response did not contain a message")
    return message, {"finish_reason": choice.get("finish_reason"), "usage": parsed.get("usage")}


def _public_diagnostics(reply_meta: Mapping[str, object], call_diag: Mapping[str, object]) -> dict:
    """Metadata only. Never copy argument text, headers, or secrets."""
    out: dict = {}
    reason = reply_meta.get("finish_reason")
    if isinstance(reason, str) and re.fullmatch(r"[a-z0-9_]{1,32}", reason):
        out["finish_reason"] = reason
    usage = reply_meta.get("usage")
    counts: dict[str, int] = {}
    if isinstance(usage, dict):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                counts[key] = value
    if counts:
        out["usage"] = counts
    tool_name = call_diag.get("tool_name")
    if isinstance(tool_name, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", tool_name):
        out["tool_name"] = tool_name
    if call_diag.get("argument_type") in {"str", "dict", "list", "int", "float", "bool", "NoneType"}:
        out["argument_type"] = call_diag["argument_type"]
    length = call_diag.get("argument_length")
    if isinstance(length, int) and not isinstance(length, bool) and 0 <= length <= 10_000_000:
        out["argument_length"] = length
    if call_diag.get("json_error") == "JSONDecodeError":
        out["json_error"] = "JSONDecodeError"
    return out


def _parse_call(call: object) -> tuple[str, dict, str]:
    if not isinstance(call, dict):
        raise ModelError("tool call is invalid")
    function = call.get("function")
    if not isinstance(function, dict):
        raise ModelError("tool call is invalid")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ModelError("tool name is invalid")
    raw = function.get("arguments", {})
    if isinstance(raw, str):
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError:
            raise ModelError(
                "tool arguments are not valid json",
                diagnostics={
                    "tool_name": name,
                    "argument_type": "str",
                    "argument_length": len(raw),
                    "json_error": "JSONDecodeError",
                },
            ) from None
    elif isinstance(raw, dict):
        arguments = raw
    else:
        raise ModelError(
            "tool arguments must be an object",
            diagnostics={"tool_name": name, "argument_type": type(raw).__name__},
        )
    if not isinstance(arguments, dict):
        raise ModelError("tool arguments must be an object")
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id or len(call_id) > 80:
        raise ModelError("tool call id is invalid")
    return name, arguments, call_id


def _summary_result(arguments: Mapping[str, object], summaries: Mapping[str, Mapping[str, object]]) -> dict:
    _reject_keys(arguments)
    symbol = arguments.get("symbol")
    if not isinstance(symbol, str) or symbol not in summaries:
        return {"error": "unknown symbol"}
    summary = dict(summaries[symbol])
    _assert_summaries({symbol: summary})
    return summary


def _assert_summaries(summaries: Mapping[str, Mapping[str, object]]) -> None:
    for summary in summaries.values():
        if "out_of_sample" in summary or "decision" in summary or "rule_id" in summary:
            raise ModelError("in-sample summaries must not include a verdict or out-of-sample fields")


def offline_banner() -> str:
    return OFFLINE_BANNER
