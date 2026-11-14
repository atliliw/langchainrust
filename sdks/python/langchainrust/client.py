"""Zero-dependency client for the langchainrust OpenAI-compatible server.

The client only speaks HTTP: it does not re-implement any agent logic. Point
it at a running ``lc serve`` (or any OpenAI-compatible endpoint) and use
``chat`` / ``stream`` / ``chat_with_tools``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

Message = Dict[str, Any]


class LangChainRustError(RuntimeError):
    """Raised when the server returns a non-2xx response or invalid JSON."""

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class ToolCall:
    """One tool call requested by the model."""

    id: str
    name: str
    arguments: str


@dataclass
class ChatResult:
    """Non-streaming chat result."""

    content: str
    finish_reason: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Optional[Dict[str, Any]] = None
    raw: Optional[Dict[str, Any]] = None


def _as_messages(input: Union[str, Sequence[Message]]) -> List[Message]:
    if isinstance(input, str):
        return [{"role": "user", "content": input}]
    return list(input)


class Client:
    """Minimal OpenAI-compatible client.

    Args:
        base_url: e.g. ``http://localhost:8080``.
        model: model id advertised by the server (``/v1/models``).
        api_key: optional bearer token.
        timeout: per-request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model: str = "default",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    # ------------------------------------------------------------------ HTTP
    def _headers(self) -> Dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=self._headers(),
            method=method,
        )
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LangChainRustError(
                f"HTTP {exc.code}: {body}", status=exc.code
            ) from exc

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._request("POST", path, payload) as resp:
            body = resp.read().decode("utf-8")
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise LangChainRustError(f"invalid JSON response: {body[:200]}") from exc

    # ------------------------------------------------------------------ API
    def models(self) -> List[Dict[str, Any]]:
        """Returns the server's model list."""
        with self._request("GET", "/v1/models") as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload.get("data", [])

    def chat(
        self,
        input: Union[str, Sequence[Message]],
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        **options: Any,
    ) -> ChatResult:
        """One non-streaming completion."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": _as_messages(input),
        }
        if tools:
            payload["tools"] = list(tools)
        payload.update(options)
        return self._parse_result(self._post_json("/v1/chat/completions", payload))

    def chat_with_tools(
        self,
        input: Union[str, Sequence[Message]],
        tools: Sequence[Dict[str, Any]],
        **options: Any,
    ) -> ChatResult:
        """Convenience wrapper for ``chat(..., tools=...)``."""
        return self.chat(input, tools=tools, **options)

    def stream(
        self,
        input: Union[str, Sequence[Message]],
        **options: Any,
    ) -> Iterator[str]:
        """Yields text deltas from an SSE response."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": _as_messages(input),
            "stream": True,
        }
        payload.update(options)

        resp = self._request("POST", "/v1/chat/completions", payload)
        try:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    return
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = delta.get("content")
                if text:
                    yield text
        finally:
            resp.close()

    # ------------------------------------------------------------------ parse
    @staticmethod
    def _parse_result(payload: Dict[str, Any]) -> ChatResult:
        choices = payload.get("choices") or []
        if not choices:
            raise LangChainRustError("response contained no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        tool_calls = [
            ToolCall(
                id=call.get("id", ""),
                name=(call.get("function") or {}).get("name", ""),
                arguments=(call.get("function") or {}).get("arguments", ""),
            )
            for call in (message.get("tool_calls") or [])
        ]
        return ChatResult(
            content=message.get("content") or "",
            finish_reason=choice.get("finish_reason"),
            tool_calls=tool_calls,
            usage=payload.get("usage"),
            raw=payload,
        )
