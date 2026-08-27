"""Runtime agent: ``ResponsesAgent`` consuming a scaffold snapshot.

The agent loads the scaffold once at construction (via
:func:`anvil.runtime.loader.load_harness`) and keeps a reference to
it for the lifetime of the instance. ``predict`` runs a plain-Python
tool-calling loop against the configured Databricks serving endpoint,
bounded by ``sampling.max_tool_calls``.

Design constraints:

* No agent framework. The loop is explicit so the optimizer can
  reason about it with minimal indirection between scaffold files
  and the final prompt.
* Tool execution is delegated to a ``ToolExecutor`` callable so
  tests can inject fakes without standing up Unity Catalog functions.
* The scaffold is read once per instance. Each ANVIL round builds a
  fresh agent against the round's branch.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Protocol

from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from anvil.observability import SOURCE_PRODUCTION, SourceTag, tag_current_trace
from anvil.runtime.client import ChatClient, build_gateway_client
from anvil.runtime.loader import HarnessSnapshot, default_runtime_config_path, load_harness
from anvil.runtime.models import ToolRef


class ToolExecutor(Protocol):
    """Callable contract for executing a named tool with JSON arguments."""

    def __call__(self, name: str, arguments_json: str) -> str: ...


def _no_tools_executor(name: str, arguments_json: str) -> str:
    raise RuntimeError(
        f"Runtime agent has no ToolExecutor configured but model called tool "
        f"{name!r}. Register tools in scaffold/harness.yaml:tools[] or "
        f"inject a ToolExecutor in the constructor."
    )


class AnvilAgent(ResponsesAgent):
    """ANVIL runtime agent: ``ResponsesAgent`` over a scaffold snapshot."""

    def __init__(
        self,
        scaffold_root: Path | str,
        *,
        runtime_config_path: Path | str | None = None,
        source: SourceTag = SOURCE_PRODUCTION,
        client: ChatClient | None = None,
        tool_executor: ToolExecutor | None = None,
    ) -> None:
        self._scaffold_root: Path = Path(scaffold_root).resolve()
        resolved_runtime_config = (
            Path(runtime_config_path)
            if runtime_config_path is not None
            else default_runtime_config_path(self._scaffold_root)
        )
        self._snapshot: HarnessSnapshot = load_harness(self._scaffold_root, resolved_runtime_config)
        self._client: ChatClient = client if client is not None else build_gateway_client()
        self._tool_executor: ToolExecutor = tool_executor or _no_tools_executor
        self._source: SourceTag = source

    @property
    def snapshot(self) -> HarnessSnapshot:
        return self._snapshot

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        snap = self._snapshot
        tag_current_trace(
            source=self._source,
            scaffold_root=self._scaffold_root,
            runtime_endpoint=snap.runtime_endpoint,
        )
        messages = self._build_messages(request)
        tools = _openai_tool_defs(snap.tools)

        output_items: list[dict[str, Any]] = []
        # The +1 allows one terminal (no tool call) completion after
        # exhausting the tool-call budget.
        for _ in range(snap.sampling.max_tool_calls + 1):
            call_kwargs: dict[str, Any] = {
                "model": snap.runtime_endpoint,
                "messages": messages,
                "max_tokens": snap.sampling.max_tokens,
            }
            # Only include sampling params that are set. Databricks
            # serving rejects both `temperature` and `top_p` together
            # on some models (e.g. Claude Sonnet 4.6), and rejects
            # `tools=None` outright.
            if snap.sampling.temperature is not None:
                call_kwargs["temperature"] = snap.sampling.temperature
            if snap.sampling.top_p is not None:
                call_kwargs["top_p"] = snap.sampling.top_p
            if tools:
                call_kwargs["tools"] = tools
                call_kwargs["tool_choice"] = snap.sampling.tool_choice
            completion = self._client.chat.completions.create(**call_kwargs)
            msg = completion.choices[0].message
            tool_calls = list(msg.tool_calls or [])

            if not tool_calls:
                output_items.append(_assistant_message_item(msg.content or ""))
                return ResponsesAgentResponse(output=output_items)

            messages.append(_assistant_tool_call_message(msg, tool_calls))
            for tc in tool_calls:
                output_items.append(
                    {
                        "id": _new_id("fc"),
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                )
                result = self._tool_executor(tc.function.name, tc.function.arguments)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                output_items.append(
                    {
                        "id": _new_id("fco"),
                        "type": "function_call_output",
                        "call_id": tc.id,
                        "output": result,
                    }
                )

        output_items.append(
            _assistant_message_item(
                f"(max_tool_calls={snap.sampling.max_tool_calls} exceeded without a final reply)"
            )
        )
        return ResponsesAgentResponse(output=output_items)

    def _build_messages(self, request: ResponsesAgentRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self._snapshot.system_prompt:
            messages.append({"role": "system", "content": self._snapshot.system_prompt})

        for item in request.input:
            messages.append(_openai_message_from_input_item(item))
        return messages


def _openai_tool_defs(tools: list[ToolRef]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in tools:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    # An empty schema leaves the model to guess argument names
                    # from the description; measured on the MultiHopRAG domain,
                    # half to three quarters of rows then arrive as `{}` calls
                    # and die in the executor's validation.
                    "parameters": t.parameters
                    or {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    },
                },
            }
        )
    return out


def _openai_message_from_input_item(item: Any) -> dict[str, Any]:
    data = _as_dict(item)
    item_type = data.get("type", "message")

    if item_type == "message":
        role = data.get("role", "user")
        content = data.get("content", "")
        if isinstance(content, list):
            text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            content_str = "".join(text_parts)
        else:
            content_str = str(content)
        return {"role": role, "content": content_str}

    if item_type == "function_call_output":
        return {
            "role": "tool",
            "tool_call_id": data.get("call_id", ""),
            "content": data.get("output", ""),
        }

    return {"role": "user", "content": json.dumps(data, default=str)}


def _as_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        return dump()
    return dict(getattr(item, "__dict__", {}))


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _assistant_message_item(text: str) -> dict[str, Any]:
    return {
        "id": _new_id("msg"),
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def _assistant_tool_call_message(msg: Any, tool_calls: list[Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ],
    }
