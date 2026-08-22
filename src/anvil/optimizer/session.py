"""Async wrapper around ``ClaudeSDKClient`` — the optimizer's session.

The optimizer is the only async piece of ANVIL because the
``claude-agent-sdk`` package is async-only. Everything else (runtime,
eval, loop) is synchronous; the loop's :func:`run_round` calls
``asyncio.run(run_optimizer_session(...))`` to bridge.

Two responsibilities:

1. **Configure the env** — point the bundled Claude Code subprocess at
   the workspace's AI Gateway anthropic route (``ANTHROPIC_BASE_URL``,
   ``ANTHROPIC_DEFAULT_OPUS_MODEL``, the
   custom coding-agent header, the experimental-betas opt-out).
2. **Run one bounded session** — open ``ClaudeSDKClient`` with the
   prompt, drain ``receive_response`` into a transcript, and parse the
   final action JSON.

The prompt is composed elsewhere (loop.builder); this module only
takes a ready-to-send ``prompt`` string. That keeps the session
function loop-side-agnostic and trivially mockable in tests.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import mlflow
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk.types import (
    PermissionResultAllow,
    PermissionResultDeny,
    SandboxSettings,
)

from anvil.optimizer.actions import OptimizerAction
from anvil.optimizer.parser import ParseResult, parse_action
from anvil.optimizer.policy import ToolPolicy

logger = logging.getLogger(__name__)

# AI Gateway host for Claude Agent SDK. Set ``ANVIL_AI_GATEWAY_URL``
# to your workspace's gateway URL — typically
# ``https://<workspace-id>.ai-gateway.cloud.databricks.com/anthropic``.
# This is NOT the workspace's ``<host>/serving-endpoints/anthropic``
# route: that one rejects the Claude Code CLI's beta flags with
# HTTP 400. The AI Gateway path implements the same Anthropic Messages
# API but speaks the Databricks-native auth header set.
ANTHROPIC_BASE_URL = os.environ.get("ANVIL_AI_GATEWAY_URL", "")


def setup_anthropic_env(
    profile: str | None = None,
    optimizer_endpoint: str | None = None,
) -> None:
    """Point the Claude Agent SDK at the Databricks-hosted Anthropic gateway.

    Idempotent: existing values in ``os.environ`` are left alone so a
    developer running with a direct Anthropic key locally is not
    overridden.

    ``optimizer_endpoint`` is the FMAPI model name from
    ``harness/config.yaml > optimizer_endpoint``. When set, it becomes
    the Claude Code CLI's default model (``ANTHROPIC_MODEL`` and
    ``ANTHROPIC_DEFAULT_OPUS_MODEL``). When None, falls back to the
    built-in default (``databricks-claude-opus-4-7``).

    Gateway authentication is handled automatically by Claude Code through
    the Databricks CLI (using ``DATABRICKS_CONFIG_PROFILE`` or
    ``DATABRICKS_HOST``), so no secret or token is required. An operator-set
    ``ANTHROPIC_AUTH_TOKEN`` is preserved as an optional override.
    """
    if "ANTHROPIC_BASE_URL" not in os.environ:
        if not ANTHROPIC_BASE_URL:
            raise RuntimeError(
                "ANVIL_AI_GATEWAY_URL is unset and ANTHROPIC_BASE_URL is not "
                "in the environment. Set ANVIL_AI_GATEWAY_URL to your "
                "workspace's AI Gateway endpoint, e.g. "
                "https://<workspace-id>.ai-gateway.cloud.databricks.com/anthropic"
            )
        os.environ["ANTHROPIC_BASE_URL"] = ANTHROPIC_BASE_URL
    os.environ.setdefault("CLAUDE_CODE_USE_GATEWAY", "1")
    optimizer_model = optimizer_endpoint or "databricks-claude-opus-4-7"
    os.environ.setdefault("ANTHROPIC_MODEL", optimizer_model)
    os.environ.setdefault("ANTHROPIC_DEFAULT_OPUS_MODEL", optimizer_model)
    os.environ.setdefault("ANTHROPIC_DEFAULT_SONNET_MODEL", "databricks-claude-sonnet-4-6")
    os.environ.setdefault("ANTHROPIC_DEFAULT_HAIKU_MODEL", "databricks-claude-haiku-4-5")
    os.environ.setdefault(
        "ANTHROPIC_CUSTOM_HEADERS", "x-databricks-use-coding-agent-mode: true"
    )
    os.environ.setdefault("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "1")


async def run_optimizer_session(
    *,
    prompt: str,
    cwd: str,
    max_turns: int = 30,
    profile: str | None = None,
    setup_env: bool = True,
    optimizer_endpoint: str | None = None,
    experiment_name: str | None = None,
    round_id: int | None = None,
    policy: ToolPolicy | None = None,
    max_budget_usd: float | None = None,
) -> tuple[OptimizerAction, str, ParseResult]:
    """Open one ``ClaudeSDKClient`` session, drain it, parse the action.

    Args:
        prompt: Fully-composed user prompt (built by ``loop.builder``).
        cwd: Working directory for the Claude Code subprocess (typically
            the repo root).
        policy: Confinement policy. Defaults to :class:`ToolPolicy` rooted at
            ``cwd``, which permits writes under ``scaffold/`` and ``agents/``
            and denies reads of the evaluation answer key. See
            :mod:`anvil.optimizer.policy` for why each rule exists.
        max_budget_usd: Hard spend ceiling for the session, enforced by the
            SDK. Wire this to ``harness/config.yaml >
            loop.cost_budget_usd_per_round``.
        max_turns: Hard cap on optimizer CLI turns. The session aborts
            and returns whatever transcript it has if exceeded; the
            parser then falls back to ``NoopAction``.
        profile: Databricks CLI profile used by Claude Code for gateway auth.
        setup_env: If True (default), call :func:`setup_anthropic_env`
            before opening the session. Disable in tests.
        optimizer_endpoint: FMAPI model name from
            ``harness/config.yaml > optimizer_endpoint``. Forwarded to
            :func:`setup_anthropic_env` so the Claude Code CLI uses the
            configured model. When None, the built-in default is used.
        experiment_name: When set, the session opens an MLflow trace
            under this experiment and turns on
            ``mlflow.anthropic.autolog`` so each Anthropic API call
            inside the Claude Code subprocess becomes a child span.
            Disable in tests by passing ``None`` (default).
        round_id: Tag value for the ``round`` trace tag. Optional.

    Returns:
        A 3-tuple ``(action, transcript, parse_result)`` where ``action``
        is the parsed ``OptimizerAction`` (always populated; ``NoopAction``
        on parse failure), ``transcript`` is the raw text Claude emitted,
        and ``parse_result`` carries diagnostic metadata about the parse.
    """
    if setup_env:
        setup_anthropic_env(profile=profile, optimizer_endpoint=optimizer_endpoint)

    if experiment_name:
        if profile:
            mlflow.set_tracking_uri(f"databricks://{profile}")
        mlflow.set_experiment(experiment_name)
        # Turn on Anthropic autolog so each LLM call inside the Claude
        # Code subprocess becomes a CHAT_MODEL child span. Idempotent.
        # autolog can fail at import time on unsupported SDK versions;
        # the trace still wraps the session, just without per-call
        # children.
        with contextlib.suppress(Exception):
            mlflow.anthropic.autolog()

    policy = policy or ToolPolicy(root=Path(cwd))
    options = ClaudeAgentOptions(
        cwd=cwd,
        # Allowlist, not denylist: a tool added by a future SDK release arrives
        # denied. The previous `disallowed_tools=["AskUserQuestion",
        # "ExitPlanMode"]` denied two tools and permitted everything else,
        # including Bash and Write over the whole repo.
        allowed_tools=list(policy.allowed_tools),
        disallowed_tools=["AskUserQuestion", "ExitPlanMode"],
        setting_sources=[],
        max_turns=max_turns,
        # OS-level sandbox, independent of the permission callback below.
        # `allowUnsandboxedCommands=False` is the part that matters: without it
        # the CLI may escalate out of the sandbox to run a command.
        sandbox=SandboxSettings(
            enabled=True,
            allowUnsandboxedCommands=False,
            autoAllowBashIfSandboxed=False,
        ),
        # Enforces harness/config.yaml's `cost_budget_usd_per_round`, which the
        # loop declared and never applied.
        max_budget_usd=max_budget_usd,
        can_use_tool=_permission_callback(policy),
    )

    async def _drain_session() -> str:
        parts: list[str] = []
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                text = extract_message_text(message)
                if text:
                    parts.append(text)
        return "\n\n".join(parts)

    if experiment_name:
        with mlflow.start_span(name="anvil_optimizer_round") as span:
            tags: dict[str, str] = {"source": "optimizer"}
            if round_id is not None:
                tags["round"] = str(round_id)
            tags["max_turns"] = str(max_turns)
            with contextlib.suppress(Exception):
                mlflow.update_current_trace(tags=tags)
            span.set_inputs({"prompt_chars": len(prompt), "round_id": round_id})
            transcript = await _drain_session()
            span.set_outputs({"transcript_chars": len(transcript)})
    else:
        transcript = await _drain_session()

    parse_result = parse_action(transcript)
    return parse_result.action, transcript, parse_result


def extract_message_text(message) -> str:
    """Best-effort plain-text extractor for a ``claude-agent-sdk`` message.

    The SDK ships several message types — ``SystemMessage``,
    ``AssistantMessage`` (with ``content`` as a list of ``ThinkingBlock``,
    ``ToolUseBlock``, ``TextBlock`` ...), ``UserMessage`` (tool results),
    ``ResultMessage`` (final answer in ``.result``).

    The parser needs the raw text where the optimizer wrote its
    ```json-action ` fenced block. ``str(message)`` is wrong: it
    emits the Python ``repr`` with ``\\n`` escaped, breaking the
    regex's ``\\n`` matches. Instead, walk the typed shape via
    ``getattr`` (no hard import of SDK types — keeps the wrapper
    forward-compatible).

    Returns concatenated text from:

      * ``message.result`` if it's a string (``ResultMessage``).
      * ``block.text`` for every block in ``message.content`` that
        has a ``.text`` string attribute (``TextBlock`` inside an
        ``AssistantMessage``).

    Other block types (``ThinkingBlock``, ``ToolUseBlock``) are
    dropped — the optimizer's user-visible reasoning lives in
    ``TextBlock`` and the final ``ResultMessage`` only.
    """
    parts: list[str] = []

    result = getattr(message, "result", None)
    if isinstance(result, str):
        parts.append(result)

    content = getattr(message, "content", None)
    if isinstance(content, list):
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)

    return "\n".join(parts)


def _permission_callback(
    policy: ToolPolicy,
) -> Callable[[str, dict[str, Any], Any], Awaitable[PermissionResultAllow | PermissionResultDeny]]:
    """Build the SDK permission callback from ``policy``.

    Returns the SDK's typed results rather than the raw
    ``{"behavior": "allow"}`` dict the previous implementation used -- that
    worked by coercion and silently bypassed the declared contract.

    A denial carries the policy's reason so the model learns the boundary
    instead of retrying the same wall until it burns its turn budget.
    """

    async def _decide(
        tool_name: str, tool_input: dict[str, Any], _ctx: Any
    ) -> PermissionResultAllow | PermissionResultDeny:
        decision = policy.decide(tool_name, tool_input)
        if decision.allowed:
            return PermissionResultAllow(updated_input=tool_input)
        logger.warning("optimizer tool call denied: %s -- %s", tool_name, decision.reason)
        return PermissionResultDeny(message=decision.reason)

    return _decide
