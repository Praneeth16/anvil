"""Strict parser for ``OptimizerAction`` JSON blocks emitted by the optimizer.

The optimizer's session ends with a single JSON object matching one of
the :mod:`anvil.optimizer.actions` variants, wrapped in a fenced block::

    ```json-action
    {
      "action": "add_rule",
      "target_file": "rules/foo.md",
      "content": "...",
      "rationale": "..."
    }
    ```

This module finds and parses that block. **Every failure mode collapses to
a ``NoopAction``** so the loop never crashes on a bad transcript:

* No JSON block at all                          → noop
* JSON parses but does not match any schema     → noop
* Multiple action blocks                        → noop (ambiguous)

The original transcript is always returned alongside the action so the
loop can persist it (`scaffold/memory/round_NNN_critique.md`) for the
next round's lookback.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import TypeAdapter, ValidationError

from anvil.optimizer.actions import NoopAction, OptimizerAction

_FENCE_RE = re.compile(
    r"```(?:json-action|json|action)\s*\n(?P<body>.*?)\n```",
    flags=re.DOTALL,
)

_action_adapter: TypeAdapter[OptimizerAction] = TypeAdapter(OptimizerAction)


@dataclass(frozen=True)
class ParseResult:
    """Output of :func:`parse_action` — always populated, never raises."""

    action: OptimizerAction
    parse_status: str  # "ok" | "ok_last_of_many" | "no_block" | "bad_json" | "schema_mismatch"
    raw_block: str | None = None
    n_blocks_found: int = 0


def parse_action(transcript: str) -> ParseResult:
    """Extract a single ``OptimizerAction`` from a transcript.

    The optimizer occasionally emits more than one fenced block — for
    example, a worked example earlier in its reasoning plus the final
    decision at the bottom. Treat the **last** block as the decision;
    earlier blocks were exploration and are dropped. Mark the parse
    status as ``ok_last_of_many`` when this happens so the loop's
    critique md records that the transcript had multiple candidates.

    Defensive: every other failure mode returns a ``NoopAction``.
    """
    blocks = _FENCE_RE.findall(transcript or "")
    n_blocks = len(blocks)
    if n_blocks == 0:
        return ParseResult(
            action=NoopAction(rationale="parser: no `json-action` fenced block in transcript"),
            parse_status="no_block",
            n_blocks_found=0,
        )
    raw_block = blocks[-1].strip()
    base_status = "ok" if n_blocks == 1 else "ok_last_of_many"
    try:
        data = json.loads(raw_block)
    except json.JSONDecodeError as exc:
        return ParseResult(
            action=NoopAction(rationale=f"parser: JSON malformed ({exc})"),
            parse_status="bad_json",
            raw_block=raw_block,
            n_blocks_found=n_blocks,
        )
    try:
        action = _action_adapter.validate_python(data)
    except ValidationError as exc:
        # Take the first error to keep rationale short.
        first_err = exc.errors()[0]
        loc = ".".join(str(x) for x in first_err.get("loc", ()))
        msg = first_err.get("msg", str(exc))
        return ParseResult(
            action=NoopAction(rationale=f"parser: schema mismatch at {loc!r}: {msg}"),
            parse_status="schema_mismatch",
            raw_block=raw_block,
            n_blocks_found=n_blocks,
        )
    return ParseResult(
        action=action, parse_status=base_status, raw_block=raw_block, n_blocks_found=n_blocks
    )
