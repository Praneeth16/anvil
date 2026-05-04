"""Tests for the message-text extractor in ``optimizer.session``.

The extractor is the bridge between the ``claude-agent-sdk`` SDK's
typed messages and the parser's regex-over-text contract. Round 1 (the
first real round) failed because the legacy ``str(message)`` emitted
``\\n``-escaped output, breaking the regex. This test pins the new
contract on dataclass fakes that mimic the SDK shapes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from anvil.optimizer.parser import parse_action
from anvil.optimizer.session import extract_message_text


# ---------------------------------------------------------------------------
# Fake SDK shapes — only the attributes ``extract_message_text`` reads.
# ---------------------------------------------------------------------------


@dataclass
class FakeTextBlock:
    text: str


@dataclass
class FakeThinkingBlock:
    thinking: str
    # NOTE: no `.text` — extractor must skip.


@dataclass
class FakeToolUseBlock:
    name: str
    input: dict
    # NOTE: no `.text` — extractor must skip.


@dataclass
class FakeAssistantMessage:
    content: list


@dataclass
class FakeResultMessage:
    result: str
    content: list | None = None


@dataclass
class FakeSystemMessage:
    subtype: str
    data: dict


# ---------------------------------------------------------------------------
# Extractor unit tests
# ---------------------------------------------------------------------------


def test_extract_text_from_assistant_with_text_blocks() -> None:
    msg = FakeAssistantMessage(
        content=[FakeThinkingBlock(thinking="hidden"), FakeTextBlock(text="hello world")]
    )
    assert extract_message_text(msg) == "hello world"


def test_extract_text_concatenates_multiple_text_blocks() -> None:
    msg = FakeAssistantMessage(
        content=[FakeTextBlock(text="part one"), FakeTextBlock(text="part two")]
    )
    assert extract_message_text(msg) == "part one\npart two"


def test_extract_text_skips_thinking_and_tool_use() -> None:
    msg = FakeAssistantMessage(
        content=[
            FakeThinkingBlock(thinking="should not appear"),
            FakeToolUseBlock(name="Read", input={"file_path": "x"}),
            FakeTextBlock(text="visible"),
        ]
    )
    assert extract_message_text(msg) == "visible"


def test_extract_result_from_result_message() -> None:
    msg = FakeResultMessage(result="final answer text")
    assert extract_message_text(msg) == "final answer text"


def test_extract_returns_empty_for_system_message() -> None:
    msg = FakeSystemMessage(subtype="init", data={"foo": "bar"})
    # No `.result`, no `.content` list.
    assert extract_message_text(msg) == ""


# ---------------------------------------------------------------------------
# Round-trip: extractor + parser see a real-shape transcript end-to-end.
# ---------------------------------------------------------------------------


def test_round_trip_finds_action_block() -> None:
    """The fix: a TextBlock containing a real ``json-action`` block parses end-to-end."""
    block_body = """
some reasoning ...

```json-action
{
  "action": "add_rule",
  "target_file": "rules/foo.md",
  "content": "---\\nrule_id: foo\\napplies_to: runtime\\n---\\n\\n# foo\\nbody\\n",
  "rationale": "seed test"
}
```
"""
    msg = FakeAssistantMessage(content=[FakeTextBlock(text=block_body)])
    transcript = extract_message_text(msg)

    result = parse_action(transcript)
    assert result.parse_status == "ok"
    assert result.action.action == "add_rule"
    assert result.action.target_file == "rules/foo.md"
    assert "seed test" in result.action.rationale


def test_round_trip_picks_up_result_message_block() -> None:
    """The SDK's final ``ResultMessage.result`` carries the same text — should also work."""
    final = """
Confirmed.

```json-action
{
  "action": "noop",
  "rationale": "no actionable failure cluster this round"
}
```
"""
    msg = FakeResultMessage(result=final)
    transcript = extract_message_text(msg)

    result = parse_action(transcript)
    assert result.parse_status == "ok"
    assert result.action.action == "noop"


def test_round_trip_legacy_str_repr_does_not_parse() -> None:
    """Regression: the legacy ``str(message)`` shape emits ``\\\\n`` and breaks parsing.

    Pin the shape to make sure nobody silently regresses to ``str(message)``.
    """
    legacy_repr = (
        "AssistantMessage(content=[TextBlock(text='```json-action\\n"
        '{\\n  "action": "noop",\\n  "rationale": "x"\\n}\\n'
        "```')], model='claude-opus-4-7')"
    )
    result = parse_action(legacy_repr)
    assert result.parse_status == "no_block"
    # And confirms the new extractor would NOT touch this — only test that
    # the parser itself is regex-strict.
