"""Tool-spec construction for the runtime agent.

The load-bearing guard is the shipped-scaffold check: `_openai_tool_defs`
falls back to an empty JSON Schema when a tool declares no `parameters`,
and an empty schema leaves the model to guess argument names from the
description — measured on the MultiHopRAG domain, half to three quarters of
tool calls then arrive as `{}` and die in the executor's validation. Every
shipped scaffold must declare its tools' schemas.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anvil.runtime.agent import _openai_tool_defs
from anvil.runtime.loader import load_harness
from anvil.runtime.models import ToolRef

REPO_ROOT = Path(__file__).resolve().parent.parent

SHIPPED_SCAFFOLDS = [
    REPO_ROOT / "scaffold",
    REPO_ROOT / "examples" / "neovolt" / "scaffold",
    REPO_ROOT / "examples" / "pyloom-docs" / "scaffold",
]


@pytest.mark.unit
def test_declared_parameters_reach_the_tool_spec() -> None:
    tools = [
        ToolRef(
            name="search_knowledge_base",
            description="Search the KB.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    ]
    (spec,) = _openai_tool_defs(tools)
    assert spec["function"]["parameters"]["required"] == ["query"]
    assert "query" in spec["function"]["parameters"]["properties"]


@pytest.mark.unit
def test_undeclared_parameters_fall_back_to_an_open_schema() -> None:
    (spec,) = _openai_tool_defs([ToolRef(name="legacy_tool")])
    assert spec["function"]["parameters"] == {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "scaffold", SHIPPED_SCAFFOLDS, ids=[p.parent.name for p in SHIPPED_SCAFFOLDS]
)
def test_every_shipped_tool_declares_a_schema(scaffold: Path) -> None:
    snapshot = load_harness(scaffold)
    assert snapshot.tools, f"{scaffold}: no tools registered"
    for tool in snapshot.tools:
        params = tool.parameters or {}
        properties = params.get("properties", {})
        assert properties, (
            f"{scaffold}: tool {tool.name!r} declares no parameter schema — "
            "the model will guess argument names, and a measured 50-78% of "
            "calls then arrive as empty objects"
        )
        for name in params.get("required", []):
            assert name in properties, (
                f"{scaffold}: tool {tool.name!r} requires {name!r} but does not declare it"
            )
