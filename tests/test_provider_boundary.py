"""The provider boundary, the trace source tag, and the code-mode constructor.

Three pieces of type debt were silenced module-wide rather than fixed, and each
had a runtime consequence the annotations were hiding:

* every annotation said ``openai.OpenAI`` while the object passed was always a
  ``GatewayClient``, so the checker was switched off at the one boundary where a
  swapped client matters;
* ``source`` tags every trace and is what observability queries filter on, but
  the constants were bare ``str`` against a ``Literal`` parameter, so a typo
  produced traces nothing matched;
* code mode passed the *unresolved* client parameter, so a round with no
  explicitly injected client handed every ``MemorySystem`` ``llm_client=None``.

The last one is a behaviour bug, not a typing nicety, and it is the reason these
tests exist as call-through tests rather than assertions about annotations.
"""

from __future__ import annotations

import typing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from anvil.observability import (
    SOURCE_EVAL,
    SOURCE_OPTIMIZER,
    SOURCE_PRODUCTION,
    SourceTag,
)
from anvil.runtime.client import ChatClient, GatewayClient


@pytest.mark.unit
def test_gateway_client_satisfies_the_chat_client_protocol() -> None:
    """The concrete client must keep the surface every plane depends on."""
    client = GatewayClient(base_url="http://gateway.invalid", token_fn=lambda: "t")
    assert isinstance(client, ChatClient)


@pytest.mark.unit
def test_protocol_rejects_an_object_without_the_surface() -> None:
    """Guards the check above from passing vacuously."""
    assert not isinstance(SimpleNamespace(), ChatClient)


@pytest.mark.unit
def test_source_constants_are_exactly_the_literal_values() -> None:
    """A constant drifting from the Literal is a trace nothing queries.

    Both directions: every constant must be an accepted tag, and every accepted
    tag must have a constant, so adding one to the ``Literal`` without exporting
    it is caught too.
    """
    allowed = set(typing.get_args(SourceTag))
    constants = {SOURCE_PRODUCTION, SOURCE_EVAL, SOURCE_OPTIMIZER}
    assert constants == allowed


def _code_mode_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    runtime_client: object | None,
) -> dict[str, Any]:
    """Run ``evaluate_branch`` in code mode, capturing the agent's kwargs."""
    import pandas as pd

    from anvil.eval import runner
    from anvil.runtime.models import (
        EvalConfig,
        EvalModeConfig,
        ExperimentsConfig,
        HarnessConfig,
        ScorerConfig,
    )

    config = HarnessConfig(
        mode="code",
        agent_module="anvil.agents.baseline",
        runtime_endpoint="rt",
        optimizer_endpoint="op",
        judge_endpoint="j",
        experiments=ExperimentsConfig(runtime="r", eval="e", optimizer="o"),
        eval=EvalConfig(
            default_mode="quick",
            scorers=[ScorerConfig(name="correctness")],
            modes={"quick": EvalModeConfig(rows=1, buckets={"direct": 1})},
        ),
    )
    gold = {
        "example_id": "g1",
        "query": "q",
        "category": "direct",
        "expected_doc_ids": ["d"],
        "reference_answer": "a",
        "should_refuse": False,
        "expected_citations": ["d"],
        "must_include": ["a"],
        "must_not_include": [],
        "notes_for_judge": "",
    }

    sentinel = SimpleNamespace(marker="built-by-the-factory")
    captured: dict[str, Any] = {}

    def _capture_load(module_path: str, **kwargs: Any) -> object:
        captured["module_path"] = module_path
        captured.update(kwargs)
        return SimpleNamespace(predict=lambda _q: ("answer", {}))

    monkeypatch.setattr(runner, "load_harness", lambda *a, **kw: SimpleNamespace(config=config))
    monkeypatch.setattr(runner, "load_golden_set", lambda _p: [gold])
    monkeypatch.setattr(runner, "select_subset", lambda exs, **_k: exs)
    monkeypatch.setattr(runner, "_load_memory_system", _capture_load)
    monkeypatch.setattr(runner, "build_gateway_client", lambda *a, **kw: sentinel)
    monkeypatch.setattr(runner, "build_scorers", lambda **_k: [])
    monkeypatch.setattr(runner, "enable_runtime_tracing", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "set_experiment", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "set_tracking_uri", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "get_experiment_by_name", lambda *a, **kw: None)
    monkeypatch.setattr(
        runner.mlflow.genai,
        "evaluate",
        lambda **_k: SimpleNamespace(
            result_df=pd.DataFrame({"correctness/value": [1.0], "trace_id": ["t0"]}),
            metrics={},
            run_id="run-1",
        ),
    )

    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "d.md").write_text("---\ndoc_id: d\ntitle: D\n---\nbody\n", encoding="utf-8")
    golden = tmp_path / "golden.jsonl"
    golden.write_text("{}\n", encoding="utf-8")

    runner.evaluate_branch(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=tmp_path / "config.yaml",
        kb_dir=kb,
        golden_set_path=golden,
        runtime_client=runtime_client,
        judge_client=SimpleNamespace(),
    )
    captured["_sentinel"] = sentinel
    return captured


@pytest.mark.unit
def test_code_mode_gets_the_resolved_client_not_the_bare_parameter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no client injected, code mode must still get a real one.

    Prompt mode resolved ``runtime_client or build_gateway_client()`` and code
    mode passed the parameter through, so an ordinary code-mode round -- nobody
    injects a client outside tests -- constructed every candidate with
    ``llm_client=None``. ``BaselineExtractor`` treats that as "echo the input",
    so the eval scored a passthrough rather than the agent, and a candidate that
    actually calls the LLM died on an attribute of None inside the eval, where
    judgeability reads it as infrastructure and aborts the round.
    """
    captured = _code_mode_harness(monkeypatch, tmp_path, runtime_client=None)
    assert captured["llm_client"] is captured["_sentinel"]


@pytest.mark.unit
def test_code_mode_still_honours_an_injected_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards the fix from becoming "always use the factory"."""
    injected = SimpleNamespace(marker="injected")
    captured = _code_mode_harness(monkeypatch, tmp_path, runtime_client=injected)
    assert captured["llm_client"] is injected
