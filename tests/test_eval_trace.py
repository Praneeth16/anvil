"""Regression tests for the per-row eval-trace guarantee.

``mlflow.genai.evaluate``'s harness retrieves each row's trace via
``mlflow.get_trace(request_id)`` and stores it as ``eval_item.trace``.
When ``predict_fn`` is supplied there is NO fallback to a minimal
trace, so a row whose ``predict_fn`` yields no span leaves
``eval_item.trace`` None and the harness crashes in
``_get_new_expectations`` reading ``trace.info.assessments``
(``AttributeError: 'NoneType' object has no attribute 'info'``).

The runtime ``predict_fn`` previously relied on
``mlflow.openai.autolog`` to produce that trace from
``chat.completions.create``. That is fragile — on the live backend the
autolog trace was not retrievable by the row's request id, so rows had
no trace and ``make_baseline`` crashed partway through the eval.

``evaluate_branch`` now wraps every ``predict_fn`` invocation in an
explicit root ``CHAIN`` span so a per-row trace always exists. These
tests:

* reproduce the crash with an untraced ``predict_fn`` (the red baseline
  that proves the fix is necessary),
* prove the root-span wrapper yields a per-row trace and nests the
  ``search_knowledge_base`` ``RETRIEVER`` span under it (the
  ``RetrievalGroundedness`` requirement),
* exercise the real ``evaluate_branch`` path end-to-end with an agent
  that produces no span, confirming the wrapper is wired in.

No LLM, no Databricks workspace, no network: ``mlflow.genai.evaluate``
runs against a local file tracking store with a trivial programmatic
scorer.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def local_mlruns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point mlflow at an isolated local file store for the test.

    Real (not mocked) so ``mlflow.genai.evaluate`` can log runs/traces.
    Restores the prior tracking URI on teardown.
    """
    import mlflow

    store = tmp_path / "mlruns"
    old_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"file://{store}")
    yield store
    mlflow.set_tracking_uri(old_uri)


def _gold_row(example_id: str) -> dict:
    return {
        "inputs": {"query": f"q-{example_id}", "category": "direct"},
        "expectations": {
            "expected_facts": ["fact-a"],
            "should_refuse": False,
        },
        "tags": {"example_id": example_id},
    }


def _make_passing_scorer():
    """A trivial programmatic scorer that needs no LLM and always passes."""
    from mlflow.genai.scorers import scorer

    @scorer
    def _passing(inputs, outputs, expectations, trace):
        return 1.0

    return _passing


# ---------------------------------------------------------------------------
# 1. Red baseline — an untraced predict_fn crashes the harness
# ---------------------------------------------------------------------------


def test_untraced_predict_fn_crashes_eval_harness(
    local_mlruns: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``predict_fn`` that yields no span leaves ``eval_item.trace`` None,
    so ``mlflow.genai.evaluate`` crashes reading ``trace.info`` — the exact
    ``make_baseline`` failure. Trace validation is skipped so the harness
    does NOT auto-wrap the predict_fn (mirroring the real flow, where
    autolog produces a span during validation and suppresses the auto-wrap),
    leaving the row with no trace. This is the bug the root-span fix targets.
    """
    import mlflow

    monkeypatch.setenv("MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION", "True")
    mlflow.set_experiment("test_untraced_crash")

    def predict_fn(query, **_kwargs):
        # No span created — mimics a runtime agent whose trace is not
        # retrievable by the row's request id.
        return "answer"

    with pytest.raises(Exception) as exc:
        mlflow.genai.evaluate(
            data=[_gold_row("r1")],
            scorers=[_make_passing_scorer()],
            predict_fn=predict_fn,
        )
    assert "NoneType" in str(exc.value)
    assert "info" in str(exc.value)


# ---------------------------------------------------------------------------
# 2. The fix — a root-span wrapper yields a per-row trace
# ---------------------------------------------------------------------------


def test_predict_fn_root_span_yields_per_row_trace(local_mlruns: Path) -> None:
    """Wrapping the predict_fn body in an explicit root ``CHAIN`` span
    guarantees a per-row trace even when the inner agent produces no span.
    The harness finds the trace (no crash) and it carries the root span."""
    import mlflow
    from mlflow.entities import SpanType

    mlflow.set_experiment("test_root_span_trace")

    def predict_fn(query, **_kwargs):
        # The fix: an explicit root span around a no-span inner call.
        with mlflow.start_span(name="anvil.predict", span_type=SpanType.CHAIN) as span:
            span.set_inputs({"query": query})
            text = "answer"  # inner agent produces no span
            span.set_outputs({"response": text})
            return text

    result = mlflow.genai.evaluate(
        data=[_gold_row("r1"), _gold_row("r2")],
        scorers=[_make_passing_scorer()],
        predict_fn=predict_fn,
    )
    trace_ids = list(result.result_df["trace_id"])
    assert len(trace_ids) == 2
    assert all(tid for tid in trace_ids)

    # Each row's trace is retrievable and carries the root CHAIN span.
    for tid in trace_ids:
        trace = mlflow.get_trace(tid)
        assert trace is not None
        types = [s.span_type for s in trace.data.spans]
        assert "CHAIN" in types
        assert trace.data.spans[0].parent_id is None  # root


# ---------------------------------------------------------------------------
# 3. The RETRIEVER span still nests under the row trace
# ---------------------------------------------------------------------------


def test_predict_fn_root_span_nests_retriever(local_mlruns: Path) -> None:
    """The ``search_knowledge_base`` tool emits a ``RETRIEVER`` span via
    ``mlflow.start_span``. With the root-span wrapper it must nest UNDER
    the root ``CHAIN`` span in the same per-row trace — that is what
    ``RetrievalGroundedness`` reads. Uses the real KB executor over
    ``data/kb`` so the RETRIEVER span is genuine."""
    import mlflow
    from mlflow.entities import SpanType

    from anvil.tools.search_knowledge_base import make_kb_executor

    mlflow.set_experiment("test_root_span_nests_retriever")
    executor = make_kb_executor(REPO_ROOT / "data" / "kb")

    def predict_fn(query, **_kwargs):
        with mlflow.start_span(name="anvil.predict", span_type=SpanType.CHAIN) as span:
            span.set_inputs({"query": query})
            retrieved = executor("search_knowledge_base", json.dumps({"query": query, "k": 3}))
            text = f"answer: {retrieved[:16]}"
            span.set_outputs({"response": text})
            return text

    result = mlflow.genai.evaluate(
        data=[_gold_row("r1")],
        scorers=[_make_passing_scorer()],
        predict_fn=predict_fn,
    )
    tid = result.result_df["trace_id"].iloc[0]
    trace = mlflow.get_trace(tid)
    assert trace is not None

    by_type = {s.span_type: s for s in trace.data.spans}
    assert "CHAIN" in by_type, "root CHAIN span missing"
    assert "RETRIEVER" in by_type, "RETRIEVER span missing — RetrievalGroundedness would break"

    root = by_type["CHAIN"]
    retriever = by_type["RETRIEVER"]
    # The RETRIEVER span nests under the root span (same trace, child).
    assert retriever.parent_id == root.span_id


# ---------------------------------------------------------------------------
# 4. evaluate_branch wiring — the wrapper is wired into the real path
# ---------------------------------------------------------------------------


class _NoSpanAgent:
    """Runtime agent stand-in that produces NO mlflow span.

    Mimics the failing-autolog path where the row's trace would be None
    without ``evaluate_branch``'s root-span wrapper. Returns a response
    shape that ``_extract_final_text`` can walk.
    """

    def predict(self, request):  # noqa: ANN001
        return SimpleNamespace(
            output=[{"type": "message", "content": [{"type": "output_text", "text": "answer"}]}]
        )


def test_evaluate_branch_yields_trace_with_untraced_agent(
    local_mlruns: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end: ``evaluate_branch`` with an agent that produces no span
    still yields a per-row trace (the root-span wrapper), so the real
    ``mlflow.genai.evaluate`` harness does not crash. Trace validation is
    skipped so the harness cannot auto-wrap the predict_fn — making this a
    true regression test: removing the root-span wrapper would reproduce
    the crash (no trace -> ``eval_item.trace`` None)."""
    import mlflow

    from anvil.eval import runner
    from anvil.runtime.models import (
        EvalConfig,
        EvalModeConfig,
        ExperimentsConfig,
        HarnessConfig,
        ScorerConfig,
    )

    monkeypatch.setenv("MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION", "True")

    config = HarnessConfig(
        runtime_endpoint="rt",
        optimizer_endpoint="op",
        judge_endpoint="j",
        experiments=ExperimentsConfig(runtime="r", eval="trace_eval", optimizer="o"),
        eval=EvalConfig(
            default_mode="quick",
            scorers=[ScorerConfig(name="trivial", type="llm", weight=1.0)],
            modes={"quick": EvalModeConfig(rows=2, buckets={"direct": 2})},
            n_workers=2,
        ),
    )
    monkeypatch.setattr(runner, "load_harness", lambda *a, **kw: SimpleNamespace(config=config))
    monkeypatch.setattr(
        runner, "load_golden_set", lambda _p: [_gold("g1", "hello"), _gold("g2", "world")]
    )
    monkeypatch.setattr(runner, "select_subset", lambda exs, **_k: exs)
    monkeypatch.setattr(runner, "make_kb_executor", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(runner, "AnvilAgent", lambda *a, **kw: _NoSpanAgent())
    monkeypatch.setattr(runner, "enable_runtime_tracing", lambda *a, **kw: None)
    # Real scorers would need an LLM judge; inject a trivial programmatic one
    # so mlflow.genai.evaluate runs for real without a network call.
    monkeypatch.setattr(runner, "build_scorers", lambda **_kw: [_make_passing_scorer()])

    report = runner.evaluate_branch(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=tmp_path / "config.yaml",
        golden_set_path="unused",
        kb_dir=tmp_path / "kb",
        runtime_client=SimpleNamespace(),
        judge_client=SimpleNamespace(),
    )

    # No crash, and every row produced a trace.
    assert report.n_rows == 2
    assert len(report.trace_ids) == 2
    assert all(tid for tid in report.trace_ids)

    # The per-row trace carries the root CHAIN span (the wrapper).
    trace = mlflow.get_trace(report.trace_ids[0])
    assert trace is not None
    assert "CHAIN" in [s.span_type for s in trace.data.spans]


def _gold(example_id: str, answer: str) -> dict:
    return {
        "example_id": example_id,
        "query": f"q-{example_id}",
        "category": "direct",
        "expected_doc_ids": [],
        "reference_answer": answer,
        "should_refuse": False,
        "expected_citations": [],
        "must_include": [answer],
        "must_not_include": [],
        "notes_for_judge": "",
    }
