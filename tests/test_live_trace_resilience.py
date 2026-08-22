"""What only a real workspace revealed — and how it is pinned offline.

Every test in this file exists because a live run against
``fe-vm-lakebase-praneeth`` on 2026-08-22 failed in a way the whole offline suite
was blind to. Recorded here so the specific blind spot does not return.

The blind spot has one cause. ``_resilient_eval_harness``'s fallback for a
missing per-row trace is ``create_minimal_trace``, which ends in
``return mlflow.get_trace(root_span.trace_id)`` — i.e. it depends on the very
retrieval it exists to compensate for. Against a **local file store** that
retrieval always succeeds, so the fallback always works and every offline test
passes. Against the **Databricks Tracing Server** it can return ``None``,
repeatedly, and then nothing has been fixed: the crash simply relocates to the
next unguarded dereference of ``eval_item.trace``.

Live, it relocated three times, to exactly the three sites
``_resilient_eval_harness``'s own docstring had named in advance:

1. ``batch_link_traces_to_run`` (``trace_utils.py``:1014) — an unguarded list
   comprehension. Killed the run *after* all 8 predictions and all 24 judge calls
   had been paid for.
2. ``construct_eval_result_df`` (``trace_utils.py``:925) — swallows the
   ``AttributeError`` and returns ``None``, which reaches ``_aggregate_report``
   as ``len(None)`` → ``TypeError``.
3. And separately, on the error path: a prediction failure thrown *into*
   ``mlflow.start_span``'s context manager came out as ``RuntimeError: generator
   didn't stop after throw()``, with the underlying 404 gone.

These tests therefore simulate a ``None`` trace directly rather than hoping a
local store produces one, which is the only way to reach these paths offline.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# 1. A prediction failure must survive as itself
# ---------------------------------------------------------------------------


def test_traced_predict_raises_the_real_error_not_generator_plumbing() -> None:
    """The failure is re-raised *after* the span closes, so it is never thrown
    into ``mlflow.start_span``'s generator.

    Live, one wrong endpoint name in harness/config.yaml surfaced as
    ``RuntimeError: generator didn't stop after throw()`` and nothing else — the
    real ``openai.NotFoundError: 404 RESOURCE_DOES_NOT_EXIST`` was replaced.

    This is a Failure-vs-Error problem, not a cosmetic one: mlflow records
    ``error_message`` from whatever escapes ``predict_fn``, and the shim captures
    it so the row can be excluded and the round guarded. If what escapes is
    generator plumbing, the evidence records that instead of the endpoint
    failure, and whoever debugs the degraded round is sent to the wrong place.
    """
    from anvil.eval.runner import _traced_predict

    sentinel = RuntimeError("404 RESOURCE_DOES_NOT_EXIST: endpoint does not exist")

    def _body() -> str:
        raise sentinel

    with pytest.raises(RuntimeError) as exc:
        _traced_predict({"query": "q"}, _body)

    # The same exception object, not a re-wrapped or replaced one.
    assert exc.value is sentinel
    assert "generator didn't stop" not in str(exc.value)


def test_traced_predict_raises_the_real_error_with_tracing_disabled() -> None:
    """The live masking happened specifically under ``@trace_disabled``, which is
    how mlflow invokes ``predict_fn`` during its pre-flight check — the span
    becomes a no-op whose context manager mishandles ``throw()``. So the
    no-tracing case is the one that has to hold."""
    from mlflow.tracing.provider import trace_disabled

    from anvil.eval.runner import _traced_predict

    sentinel = ValueError("the actual cause")

    @trace_disabled
    def _call() -> str:
        def _body() -> str:
            raise sentinel

        return _traced_predict({"query": "q"}, _body)

    with pytest.raises(ValueError) as exc:
        _call()
    assert exc.value is sentinel


def test_traced_predict_returns_and_records_the_answer() -> None:
    """The happy path is unchanged: the body's return value is what comes back."""
    from anvil.eval.runner import _traced_predict

    assert _traced_predict({"query": "q"}, lambda: "the answer") == "the answer"


# ---------------------------------------------------------------------------
# 2. A row with no trace must not take the whole run down
# ---------------------------------------------------------------------------


def _eval_result(trace_id: str | None):
    """An mlflow eval_result stand-in whose trace is present or None."""
    trace = (
        SimpleNamespace(info=SimpleNamespace(trace_id=trace_id)) if trace_id is not None else None
    )
    return SimpleNamespace(eval_item=SimpleNamespace(trace=trace, request_id=f"req-{trace_id}"))


def test_batch_link_skips_rows_with_no_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """``trace_utils.py``:1014 is an unguarded
    ``[r.eval_item.trace.info.trace_id for r in eval_results]``.

    Live, one traceless row out of eight killed the run *there* — after every
    prediction and every judge call had already been paid for. Losing one row
    from the frame is worth vastly more than losing all eight.
    """
    import mlflow.genai.evaluation.harness as harness

    from anvil.eval.runner import _resilient_eval_harness

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        harness,
        "batch_link_traces_to_run",
        lambda run_id, eval_results: seen.update(linked=[r for r in eval_results]),
    )

    results = [_eval_result("t0"), _eval_result(None), _eval_result("t2")]
    with _resilient_eval_harness():
        harness.batch_link_traces_to_run(run_id="r", eval_results=results)

    linked = seen["linked"]
    assert len(linked) == 2, "the traceless row is dropped, the others are linked"
    assert all(r.eval_item.trace is not None for r in linked)


def test_batch_link_does_nothing_when_no_row_has_a_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With nothing linkable the original is not called at all, rather than
    called with an empty list whose behaviour is mlflow's business."""
    import mlflow.genai.evaluation.harness as harness

    from anvil.eval.runner import _resilient_eval_harness

    calls: list[int] = []
    monkeypatch.setattr(
        harness,
        "batch_link_traces_to_run",
        lambda run_id, eval_results: calls.append(len(eval_results)),
    )

    with _resilient_eval_harness():
        harness.batch_link_traces_to_run(run_id="r", eval_results=[_eval_result(None)])

    assert calls == []


def test_result_frame_construction_skips_rows_with_no_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``construct_eval_result_df`` derefs the trace inside a ``try`` that
    swallows the error and returns ``None``. That ``None`` then reaches
    ``_aggregate_report`` as ``len(None)``.

    Guarding the link site alone merely moved the live crash here — which is what
    happened: "Evaluation completed", the run logged to MLflow, then TypeError.
    """
    import mlflow.genai.evaluation.harness as harness

    from anvil.eval.runner import _resilient_eval_harness

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        harness,
        "construct_eval_result_df",
        lambda run_id, traces, eval_results: seen.update(rows=list(eval_results)),
    )

    results = [_eval_result("t0"), _eval_result(None)]
    with _resilient_eval_harness():
        harness.construct_eval_result_df("r", [], results)

    assert len(seen["rows"]) == 1


def test_the_shim_restores_every_symbol_it_patches() -> None:
    """Four harness symbols are patched now. A leaked patch would silently change
    behaviour for every later ``mlflow.genai.evaluate`` call in the process,
    including ones anvil does not own."""
    import mlflow.genai.evaluation.harness as harness

    from anvil.eval.runner import _resilient_eval_harness

    names = (
        "_get_new_expectations",
        "_run_predict",
        "batch_link_traces_to_run",
        "construct_eval_result_df",
    )
    before = {n: getattr(harness, n) for n in names}
    with _resilient_eval_harness():
        assert all(getattr(harness, n) is not before[n] for n in names), "all four patched"
    assert {n: getattr(harness, n) for n in names} == before


# ---------------------------------------------------------------------------
# 3. A dropped row must not shrink the sample invisibly
# ---------------------------------------------------------------------------


def _report(*, n_rows: int, n_errors: int = 0, n_dropped: int = 0):
    from anvil.eval.runner import EvalReport

    return EvalReport(
        aggregate=1.0,
        per_judge={"correctness": 1.0},
        per_bucket={},
        failures=[],
        run_id="r",
        experiment_id="e",
        n_rows=n_rows,
        mode="quick",
        scorers=["correctness"],
        evaluated_at="2026-08-22T12:00:00+00:00",
        n_errors=n_errors,
        n_dropped_rows=n_dropped,
    )


def test_dropped_rows_cannot_lower_the_floor_they_should_trip() -> None:
    """The trap this fix nearly walked into twice.

    A dropped row is not an *error* — the prediction succeeded — so
    ``error_rate`` never sees it, and it is absent from ``n_rows``. With the floor
    capped at ``n_rows``, losing six of eight rows leaves ``n_rows=2``,
    ``error_rate=0.0``, and a floor that shrank to 2 along with the sample: no
    guard fires, and a two-row aggregate is compared to the frontier.

    Capping against ``n_attempted`` instead is what makes the floor bite. This is
    the same shape of silent sample loss that excluding errored rows introduced
    in the first place.
    """
    from anvil.eval.judgeability import unjudgeable_reason

    survived_two_of_eight = _report(n_rows=2, n_dropped=6)
    assert survived_two_of_eight.error_rate == 0.0, "no row errored; only vanished"
    assert survived_two_of_eight.n_attempted == 8

    reason = unjudgeable_reason(survived_two_of_eight, min_scorable_rows=4)
    assert reason != "", "row loss must not pass the sample-size floor"
    assert "dropped for want of a trace" in reason


def test_a_single_dropped_row_is_still_judgeable() -> None:
    """Seven of eight assessed is a measurement. Refusing here would make the
    guard fire on the ordinary live flakiness it was built to absorb."""
    from anvil.eval.judgeability import unjudgeable_reason

    assert unjudgeable_reason(_report(n_rows=7, n_dropped=1), min_scorable_rows=4) == ""


def test_trace_fallback_retries_then_records_the_row_as_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback retries, and when it still fails the row is *counted*.

    A single ``create_minimal_trace`` call is not a fallback on the live tracing
    server — the live run logged "no retrievable trace after 3 fallback attempts",
    so the None is persistent rather than a transient lag. Retrying is what makes
    the fallback worth the name; counting the failure is what stops the resulting
    row loss from being invisible.
    """
    import mlflow.genai.evaluation.harness as harness
    from mlflow.genai.utils import trace_utils

    from anvil.eval import runner
    from anvil.eval.runner import _resilient_eval_harness

    monkeypatch.setattr(runner, "_TRACE_FALLBACK_BACKOFF_S", 0.0)
    # The original leaves the trace None, as the live backend did.
    monkeypatch.setattr(harness, "_run_predict", lambda *a, **kw: None)

    calls = {"n": 0}

    def _always_none(_eval_item):
        calls["n"] += 1
        return None

    monkeypatch.setattr(trace_utils, "create_minimal_trace", _always_none)

    item = SimpleNamespace(trace=None, request_id="req-x", error_message=None)
    dropped: set[str] = set()
    with _resilient_eval_harness(dropped_sink=dropped):
        harness._run_predict(item, object(), None, None)

    assert calls["n"] == runner._TRACE_FALLBACK_ATTEMPTS, "retried, not attempted once"
    assert dropped == {"req-x"}, "the unrepresentable row is counted, not just logged"


def test_evaluate_branch_refuses_an_empty_result_frame(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When every row loses its trace, mlflow builds no frame at all. Raising a
    named error beats letting ``len(None)`` surface as a TypeError three frames
    deeper, which is how this presented live."""
    from anvil.eval import runner
    from anvil.runtime.models import (
        EvalConfig,
        EvalModeConfig,
        ExperimentsConfig,
        HarnessConfig,
        ScorerConfig,
    )

    config = HarnessConfig(
        runtime_endpoint="rt",
        optimizer_endpoint="op",
        judge_endpoint="j",
        experiments=ExperimentsConfig(runtime="r", eval="e", optimizer="o"),
        eval=EvalConfig(
            default_mode="quick",
            scorers=[ScorerConfig(name="correctness")],
            modes={"quick": EvalModeConfig(rows=2, buckets={"direct": 2})},
        ),
    )
    gold = {
        "example_id": "g1",
        "query": "q",
        "category": "direct",
        "expected_doc_ids": [],
        "reference_answer": "a",
        "should_refuse": False,
        "expected_citations": [],
        "must_include": ["a"],
        "must_not_include": [],
        "notes_for_judge": "",
    }
    monkeypatch.setattr(runner, "load_harness", lambda *a, **kw: SimpleNamespace(config=config))
    monkeypatch.setattr(runner, "load_golden_set", lambda _p: [gold])
    monkeypatch.setattr(runner, "select_subset", lambda exs, **_k: exs)
    monkeypatch.setattr(runner, "make_kb_executor", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(runner, "AnvilAgent", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(runner, "enable_runtime_tracing", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "build_scorers", lambda **_kw: [])
    monkeypatch.setattr(runner.mlflow, "set_experiment", lambda *a, **kw: None)
    monkeypatch.setattr(
        runner.mlflow.genai,
        "evaluate",
        lambda **_kw: SimpleNamespace(result_df=None, metrics={}, run_id="run-1"),
    )

    with pytest.raises(RuntimeError, match="no result frame"):
        runner.evaluate_branch(
            scaffold_root=tmp_path / "scaffold",
            runtime_config_path=tmp_path / "config.yaml",
            runtime_client=SimpleNamespace(),
            judge_client=SimpleNamespace(),
        )


def test_evaluate_branch_scopes_the_validation_skip(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mlflow's pre-flight check is skipped during the call and the prior value
    restored after.

    Skipping it is deliberate: the check is what masks a prediction failure as
    ``generator didn't stop after throw()`` and then aborts the run, so the row
    never becomes an error record and none of the Phase 2 guards engage. But the
    override must not leak — it changes how mlflow behaves for anything else in
    the process.
    """
    import os

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
        "expected_doc_ids": [],
        "reference_answer": "a",
        "should_refuse": False,
        "expected_citations": [],
        "must_include": ["a"],
        "must_not_include": [],
        "notes_for_judge": "",
    }
    monkeypatch.setattr(runner, "load_harness", lambda *a, **kw: SimpleNamespace(config=config))
    monkeypatch.setattr(runner, "load_golden_set", lambda _p: [gold])
    monkeypatch.setattr(runner, "select_subset", lambda exs, **_k: exs)
    monkeypatch.setattr(runner, "make_kb_executor", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(runner, "AnvilAgent", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(runner, "enable_runtime_tracing", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "build_scorers", lambda **_kw: [])
    monkeypatch.setattr(runner.mlflow, "set_experiment", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "get_experiment_by_name", lambda *a, **kw: None)

    env = runner._MLFLOW_SKIP_VALIDATION_ENV
    monkeypatch.setenv(env, "prior-value")
    captured: dict[str, str | None] = {}

    def _evaluate(**_kw):
        captured["during"] = os.environ.get(env)
        return SimpleNamespace(
            result_df=pd.DataFrame({"correctness/value": [1.0], "trace_id": ["t0"]}),
            metrics={},
            run_id="run-1",
        )

    monkeypatch.setattr(runner.mlflow.genai, "evaluate", _evaluate)

    runner.evaluate_branch(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=tmp_path / "config.yaml",
        runtime_client=SimpleNamespace(),
        judge_client=SimpleNamespace(),
    )

    assert captured["during"] == "True"
    assert os.environ.get(env) == "prior-value", "the override must not leak"


def test_n_attempted_counts_rows_the_run_actually_tried() -> None:
    from anvil.eval.runner import EvalReport

    assert _report(n_rows=8).n_attempted == 8
    assert _report(n_rows=5, n_dropped=3).n_attempted == 8
    assert EvalReport(
        aggregate=0.0,
        per_judge={},
        per_bucket={},
        failures=[],
        run_id="r",
        experiment_id="e",
        n_rows=0,
        mode="quick",
        scorers=[],
        evaluated_at="x",
    ).n_attempted == 0
