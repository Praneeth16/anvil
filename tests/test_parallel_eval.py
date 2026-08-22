"""Tests for parallel eval (Phase 4 task 4.4).

Covers the acceptance contract:

* :func:`_run_predictions_parallel` runs ``predict_fn`` across queries:
  sequential when ``n_workers <= 1`` (backward compatible), parallel via
  :class:`ThreadPoolExecutor` otherwise.
* Results preserve input order regardless of completion order.
* A prediction that raises is recorded as a :class:`CaseRecord` whose
  outcome is ``error`` and logged — one bad row does not abort the whole
  eval, and the row is *excluded* from scoring rather than scored as an
  empty (i.e. very bad) answer. The empty-string contract these tests
  used to assert was the Phase 2 defect; see
  ``docs/design/failure-vs-error.md``.
* :func:`evaluate_branch` wires ``eval.n_workers`` from
  ``harness/config.yaml`` into the ``MLFLOW_GENAI_EVAL_MAX_WORKERS`` env
  var so the configured value controls mlflow's predict/score thread
  pool, while still passing ``predict_fn`` (so per-row ``RETRIEVER``
  traces are preserved for ``RetrievalGroundedness``). The env-var
  override is scoped to the ``mlflow.genai.evaluate`` call.

No LLM calls and no Databricks calls are made — ``mlflow.genai.evaluate``
and the runtime agent are mocked.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def _patch_runner_common(
    monkeypatch: pytest.MonkeyPatch,
    config,
) -> None:
    """Patch the runner's external dependencies for a mocked eval run.

    Mirrors the harness in ``test_code_mode.py`` so ``evaluate_branch``
    can run without an LLM, a knowledge base, or a live Databricks
    workspace.
    """
    from anvil.eval import runner

    monkeypatch.setattr(runner, "load_harness", lambda *a, **kw: SimpleNamespace(config=config))
    monkeypatch.setattr(
        runner, "load_golden_set", lambda _p: [_gold("g1", "hello"), _gold("g2", "world")]
    )
    monkeypatch.setattr(runner, "select_subset", lambda exs, **_k: exs)
    monkeypatch.setattr(runner, "make_kb_executor", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(runner, "AnvilAgent", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(runner, "enable_runtime_tracing", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "set_experiment", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "set_tracking_uri", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "get_experiment_by_name", lambda *a, **kw: None)


def _result_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "correctness/value": [1.0, 1.0],
            "trace_id": ["t0", "t1"],
        }
    )


def _outputs(records) -> list[str]:
    return [r.output for r in records]


def _outcomes(records) -> list[str]:
    return [str(r.outcome) for r in records]


# ---------------------------------------------------------------------------
# 1. _run_predictions_parallel — sequential mode (n_workers <= 1)
# ---------------------------------------------------------------------------


def test_run_predictions_sequential_default() -> None:
    """With the default n_workers=1, predictions run sequentially and
    in input order — the backward-compatible path."""
    from anvil.eval.runner import _run_predictions_parallel

    calls: list[str] = []

    def predict_fn(q: str) -> str:
        calls.append(q)
        return f"r-{q}"

    queries = ["a", "b", "c"]
    out = _run_predictions_parallel(predict_fn, queries)
    assert _outputs(out) == ["r-a", "r-b", "r-c"]
    assert _outcomes(out) == ["ok", "ok", "ok"]
    # Called exactly once per query, in input order.
    assert calls == ["a", "b", "c"]


def test_run_predictions_sequential_n_zero_and_negative() -> None:
    """n_workers of 0 or a negative value collapses to the sequential
    path (the ``<= 1`` guard), never to a zero/negative-sized pool."""
    from anvil.eval.runner import _run_predictions_parallel

    def predict_fn(q: str) -> str:
        return q.upper()

    for n in (0, -1, -5):
        out = _run_predictions_parallel(predict_fn, ["a", "b"], n_workers=n)
        assert _outputs(out) == ["A", "B"]


def test_run_predictions_empty_queries() -> None:
    """An empty query list yields an empty result list (no pool work)."""
    from anvil.eval.runner import _run_predictions_parallel

    assert _run_predictions_parallel(lambda q: q, [], n_workers=4) == []
    assert _run_predictions_parallel(lambda q: q, []) == []


def test_run_predictions_case_ids_default_to_row_index() -> None:
    """Every record carries a case_id. Absent explicit ids the row index
    is used, so a record is always attributable to a row."""
    from anvil.eval.runner import _run_predictions_parallel

    out = _run_predictions_parallel(lambda q: q, ["a", "b"], n_workers=2)
    assert [r.case_id for r in out] == ["0", "1"]

    out = _run_predictions_parallel(
        lambda q: q, ["a", "b"], n_workers=2, case_ids=["g1", "g2"]
    )
    assert [r.case_id for r in out] == ["g1", "g2"]


def test_run_predictions_rejects_mismatched_case_ids() -> None:
    """A case_ids list that does not line up with queries is a caller bug
    that would silently mis-attribute every record — reject it."""
    from anvil.eval.runner import _run_predictions_parallel

    with pytest.raises(ValueError, match="case_ids"):
        _run_predictions_parallel(lambda q: q, ["a", "b"], case_ids=["only-one"])


# ---------------------------------------------------------------------------
# 2. _run_predictions_parallel — parallel mode (n_workers > 1)
# ---------------------------------------------------------------------------


def test_run_predictions_parallel_preserves_order() -> None:
    """Parallel execution returns results in INPUT order even when the
    futures complete in a different order. Earlier queries sleep longer
    so they finish last; the result list must still match the input."""
    from anvil.eval.runner import _run_predictions_parallel

    n = 6
    queries = [f"q{i}" for i in range(n)]
    completion_order: list[int] = []

    def predict_fn(q: str) -> str:
        i = int(q[1:])
        # q0 sleeps the longest → finishes last; q(n-1) finishes first.
        time.sleep(0.02 * (n - i))
        completion_order.append(i)
        return f"r{i}"

    out = _run_predictions_parallel(predict_fn, queries, n_workers=4)

    # Results are in input order regardless of completion order.
    assert _outputs(out) == [f"r{i}" for i in range(n)]
    # And parallelism actually happened: completion order is NOT the
    # input order (the longest-sleeping q0 finished after the quick ones).
    assert completion_order != list(range(n))


def test_run_predictions_parallel_more_workers_than_queries() -> None:
    """A pool larger than the query count still produces every result in
    order (excess workers simply idle)."""
    from anvil.eval.runner import _run_predictions_parallel

    out = _run_predictions_parallel(lambda q: f"x{q}", ["a", "b"], n_workers=16)
    assert _outputs(out) == ["xa", "xb"]


def test_run_predictions_parallel_single_worker_equivalent_to_sequential() -> None:
    """n_workers=1 takes the sequential branch (no executor), so the
    output is identical to a plain list comprehension."""
    from anvil.eval.runner import _run_predictions_parallel

    queries = [f"q{i}" for i in range(5)]
    expected = [f"r{q}" for q in queries]
    out = _run_predictions_parallel(lambda q: f"r{q}", queries, n_workers=1)
    assert _outputs(out) == expected


# ---------------------------------------------------------------------------
# 3. _run_predictions_parallel — errors are errors, not zero-scored answers
# ---------------------------------------------------------------------------


def test_run_predictions_failed_row_is_an_error_record_not_empty_output(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A prediction that raises yields an ``error`` record naming the
    exception, and the surrounding rows still complete.

    This is the Phase 2 correction. The old contract recorded ``""`` for a
    raised prediction, which the judges score at or near 0.0 — so a
    throttled gateway was indistinguishable from a bad answer and moved
    the promotion gate the same way. The record's outcome now says the
    case was never assessed, and ``scorable`` is False so nothing
    downstream averages it in.
    """
    from anvil.eval.runner import _run_predictions_parallel

    def predict_fn(q: str) -> str:
        if q == "boom":
            raise RuntimeError("synthetic failure")
        return f"ok-{q}"

    with caplog.at_level(logging.WARNING, logger="anvil.eval.runner"):
        out = _run_predictions_parallel(predict_fn, ["a", "boom", "c"], n_workers=3)

    assert _outcomes(out) == ["ok", "error", "ok"]
    assert _outputs(out) == ["ok-a", "", "ok-c"]
    assert [r.scorable for r in out] == [True, False, True]

    errored = out[1]
    assert errored.error_type == "RuntimeError"
    assert "synthetic failure" in errored.error_message
    assert "prediction failed for row 1" in caplog.text
    assert "synthetic failure" in caplog.text


def test_run_predictions_all_fail_are_all_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every row failing yields a full-length list of ``error`` records —
    the eval is not aborted by a uniformly broken agent, and no row
    contributes a zero to the aggregate."""
    from anvil.eval.runner import _run_predictions_parallel

    def predict_fn(q: str) -> str:
        raise ValueError("nope")

    with caplog.at_level(logging.WARNING, logger="anvil.eval.runner"):
        out = _run_predictions_parallel(predict_fn, ["a", "b", "c"], n_workers=2)
    assert _outcomes(out) == ["error", "error", "error"]
    assert not any(r.scorable for r in out)
    assert caplog.text.count("prediction failed") == 3


def test_run_predictions_sequential_failure_isolated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The sequential path (n_workers <= 1) isolates failures exactly as
    the parallel path does, so the contract holds uniformly across both."""
    from anvil.eval.runner import _run_predictions_parallel

    def predict_fn(q: str) -> str:
        if q == "boom":
            raise RuntimeError("synthetic failure")
        return q

    with caplog.at_level(logging.WARNING, logger="anvil.eval.runner"):
        out = _run_predictions_parallel(predict_fn, ["a", "boom", "c"], n_workers=1)
    assert _outcomes(out) == ["ok", "error", "ok"]
    assert _outputs(out) == ["a", "", "c"]
    assert "prediction failed for row 1" in caplog.text


def test_run_predictions_summarizes_to_an_error_rate() -> None:
    """The records summarise to the error rate the round-level guard reads."""
    from anvil.eval.outcome import summarize
    from anvil.eval.runner import _run_predictions_parallel

    def predict_fn(q: str) -> str:
        if q.startswith("bad"):
            raise TimeoutError("gateway timeout")
        return q

    out = _run_predictions_parallel(
        predict_fn, ["a", "bad1", "b", "bad2"], n_workers=2
    )
    summary = summarize(out)
    assert summary.total == 4
    assert summary.error == 2
    assert summary.error_rate == 0.5
    assert summary.scorable == 2


# ---------------------------------------------------------------------------
# 3b. Retries — on error only, never on a returned answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_workers", [1, 3])
def test_run_predictions_retries_an_error_and_keeps_the_attempts(n_workers: int) -> None:
    """A transient error is retried up to ``max_retries`` times, and the
    failed attempts are retained on the record.

    A round that only succeeded on its third try is not the same round as
    one that succeeded immediately — the difference is a degrading
    endpoint, and the evidence should say so rather than leaving it to be
    inferred from a latency graph.
    """
    from anvil.eval.runner import _run_predictions_parallel

    calls: dict[str, int] = {}

    def predict_fn(q: str) -> str:
        calls[q] = calls.get(q, 0) + 1
        if calls[q] < 3:
            raise ConnectionError(f"attempt {calls[q]} refused")
        return f"ok-{q}"

    out = _run_predictions_parallel(
        predict_fn, ["a"], n_workers=n_workers, max_retries=2
    )
    assert _outcomes(out) == ["ok"]
    assert out[0].output == "ok-a"
    # Two failed attempts precede the success and are kept.
    assert len(out[0].attempts) == 2
    assert [a.error_type for a in out[0].attempts] == ["ConnectionError"] * 2
    assert calls["a"] == 3


def test_run_predictions_gives_up_after_max_retries() -> None:
    """Exhausted retries land as one ``error`` record carrying every
    attempt, not as a partial success."""
    from anvil.eval.runner import _run_predictions_parallel

    def predict_fn(q: str) -> str:
        raise ConnectionError("always refused")

    out = _run_predictions_parallel(predict_fn, ["a"], max_retries=2)
    assert _outcomes(out) == ["error"]
    assert out[0].error_type == "ConnectionError"
    # 1 initial + 2 retries, all recorded.
    assert len(out[0].attempts) == 3


def test_run_predictions_never_retries_a_returned_answer() -> None:
    """Retry is for errors only. A returned answer — even an empty one —
    is a *failure* at worst, which is signal about the agent and must not
    be re-rolled until it improves. Re-rolling failures is how an
    optimizer's score becomes a function of how many samples it bought."""
    from anvil.eval.runner import _run_predictions_parallel

    calls: list[str] = []

    def predict_fn(q: str) -> str:
        calls.append(q)
        return ""  # a bad answer, not an error

    out = _run_predictions_parallel(predict_fn, ["a"], max_retries=5)
    assert calls == ["a"]
    assert _outcomes(out) == ["ok"]
    assert out[0].output == ""
    assert out[0].attempts == ()


# ---------------------------------------------------------------------------
# 3c. Interruption — partial results stay readable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_workers", [1, 2])
def test_run_predictions_interrupt_accounts_for_every_row(n_workers: int) -> None:
    """Ctrl-C stops the run and raises :class:`RunInterrupted` carrying a
    record for every row, so a killed run is still readable.

    Which rows finished is timing-dependent under a thread pool, so this
    asserts the invariant that holds either way: every row is accounted
    for, and no row is misreported as an infrastructure ``error`` — an
    operator's Ctrl-C is not a degraded endpoint, and recording it as one
    would trip the round's error-rate guard.
    """
    from anvil.eval.outcome import RunInterrupted
    from anvil.eval.runner import _run_predictions_parallel

    def predict_fn(q: str) -> str:
        if q == "stop":
            raise KeyboardInterrupt
        time.sleep(0.1)
        return f"ok-{q}"

    # Far more queries than the pool can drain before the main thread reacts,
    # so "some row was never reached" is not a race: two workers need ten
    # 0.1s rounds to finish this list and the cancel happens in microseconds.
    # A tighter list would pass on a quiet laptop and flake on a loaded runner.
    queries = ["a", "stop", *[f"q{i}" for i in range(18)]]
    with pytest.raises(RunInterrupted) as exc:
        _run_predictions_parallel(predict_fn, queries, n_workers=n_workers)

    records = exc.value.records
    assert len(records) == len(queries)
    assert {str(r.outcome) for r in records} <= {"ok", "interrupted"}
    assert any(r.outcome == "interrupted" for r in records)
    # RunInterrupted derives from BaseException, like KeyboardInterrupt, so
    # the ``except Exception`` handlers between here and the CLI cannot
    # swallow a Ctrl-C and report it as an eval failure.
    assert not isinstance(exc.value, Exception)


def test_run_predictions_interrupt_marks_the_unreached_rows_sequentially() -> None:
    """The sequential path pins down the exact marking the parallel path
    can only be asserted loosely about: rows before the interrupt keep
    their result, the interrupted row and every row after it are
    ``interrupted`` — never ``error``, and never silently absent."""
    from anvil.eval.outcome import RunInterrupted
    from anvil.eval.runner import _run_predictions_parallel

    def predict_fn(q: str) -> str:
        if q == "stop":
            raise KeyboardInterrupt
        return f"ok-{q}"

    with pytest.raises(RunInterrupted) as exc:
        _run_predictions_parallel(predict_fn, ["a", "stop", "c", "d"], n_workers=1)

    records = exc.value.records
    assert _outcomes(records) == ["ok", "interrupted", "interrupted", "interrupted"]
    assert records[0].output == "ok-a"


# ---------------------------------------------------------------------------
# 4. Config model — EvalConfig.n_workers default + real config
# ---------------------------------------------------------------------------


def test_eval_config_n_workers_default() -> None:
    """EvalConfig defaults n_workers to 4 (the shipped config value)."""
    from anvil.runtime.models import EvalConfig

    cfg = EvalConfig()
    assert cfg.n_workers == 4


def test_real_config_has_n_workers() -> None:
    """The repo's harness/config.yaml carries eval.n_workers so the
    wiring has a real source to read from."""
    import yaml

    from anvil.runtime.models import RuntimeYAML

    config_path = REPO_ROOT / "harness" / "config.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "n_workers" in raw["eval"]
    cfg = RuntimeYAML.model_validate(raw)
    assert cfg.eval.n_workers == 4


# ---------------------------------------------------------------------------
# 5. evaluate_branch wiring — n_workers -> MLFLOW_GENAI_EVAL_MAX_WORKERS
# ---------------------------------------------------------------------------


def _wiring_config(n_workers: int):
    from anvil.runtime.models import (
        EvalConfig,
        EvalModeConfig,
        ExperimentsConfig,
        HarnessConfig,
        ScorerConfig,
    )

    return HarnessConfig(
        runtime_endpoint="rt",
        optimizer_endpoint="op",
        judge_endpoint="j",
        experiments=ExperimentsConfig(runtime="r", eval="e", optimizer="o"),
        eval=EvalConfig(
            default_mode="quick",
            scorers=[ScorerConfig(name="correctness", type="llm", weight=1.0)],
            modes={"quick": EvalModeConfig(rows=2, buckets={"direct": 2})},
            n_workers=n_workers,
        ),
    )


def test_evaluate_branch_sets_max_workers_from_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """evaluate_branch sets MLFLOW_GENAI_EVAL_MAX_WORKERS to cfg.n_workers
    for the duration of the mlflow.genai.evaluate call, then restores
    the prior value (here: unset before → unset after)."""
    from anvil.eval import runner

    _patch_runner_common(monkeypatch, _wiring_config(n_workers=3))

    # Clean baseline: the env var is unset before the call.
    monkeypatch.delenv("MLFLOW_GENAI_EVAL_MAX_WORKERS", raising=False)

    captured: dict[str, str | None] = {}

    def fake_evaluate(**kwargs: object) -> object:
        # The env var must be set to the configured n_workers DURING the
        # call (the try-block sets it; finally restores it after).
        captured["workers"] = os.environ.get("MLFLOW_GENAI_EVAL_MAX_WORKERS")
        captured["predict_fn_passed"] = kwargs.get("predict_fn") is not None
        return SimpleNamespace(result_df=_result_df(), metrics={}, run_id="run-1")

    monkeypatch.setattr(runner.mlflow.genai, "evaluate", fake_evaluate)

    runner.evaluate_branch(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=tmp_path / "config.yaml",
        runtime_client=SimpleNamespace(),
        judge_client=SimpleNamespace(),
    )

    assert captured["workers"] == "3"
    # predict_fn is still passed (trace-preserving path), not pre-computed.
    assert captured["predict_fn_passed"] is True
    # Restored: unset before → unset after.
    assert "MLFLOW_GENAI_EVAL_MAX_WORKERS" not in os.environ


def test_evaluate_branch_restores_prior_max_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the env var was already set before the call, evaluate_branch
    restores that prior value rather than deleting it."""
    from anvil.eval import runner

    _patch_runner_common(monkeypatch, _wiring_config(n_workers=8))
    monkeypatch.setenv("MLFLOW_GENAI_EVAL_MAX_WORKERS", "2")

    monkeypatch.setattr(
        runner.mlflow.genai,
        "evaluate",
        lambda **kw: SimpleNamespace(result_df=_result_df(), metrics={}, run_id="run-1"),
    )

    runner.evaluate_branch(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=tmp_path / "config.yaml",
        runtime_client=SimpleNamespace(),
        judge_client=SimpleNamespace(),
    )

    # The pre-call value "2" is restored, not the override "8".
    assert os.environ.get("MLFLOW_GENAI_EVAL_MAX_WORKERS") == "2"
    monkeypatch.delenv("MLFLOW_GENAI_EVAL_MAX_WORKERS", raising=False)


@pytest.mark.parametrize("prior", [None, "unexpected-value"])
def test_evaluate_branch_scopes_synchronous_trace_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prior: str | None
) -> None:
    """Trace logging is synchronous only during evaluate, then restored."""
    from anvil.eval import runner

    _patch_runner_common(monkeypatch, _wiring_config(n_workers=2))
    env_name = "MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"
    if prior is None:
        monkeypatch.delenv(env_name, raising=False)
    else:
        monkeypatch.setenv(env_name, prior)

    captured: dict[str, str | None] = {}

    def fake_evaluate(**_kwargs: object) -> object:
        captured["async_trace_logging"] = os.environ.get(env_name)
        return SimpleNamespace(result_df=_result_df(), metrics={}, run_id="run-1")

    monkeypatch.setattr(runner.mlflow.genai, "evaluate", fake_evaluate)

    runner.evaluate_branch(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=tmp_path / "config.yaml",
        runtime_client=SimpleNamespace(),
        judge_client=SimpleNamespace(),
    )

    assert captured["async_trace_logging"] == "false"
    assert os.environ.get(env_name) == prior


def test_evaluate_branch_restores_env_on_evaluate_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ``mlflow.genai.evaluate`` raises, the ``finally`` block still
    restores the prior ``MLFLOW_GENAI_EVAL_MAX_WORKERS`` value — the
    override never leaks past the call, even on the error path."""
    from anvil.eval import runner

    _patch_runner_common(monkeypatch, _wiring_config(n_workers=3))
    monkeypatch.delenv("MLFLOW_GENAI_EVAL_MAX_WORKERS", raising=False)

    def boom_evaluate(**_kwargs: object) -> object:
        raise RuntimeError("evaluate blew up")

    monkeypatch.setattr(runner.mlflow.genai, "evaluate", boom_evaluate)

    with pytest.raises(RuntimeError, match="evaluate blew up"):
        runner.evaluate_branch(
            scaffold_root=tmp_path / "scaffold",
            runtime_config_path=tmp_path / "config.yaml",
            runtime_client=SimpleNamespace(),
            judge_client=SimpleNamespace(),
        )

    # finally restored the env var: unset before the call → unset after.
    assert "MLFLOW_GENAI_EVAL_MAX_WORKERS" not in os.environ


def test_evaluate_branch_sequential_when_n_workers_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """n_workers <= 1 maps to MLFLOW_GENAI_EVAL_MAX_WORKERS=1 during the
    call, forcing mlflow's pool to a single worker (sequential)."""
    from anvil.eval import runner

    _patch_runner_common(monkeypatch, _wiring_config(n_workers=1))
    monkeypatch.delenv("MLFLOW_GENAI_EVAL_MAX_WORKERS", raising=False)

    captured: dict[str, str | None] = {}

    def fake_evaluate(**kwargs: object) -> object:
        captured["workers"] = os.environ.get("MLFLOW_GENAI_EVAL_MAX_WORKERS")
        return SimpleNamespace(result_df=_result_df(), metrics={}, run_id="run-1")

    monkeypatch.setattr(runner.mlflow.genai, "evaluate", fake_evaluate)

    runner.evaluate_branch(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=tmp_path / "config.yaml",
        runtime_client=SimpleNamespace(),
        judge_client=SimpleNamespace(),
    )

    assert captured["workers"] == "1"
    assert "MLFLOW_GENAI_EVAL_MAX_WORKERS" not in os.environ


def test_evaluate_branch_default_n_workers_is_four(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An EvalConfig that does not set n_workers inherits the default 4,
    which evaluate_branch forwards to the env var — backward compatible
    with configs written before this wiring landed."""
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
            scorers=[ScorerConfig(name="correctness", type="llm", weight=1.0)],
            modes={"quick": EvalModeConfig(rows=2, buckets={"direct": 2})},
            # n_workers intentionally unset → default 4.
        ),
    )
    _patch_runner_common(monkeypatch, config)
    monkeypatch.delenv("MLFLOW_GENAI_EVAL_MAX_WORKERS", raising=False)

    captured: dict[str, str | None] = {}

    def fake_evaluate(**kwargs: object) -> object:
        captured["workers"] = os.environ.get("MLFLOW_GENAI_EVAL_MAX_WORKERS")
        return SimpleNamespace(result_df=_result_df(), metrics={}, run_id="run-1")

    monkeypatch.setattr(runner.mlflow.genai, "evaluate", fake_evaluate)

    runner.evaluate_branch(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=tmp_path / "config.yaml",
        runtime_client=SimpleNamespace(),
        judge_client=SimpleNamespace(),
    )

    assert captured["workers"] == "4"
    assert "MLFLOW_GENAI_EVAL_MAX_WORKERS" not in os.environ
