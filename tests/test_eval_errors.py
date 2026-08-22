"""An error is not a bad answer — Phase 2, steps 3-5.

A **failure** is an expectation that was assessed and not met. An **error** is
an expectation that was never assessed. The first is a fact about the agent and
belongs in the score; the second is a fact about the infrastructure and must be
excluded from it. Conflating them means a rate-limited gateway and a bad answer
move the promotion gate by the same amount, in the same direction — so a
degraded endpoint reverts good work, and nothing in the round record says that
is what happened.

These tests pin down the three places that has to hold:

1. ``_resilient_eval_harness`` captures mlflow's per-row ``error_message``.
   mlflow already records it (``harness.py``: ``eval_item.error_message = ...``)
   and then reads it nowhere; the row proceeds to scoring with ``None`` outputs
   and the judges score the absence of an answer as a wrong answer.
2. ``_aggregate_report`` excludes those rows from ``per_judge``, ``per_bucket``,
   and the aggregate, rather than averaging their near-zero scores in — which
   also means preferring anvil's own mean over mlflow's ``{name}/mean``, since
   mlflow's includes them.
3. ``run_round`` refuses to compare a round whose error rate is above the
   configured ceiling: it is an ``INFRA_FAIL``, never a revert, and it must not
   touch the frontier.

No LLM and no Databricks workspace: the mlflow harness tests run against a
local file tracking store, and the aggregate/round tests drive the pure
functions directly.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. The shim captures mlflow's per-row error_message
# ---------------------------------------------------------------------------


@pytest.fixture
def local_mlruns(tmp_path: Path):
    """Point mlflow at an isolated local file store for the test."""
    import mlflow

    store = tmp_path / "mlruns"
    old_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"file://{store}")
    yield store
    mlflow.set_tracking_uri(old_uri)


def _gold_row(example_id: str) -> dict:
    return {
        "inputs": {"query": f"q-{example_id}", "category": "direct"},
        "expectations": {"expected_facts": ["fact-a"], "should_refuse": False},
        "tags": {"example_id": example_id},
    }


def _passing_scorer():
    from mlflow.genai.scorers import scorer

    @scorer
    def _passing(inputs, outputs, expectations, trace):
        return 1.0

    return _passing


def test_shim_captures_the_error_message_mlflow_already_records(
    local_mlruns: Path,
) -> None:
    """A row whose ``predict_fn`` raises is recorded by mlflow with an
    ``error_message`` and ``outputs`` left None — the information needed to
    exclude the row exists, and until now nothing read it.

    Drives the REAL ``mlflow.genai.evaluate`` so the assertion is about
    mlflow's actual behaviour, not a mock of it. The sink is keyed by the
    row's ``trace_id`` because that is the only identifier that also appears
    in ``result_df``, which is what the aggregate reads.
    """
    import mlflow

    from anvil.eval.runner import _resilient_eval_harness

    mlflow.set_experiment("test_error_capture")

    def predict_fn(query, **_kwargs):
        if query == "q-bad":
            raise RuntimeError("gateway said 429")
        with mlflow.start_span(name="anvil.predict") as span:
            span.set_inputs({"query": query})
            span.set_outputs({"response": "answer"})
            return "answer"

    sink: dict[str, str] = {}
    with _resilient_eval_harness(error_sink=sink):
        result = mlflow.genai.evaluate(
            data=[_gold_row("good"), _gold_row("bad")],
            scorers=[_passing_scorer()],
            predict_fn=predict_fn,
        )

    assert len(sink) == 1, f"exactly one row raised; sink={sink}"
    message = next(iter(sink.values()))
    assert "gateway said 429" in message

    # The key joins to result_df, which is how _aggregate_report finds the row.
    trace_ids = set(result.result_df["trace_id"])
    assert set(sink) <= trace_ids


def test_shim_captures_nothing_when_every_row_succeeds(local_mlruns: Path) -> None:
    """The capture is silent on the happy path — an empty sink is the signal
    that the aggregate may use mlflow's own means."""
    import mlflow

    from anvil.eval.runner import _resilient_eval_harness

    mlflow.set_experiment("test_error_capture_clean")

    def predict_fn(query, **_kwargs):
        with mlflow.start_span(name="anvil.predict") as span:
            span.set_inputs({"query": query})
            span.set_outputs({"response": "answer"})
            return "answer"

    sink: dict[str, str] = {}
    with _resilient_eval_harness(error_sink=sink):
        mlflow.genai.evaluate(
            data=[_gold_row("a"), _gold_row("b")],
            scorers=[_passing_scorer()],
            predict_fn=predict_fn,
        )

    assert sink == {}


# ---------------------------------------------------------------------------
# 2. The aggregate excludes errored rows instead of scoring them zero
# ---------------------------------------------------------------------------


def _df(scores: list[float], trace_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"correctness/value": scores, "trace_id": trace_ids})


def _examples(n: int, category: str = "direct") -> list[dict]:
    return [{"example_id": f"g{i}", "query": f"q{i}", "category": category} for i in range(n)]


def _report(**overrides: Any):
    from anvil.eval.runner import _aggregate_report

    kwargs: dict[str, Any] = {
        "result_df": _df([1.0, 0.0, 1.0], ["t0", "t1", "t2"]),
        "metrics": {},
        "scorer_names": ["correctness"],
        "aggregate_scorer_names": ["correctness"],
        "weights": {"correctness": 1.0},
        "examples": _examples(3),
        "run_id": "run-1",
        "experiment_id": "exp-1",
        "mode": "quick",
    }
    kwargs.update(overrides)
    return _aggregate_report(**kwargs)


def test_errored_row_is_excluded_from_the_aggregate_not_scored_zero() -> None:
    """The row that never ran scored 0.0 because an absent answer is a wrong
    answer. Excluded, the two rows that did run average 1.0 — the honest
    measurement of the agent. Averaged in, it is 0.667, and that 0.333 of
    pure infrastructure noise is what used to move the gate.
    """
    errored = _report(errored={"t1": "RuntimeError: gateway said 429"})
    assert errored.per_judge["correctness"] == 1.0
    assert errored.aggregate == 1.0
    assert errored.n_errors == 1
    assert errored.n_rows == 3
    assert errored.error_rate == pytest.approx(1 / 3)

    # The same frame with nothing errored still averages all three rows, so
    # the exclusion is doing the work and not a changed default.
    assert _report().per_judge["correctness"] == pytest.approx(2 / 3)
    assert _report().n_errors == 0
    assert _report().error_rate == 0.0


def test_mlflow_mean_is_ignored_when_a_row_errored() -> None:
    """``_aggregate_report`` prefers mlflow's ``{name}/mean`` when present.
    mlflow computes that mean over every row *including* the errored ones, so
    once anything errored that metric is exactly the number this change
    exists to stop trusting. anvil's own mean must win.
    """
    metrics = {"correctness/mean": 2 / 3}  # mlflow's, includes the errored row

    clean = _report(metrics=metrics)
    assert clean.per_judge["correctness"] == pytest.approx(2 / 3), (
        "with no errors, mlflow's mean is still preferred"
    )

    errored = _report(metrics=metrics, errored={"t1": "boom"})
    assert errored.per_judge["correctness"] == 1.0


def test_errored_row_is_not_reported_as_a_judge_failure() -> None:
    """A failure list that includes never-assessed rows sends the optimizer
    chasing a bad answer that was never given. The errored row is reported as
    an error, not as a case the agent got wrong."""
    report = _report(errored={"t1": "boom"})
    assert [f["example_id"] for f in report.failures] == []
    assert [e["example_id"] for e in report.errors] == ["g1"]
    assert "boom" in report.errors[0]["error_message"]


def test_per_bucket_excludes_errored_rows() -> None:
    """Per-bucket means are read by the optimizer to decide *where* to focus.
    An errored row would depress its bucket and point the next mutation at a
    category that is not actually weak."""
    report = _report(
        result_df=_df([1.0, 0.0], ["t0", "t1"]),
        examples=[
            {"example_id": "g0", "query": "q0", "category": "direct"},
            {"example_id": "g1", "query": "q1", "category": "multi_hop"},
        ],
        errored={"t1": "boom"},
    )
    assert report.per_bucket["direct"]["correctness"] == 1.0
    # The only multi_hop row never ran, so there is no measurement for it.
    # 0.0 would assert the agent failed at multi_hop, which is a claim the
    # round has no evidence for.
    assert "multi_hop" not in report.per_bucket


def test_every_row_errored_yields_no_measurement_not_a_zero_score() -> None:
    """A wholly failed eval has an error rate of 1.0. Its aggregate carries no
    information, which is precisely why the round guard must read the error
    rate rather than the score."""
    report = _report(errored={"t0": "boom", "t1": "boom", "t2": "boom"})
    assert report.error_rate == 1.0
    assert report.n_errors == 3
    assert report.per_judge["correctness"] == 0.0  # no rows left to average


def test_unmatchable_error_still_counts_toward_the_error_rate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An error captured for a trace_id absent from result_df cannot have its
    score excluded — there is no row to exclude. It must still count toward
    the error rate (so the guard fires) and be logged, rather than being
    silently dropped because it could not be joined."""
    import logging

    with caplog.at_level(logging.WARNING, logger="anvil.eval.runner"):
        report = _report(errored={"unknown-trace": "boom"})
    assert report.n_errors == 1
    assert report.error_rate == pytest.approx(1 / 3)
    assert "could not be attributed" in caplog.text


# ---------------------------------------------------------------------------
# 3. evaluate_branch surfaces the error count end to end
# ---------------------------------------------------------------------------


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


def test_evaluate_branch_reports_the_error_rate(
    local_mlruns: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End to end against the real mlflow harness: one of two rows raises, and
    the returned ``EvalReport`` says so. Before this, the report had no way to
    express it and the round could not tell a throttled gateway from a bad
    mutation."""
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
        experiments=ExperimentsConfig(runtime="r", eval="error_rate_eval", optimizer="o"),
        eval=EvalConfig(
            default_mode="quick",
            scorers=[ScorerConfig(name="trivial", type="llm", weight=1.0)],
            modes={"quick": EvalModeConfig(rows=2, buckets={"direct": 2})},
            n_workers=1,
        ),
    )

    class _Agent:
        def predict(self, request):
            query = request.input[0].content
            if query == "q-g2":
                raise RuntimeError("gateway said 429")
            return SimpleNamespace(
                output=[{"type": "message", "content": [{"type": "output_text", "text": "answer"}]}]
            )

    monkeypatch.setattr(runner, "load_harness", lambda *a, **kw: SimpleNamespace(config=config))
    monkeypatch.setattr(
        runner, "load_golden_set", lambda _p: [_gold("g1", "hello"), _gold("g2", "world")]
    )
    monkeypatch.setattr(runner, "select_subset", lambda exs, **_k: exs)
    monkeypatch.setattr(runner, "make_kb_executor", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(runner, "AnvilAgent", lambda *a, **kw: _Agent())
    monkeypatch.setattr(runner, "enable_runtime_tracing", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "build_scorers", lambda **_kw: [_passing_scorer()])

    report = runner.evaluate_branch(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=tmp_path / "config.yaml",
        golden_set_path="unused",
        kb_dir=tmp_path / "kb",
        runtime_client=SimpleNamespace(),
        judge_client=SimpleNamespace(),
    )

    assert report.n_errors == 1
    assert report.error_rate == 0.5
    assert "429" in report.errors[0]["error_message"]


# ---------------------------------------------------------------------------
# 4. The round refuses to judge a degraded eval
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    import subprocess

    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


@pytest.fixture
def anvil_repo(tmp_path: Path) -> Path:
    """A committed ANVIL repo on ``anvil/exp``, ready for ``run_round``."""
    from anvil.eval.cache import CachedBaseline, save_baseline

    repo = tmp_path
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@e.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")

    (repo / "scaffold" / "memory").mkdir(parents=True)
    (repo / "scaffold" / "skills").mkdir()
    (repo / "scaffold" / "harness.yaml").write_text("skills: []\ntools: []\n")
    (repo / "harness").mkdir()
    (repo / "harness" / "config.yaml").write_text("mode: prompt\neval:\n  max_error_rate: 0.25\n")
    (repo / "eval" / "runs").mkdir(parents=True)

    save_baseline(
        repo,
        CachedBaseline(
            scaffold_commit_sha="a" * 40,
            evaluated_at="2026-08-22T12:00:00+00:00",
            mode="test",
            scorers=["correctness"],
            runtime_endpoint="runtime",
            judge_endpoint="judge",
            aggregate=0.5,
            per_judge={"correctness": 0.5},
            per_bucket={"direct": {"correctness": 0.5}},
            n_examples=10,
        ),
    )

    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    _git(repo, "checkout", "-q", "-b", "anvil/exp")
    return repo


def _mutating_session(repo: Path):
    """An optimizer session that adds a skill — a real (non-noop) mutation, so
    the round runs an eval and reaches the gate."""
    from anvil.optimizer.actions import AddSkillAction
    from anvil.optimizer.parser import ParseResult

    async def _session(**_kwargs):
        action = AddSkillAction(
            rationale="add a skill so the round is scored",
            target_file="skills/concise.md",
            content="# Be concise\n",
        )
        return action, "transcript", ParseResult(action=action, parse_status="ok", n_blocks_found=1)

    return _session


def _report_with_error_rate(error_rate: float, *, aggregate: float = 0.9):
    """An ``EvalReport`` whose error rate is exactly ``error_rate`` over 8 rows."""
    from anvil.eval.runner import EvalReport

    n_rows = 8
    n_errors = round(error_rate * n_rows)
    return EvalReport(
        aggregate=aggregate,
        per_judge={"correctness": aggregate},
        per_bucket={"direct": {"correctness": aggregate}},
        failures=[],
        run_id="run-x",
        experiment_id="exp-x",
        n_rows=n_rows,
        mode="quick",
        scorers=["correctness"],
        evaluated_at="2026-08-22T12:00:00+00:00",
        n_errors=n_errors,
        errors=[{"example_id": f"g{i}", "error_message": "429"} for i in range(n_errors)],
    )


def test_round_refuses_to_judge_an_eval_above_the_error_ceiling(
    anvil_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Half the cases never ran. The aggregate that survives is 0.9 — better
    than the 0.5 baseline — so the old code would have KEPT this round on the
    strength of a measurement over four rows.

    The reverse case is the dangerous one and the same mechanism covers it: a
    throttled gateway that leaves a *worse*-looking aggregate would REVERT a
    good mutation. Either way the round did not measure the agent, so it is
    failed rather than compared, and the frontier is left untouched.
    """
    import anvil.loop.round as round_mod
    from anvil.loop.decision import Decision

    monkeypatch.setattr(round_mod, "run_optimizer_session", _mutating_session(anvil_repo))
    monkeypatch.setattr(round_mod, "evaluate_branch", lambda **_kw: _report_with_error_rate(0.5))

    report = round_mod.run_round(round_id=1, repo_root=anvil_repo)

    assert report.decision == Decision.INFRA_FAIL
    assert "error rate 0.50 exceeds ceiling 0.25" in report.notes
    assert "4/8 cases never assessed" in report.notes
    # The score is kept on the record even though it was not compared: "0.9,
    # but half the cases never ran" is more useful later than a null.
    assert report.mutated_score == 0.9
    # The frontier was never written, so the untrustworthy number cannot become
    # the bar the next round has to beat.
    assert not (anvil_repo / "eval" / "runs" / "frontier.json").exists()


def test_round_scores_normally_when_the_error_rate_is_under_the_ceiling(
    anvil_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stray error is not a reason to throw the round away. One case in eight
    is under the ceiling, so the round is judged on the seven that ran."""
    import anvil.loop.round as round_mod
    from anvil.loop.decision import Decision

    monkeypatch.setattr(round_mod, "run_optimizer_session", _mutating_session(anvil_repo))
    monkeypatch.setattr(round_mod, "evaluate_branch", lambda **_kw: _report_with_error_rate(0.125))

    report = round_mod.run_round(round_id=1, repo_root=anvil_repo)

    assert report.decision == Decision.KEEP
    assert report.notes == ""


def test_round_records_the_error_rate_in_the_round_json(
    anvil_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The evidence has to carry it. A round JSON with an aggregate and no
    error count cannot be re-read to tell a bad mutation from a bad afternoon
    on the gateway."""
    import json

    import anvil.loop.round as round_mod

    monkeypatch.setattr(round_mod, "run_optimizer_session", _mutating_session(anvil_repo))
    monkeypatch.setattr(round_mod, "evaluate_branch", lambda **_kw: _report_with_error_rate(0.125))

    round_mod.run_round(round_id=1, repo_root=anvil_repo)

    payload = json.loads((anvil_repo / "eval" / "runs" / "round_001.json").read_text())
    assert payload["n_errors"] == 1
    assert payload["error_rate"] == pytest.approx(0.125)
    assert payload["errors"][0]["error_message"] == "429"


def test_max_error_rate_ceiling_is_read_from_config() -> None:
    """The ceiling is configurable and validated. A NaN would make the guard's
    comparison silently False — set, but disabled."""
    from anvil.runtime.models import EvalConfig

    assert EvalConfig().max_error_rate == 0.2
    assert EvalConfig(max_error_rate=0.0).max_error_rate == 0.0
    assert EvalConfig(max_error_rate=1.0).max_error_rate == 1.0
    for bad in (-0.1, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="max_error_rate"):
            EvalConfig(max_error_rate=bad)


# ---------------------------------------------------------------------------
# 5. Judgeability — the three ways an excluded-errors report is still untrustworthy
# ---------------------------------------------------------------------------


def _eval_report(*, n_rows: int, n_errors: int = 0, n_unattributed: int = 0):
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
        n_unattributed_errors=n_unattributed,
    )


def test_a_healthy_report_is_judgeable() -> None:
    from anvil.eval.judgeability import unjudgeable_reason

    assert unjudgeable_reason(_eval_report(n_rows=8)) == ""
    assert unjudgeable_reason(_eval_report(n_rows=8, n_errors=1)) == ""


def test_an_unexcludable_error_makes_the_report_unjudgeable() -> None:
    """The hole that excluding-by-trace_id leaves open.

    An error whose row is not in ``result_df`` cannot have its score removed —
    there is no row to remove — so its infrastructure zero is still in the mean.
    One such error in eight rows sits at an error rate of 0.125, comfortably
    under the 0.2 ceiling, so without this check the round would be judged
    normally on an aggregate that still contains the zero: exactly the bug the
    exclusion exists to fix, quietly reintroduced.
    """
    from anvil.eval.judgeability import unjudgeable_reason

    report = _eval_report(n_rows=8, n_errors=1, n_unattributed=1)
    assert report.error_rate < 0.2
    reason = unjudgeable_reason(report)
    assert "could not be attributed" in reason
    assert "did not happen" in reason


def test_too_few_assessed_cases_is_unjudgeable_even_with_the_rate_guard_off() -> None:
    """The floor exists because a rate is relative.

    ``max_error_rate: 1.0`` reads as "disable the guard", i.e. as restoring the
    pre-exclusion behaviour. It does something strictly more dangerous: seven
    errors in eight rows used to score ~0.12 and be REVERTED, but excluded, the
    aggregate becomes the score of the one surviving row — 1.0 if it passed —
    which EXTENDS the frontier and becomes the bar every later round must beat.
    No rate can express "at least N cases actually ran".
    """
    from anvil.eval.judgeability import unjudgeable_reason

    report = _eval_report(n_rows=8, n_errors=7)
    assert unjudgeable_reason(report, max_error_rate=1.0) != ""
    assert "below the floor" in unjudgeable_reason(report, max_error_rate=1.0)


def test_the_floor_is_capped_at_the_run_size() -> None:
    """A deliberately small mode must stay runnable. A 2-row smoke eval cannot
    satisfy a floor of 4, and a guard that fires on correct usage gets switched
    off — so the floor asks for "4 assessed cases, or all of them if fewer"."""
    from anvil.eval.judgeability import unjudgeable_reason

    assert unjudgeable_reason(_eval_report(n_rows=2), min_scorable_rows=4) == ""
    # But it still demands ALL of them when the run is that small.
    assert unjudgeable_reason(_eval_report(n_rows=2, n_errors=1), min_scorable_rows=4) != ""


def test_the_floor_can_be_switched_off_explicitly() -> None:
    from anvil.eval.judgeability import unjudgeable_reason

    report = _eval_report(n_rows=8, n_errors=7)
    assert unjudgeable_reason(report, max_error_rate=1.0, min_scorable_rows=0) == ""


def test_min_scorable_rows_is_validated() -> None:
    from anvil.runtime.models import EvalConfig

    assert EvalConfig().min_scorable_rows == 4
    assert EvalConfig(min_scorable_rows=0).min_scorable_rows == 0
    with pytest.raises(ValueError, match="min_scorable_rows"):
        EvalConfig(min_scorable_rows=-1)


def test_an_empty_run_reads_as_maximally_errored_not_clean() -> None:
    """A guard's degenerate case must point at "refuse". ``0.0`` for a run with
    no rows but recorded errors would be a fail-open sentinel: unmeasurable, and
    passing every check."""
    from anvil.eval.judgeability import unjudgeable_reason

    assert _eval_report(n_rows=0, n_errors=3).error_rate == 1.0
    assert _eval_report(n_rows=0).error_rate == 0.0
    assert unjudgeable_reason(_eval_report(n_rows=0, n_errors=3)) != ""


def test_round_refuses_a_round_measured_on_too_few_cases(
    anvil_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The floor reaches the round gate, not just the CLI. Seven of eight cases
    errored; the surviving one scored 0.9, better than the 0.5 baseline. Without
    the floor this KEEPs and the frontier is advanced from a single row."""
    import anvil.loop.round as round_mod
    from anvil.loop.decision import Decision

    # Rate guard wide open, so only the floor can catch this.
    (anvil_repo / "harness" / "config.yaml").write_text(
        "mode: prompt\neval:\n  max_error_rate: 1.0\n  min_scorable_rows: 4\n"
    )
    _git(anvil_repo, "commit", "-qam", "open the rate guard")

    monkeypatch.setattr(round_mod, "run_optimizer_session", _mutating_session(anvil_repo))
    monkeypatch.setattr(
        round_mod, "evaluate_branch", lambda **_kw: _report_with_error_rate(0.875)
    )

    report = round_mod.run_round(round_id=1, repo_root=anvil_repo)

    assert report.decision == Decision.INFRA_FAIL
    assert "below the floor" in report.notes
    assert not (anvil_repo / "eval" / "runs" / "frontier.json").exists()


# ---------------------------------------------------------------------------
# 6. The baseline and the held-out finalization are guarded too
# ---------------------------------------------------------------------------


def _load_script(name: str):
    import importlib.util
    import sys

    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"anvil_script_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _baseline_repo(tmp_path: Path) -> Path:
    (tmp_path / "scaffold").mkdir()
    (tmp_path / "harness").mkdir()
    (tmp_path / "harness" / "config.yaml").write_text(
        "runtime_endpoint: rt\noptimizer_endpoint: op\njudge_endpoint: j\n"
        "experiments:\n  runtime: r\n  eval: e\n  optimizer: o\n"
        "eval:\n  max_error_rate: 0.2\n  min_scorable_rows: 4\n"
    )
    return tmp_path


def test_baseline_generation_refuses_a_degraded_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression that excluding errors *introduced*, and the reason this
    guard is not optional.

    Before exclusion, a baseline run that 429'd on six of eight rows produced an
    aggregate near 0.25 — visibly broken, and an operator would rerun it. After
    exclusion the same run reads the mean of the two rows that survived, which
    is *higher* than a healthy baseline and indistinguishable from one. And the
    baseline is the frontier's seed and the bar every round is compared against,
    so freezing it would mis-steer the entire run.
    """
    module = _load_script("make_baseline")
    repo = _baseline_repo(tmp_path)

    monkeypatch.setattr(
        module, "evaluate_branch", lambda **_kw: _report_with_error_rate(0.75, aggregate=0.95)
    )

    with pytest.raises(RuntimeError, match="refusing to cache a baseline"):
        module.build_baseline(scaffold_root=repo / "scaffold")


def test_baseline_generation_accepts_a_healthy_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control case, and it records how much of the eval ran: a cached
    baseline with an aggregate and no error count cannot be re-read later to
    tell a good bar from a lucky one."""
    module = _load_script("make_baseline")
    repo = _baseline_repo(tmp_path)

    monkeypatch.setattr(
        module, "evaluate_branch", lambda **_kw: _report_with_error_rate(0.125, aggregate=0.7)
    )

    baseline = module.build_baseline(scaffold_root=repo / "scaffold")
    assert baseline.aggregate == 0.7
    assert baseline.n_errors == 1
    assert baseline.to_dict()["n_errors"] == 1


def test_a_clean_baseline_keeps_the_historical_on_disk_schema() -> None:
    """``n_errors`` is additive: a baseline with none omits the key, so files
    written before the failure/error split still round-trip byte-identically."""
    from anvil.eval.cache import CachedBaseline

    clean = CachedBaseline(
        scaffold_commit_sha="a" * 40,
        evaluated_at="2026-08-22T12:00:00+00:00",
        mode="quick",
        scorers=["correctness"],
        runtime_endpoint="rt",
        judge_endpoint="j",
        aggregate=0.5,
    )
    assert "n_errors" not in clean.to_dict()
    assert CachedBaseline.from_dict(clean.to_dict()).n_errors == 0


def test_finalization_refuses_a_degraded_held_out_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The highest-stakes number the harness produces, and it was the only eval
    path with no guard. It is also write-once — ``main`` refuses to overwrite an
    existing finalized.json — so a degraded run does not merely mislead, it
    locks in until someone deletes the file by hand."""
    from anvil.loop.frontier import Frontier, save_frontier

    module = _load_script("finalize")
    repo = _baseline_repo(tmp_path)
    (repo / "harness" / "config.yaml").write_text(
        (repo / "harness" / "config.yaml").read_text() + "  held_out_test: true\n"
    )
    save_frontier(repo, Frontier.from_scores({"aggregate": 0.5}))

    monkeypatch.setattr(
        module, "evaluate_branch", lambda **_kw: _report_with_error_rate(0.75, aggregate=0.99)
    )

    with pytest.raises(RuntimeError, match="refusing to finalize"):
        module.finalize(repo_root=repo, scaffold_root=repo / "scaffold")


def test_shipped_config_declares_the_error_ceiling() -> None:
    """The guard is only useful if the shipped config surfaces it, otherwise
    nobody knows it exists to tune."""
    import yaml

    from anvil.runtime.models import RuntimeYAML

    raw = yaml.safe_load((REPO_ROOT / "harness" / "config.yaml").read_text(encoding="utf-8"))
    assert "max_error_rate" in raw["eval"]
    assert "min_scorable_rows" in raw["eval"]
    cfg = RuntimeYAML.model_validate(raw).eval
    assert cfg.max_error_rate == 0.2
    assert cfg.min_scorable_rows == 4
