"""Groundedness applicability, scorer-error visibility, and the tracing race.

Three defects that presented as one symptom: ``retrieval_groundedness`` failing
3-4 of 8 invocations on every healthy live run, with a single message —
``SCORER_ERROR: No retrieval context found in the trace.`` — while the aggregate
read fine and every guard stayed quiet.

* **Applicability** (:func:`anvil.eval.scorers._build_groundedness_scorer`).
  2 of the 8 quick-mode rows are ``out_of_scope`` with ``expected_doc_ids == []``:
  the agent correctly refuses, never retrieves, and grounding does not apply.
  mlflow's scorer has no way to say "not applicable" — it raises — so those 2
  were indistinguishable from a broken judge. And the same exclusion applied to
  a row that *was* meant to retrieve, which is a reward-hacking hole: not
  retrieving became free.
* **Scorer-error visibility** (``_row_scorer_error``, per-judge counts).
  ``construct_eval_result_df`` flattens only a feedback's *value*, so a scorer
  error and a scorer abstention are both ``None`` in the ``{name}/value`` column.
  Every guard was blind to it.
* **The tracing race** (``anvil.eval.scorers`` module docstring). The refusal
  judge's call used to run inside a process-global ``mlflow.tracing.disable()``
  while mlflow ran predictions concurrently on another pool.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gold(example_id: str, *, expected_doc_ids: list[str], should_refuse: bool = False) -> dict:
    return {
        "example_id": example_id,
        "query": f"q-{example_id}",
        "category": "out_of_scope" if should_refuse else "direct",
        "expected_doc_ids": expected_doc_ids,
        "reference_answer": "a",
        "should_refuse": should_refuse,
        "expected_citations": [],
        "must_include": [],
        "must_not_include": [],
        "notes_for_judge": "",
    }


def _scorer_error_assessment(name: str, message: str) -> dict:
    """The shape mlflow records for a scorer that raised.

    Verified against ``Feedback(error=AssessmentError(...)).to_dictionary()`` on
    mlflow 3.11.1 — a scorer exception never aborts the run, it becomes a
    valueless feedback carrying the error.
    """
    return {
        "assessment_name": name,
        "feedback": {
            "value": None,
            "error": {"error_code": "SCORER_ERROR", "error_message": message},
        },
    }


def _value_assessment(name: str, value: Any) -> dict:
    return {"assessment_name": name, "feedback": {"value": value}}


# ---------------------------------------------------------------------------
# The applicability rule
# ---------------------------------------------------------------------------


def test_groundedness_is_not_applicable_when_no_documents_are_expected() -> None:
    """A row the golden set never asked to retrieve yields no assessment.

    ``None`` and not ``0.0``: mlflow reads None as "no feedback"
    (``standardize_scorer_value``), so the row drops out of this judge's mean
    instead of asserting the agent was ungrounded on a question it was right to
    refuse.
    """
    from anvil.eval.scorers import _build_groundedness_scorer

    scorer = _build_groundedness_scorer()
    result = scorer.run(
        inputs={"query": "who won the world cup"},
        outputs="I can only help with NeoVolt questions.",
        expectations={"expected_doc_ids": [], "should_refuse": True},
        trace=object(),
    )
    assert result is None


def test_groundedness_scores_zero_when_retrieval_was_expected_but_skipped() -> None:
    """The reward-hacking guard, and the whole reason this wrapper exists.

    Groundedness is binary, so excluding the rows where the agent skipped
    retrieval moved this judge from 1/8 = 0.125 to 1/1 = 1.0 — an 0.875 swing
    available purely by searching less, and the largest single lever in the
    scoring system. Scoring it "no" makes abstention cost exactly what a wrong
    answer costs.
    """
    from anvil.eval.scorers import _build_groundedness_scorer

    scorer = _build_groundedness_scorer()
    result = scorer.run(
        inputs={"query": "what is the residential rate"},
        outputs="It is about fourteen cents, I think.",
        expectations={"expected_doc_ids": ["tariff_standard_residential"]},
        trace=object(),  # no retrieval context extractable
    )
    assert result is not None, "an expected-retrieval row must never be skipped"
    assert result.value == "no"
    assert "expected_doc_ids" in (result.rationale or "")


def test_groundedness_defers_to_the_real_judge_when_retrieval_happened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With retrieval context present, mlflow's grounding judge decides. The
    wrapper supplies applicability and the context; it must not become a second
    opinion on grounding itself."""
    from mlflow.entities import Feedback

    import anvil.eval.scorers as scorers_mod

    sentinel = Feedback(name="retrieval_groundedness", value="yes", rationale="from mlflow")
    monkeypatch.setattr(
        scorers_mod, "extract_retrieval_context_from_trace", lambda trace: {"span-1": ["chunk"]}
    )
    monkeypatch.setattr(scorers_mod, "extract_request_from_trace", lambda trace: "the question")
    monkeypatch.setattr(scorers_mod, "extract_response_from_trace", lambda trace: "the answer")
    monkeypatch.setattr(
        scorers_mod.judges, "is_grounded", lambda **kwargs: (seen.update(kwargs), sentinel)[1]
    )
    seen: dict[str, Any] = {}

    scorer = scorers_mod._build_groundedness_scorer()
    result = scorer.run(
        inputs={"query": "q"},
        outputs="a",
        expectations={"expected_doc_ids": ["doc-1"]},
        trace=object(),
    )
    assert result is sentinel
    assert seen["request"] == "the question"
    assert seen["response"] == "the answer"
    assert seen["context"] == ["chunk"]


def test_groundedness_grades_with_the_configured_judge_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #13: the wrapper called ``is_grounded`` with no model.

    Groundedness graded on mlflow's implicit default while the refusal judge
    used the configured endpoint -- one config knob, two models, and a
    ``judge_endpoint`` change that moved only one judge.
    """
    from mlflow.entities import Feedback

    import anvil.eval.scorers as scorers_mod

    sentinel = Feedback(name="retrieval_groundedness", value="yes", rationale="from mlflow")
    monkeypatch.setattr(
        scorers_mod, "extract_retrieval_context_from_trace", lambda trace: {"span-1": ["chunk"]}
    )
    monkeypatch.setattr(scorers_mod, "extract_request_from_trace", lambda trace: "q")
    monkeypatch.setattr(scorers_mod, "extract_response_from_trace", lambda trace: "a")
    monkeypatch.setattr(
        scorers_mod.judges, "is_grounded", lambda **kwargs: (seen.update(kwargs), sentinel)[1]
    )
    seen: dict[str, Any] = {}

    scorer = scorers_mod._build_groundedness_scorer(model="databricks:/my-judge-endpoint")
    result = scorer.run(
        inputs={"query": "q"},
        outputs="a",
        expectations={"expected_doc_ids": ["doc-1"]},
        trace=object(),
    )
    assert result is sentinel
    assert seen["model"] == "databricks:/my-judge-endpoint"


def test_groundedness_judges_the_union_of_every_retrieval_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The multi-span lever, and a real measurement error alongside it.

    mlflow's scorer returns one feedback per retrieval span, and
    ``construct_eval_result_df`` collapses them last-wins — so the row's score
    was decided by whichever search happened to be flattened last, which the
    agent chooses. A final narrow search whose chunks trivially support a closing
    sentence carried the row.

    It also mismeasured multi-hop: 6 of 20 golden rows expect 2-4 documents, so
    no single search supports the whole answer, and judging the complete answer
    against only the last search's chunks understates those rows systematically.
    One verdict over the union fixes both.
    """
    import anvil.eval.scorers as scorers_mod

    monkeypatch.setattr(
        scorers_mod,
        "extract_retrieval_context_from_trace",
        lambda trace: {
            "span-1": ["tariff chunk", "service charge chunk"],
            "span-2": ["outage chunk"],
        },
    )
    monkeypatch.setattr(scorers_mod, "extract_request_from_trace", lambda trace: "q")
    monkeypatch.setattr(scorers_mod, "extract_response_from_trace", lambda trace: "a")
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        scorers_mod.judges, "is_grounded", lambda **kwargs: (seen.update(kwargs), None)[1]
    )

    scorers_mod._build_groundedness_scorer().run(
        inputs={"query": "q"},
        outputs="a",
        expectations={"expected_doc_ids": ["tariff", "outage"]},
        trace=object(),
    )
    assert seen["context"] == ["tariff chunk", "service charge chunk", "outage chunk"], (
        "every retrieved chunk must reach the judge, not just the last span's"
    )


def test_the_judge_is_called_once_regardless_of_span_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One row, one verdict. Several verdicts under one assessment name is what
    let the flattening pick a winner in the first place."""
    import anvil.eval.scorers as scorers_mod

    monkeypatch.setattr(
        scorers_mod,
        "extract_retrieval_context_from_trace",
        lambda trace: {f"span-{i}": [f"chunk-{i}"] for i in range(5)},
    )
    monkeypatch.setattr(scorers_mod, "extract_request_from_trace", lambda trace: "q")
    monkeypatch.setattr(scorers_mod, "extract_response_from_trace", lambda trace: "a")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        scorers_mod.judges, "is_grounded", lambda **kwargs: (calls.append(kwargs), None)[1]
    )

    scorers_mod._build_groundedness_scorer().run(
        inputs={"query": "q"},
        outputs="a",
        expectations={"expected_doc_ids": ["d"]},
        trace=object(),
    )
    assert len(calls) == 1
    assert len(calls[0]["context"]) == 5


def test_a_synthesized_trace_is_not_scored_as_an_agent_failure() -> None:
    """The prediction ran; the tracing server would not return its trace, so the
    harness substituted a root-span-only stand-in. That trace has no RETRIEVER
    span for the same reason it has no other spans — infrastructure, not agent
    behaviour. Scoring it "no" would be the precise error this repo's
    failure-vs-error work exists to prevent, and a lost trace is what it keeps
    hitting live.
    """
    from anvil.eval.scorers import SYNTHESIZED_TRACE_TAG, _build_groundedness_scorer

    class _Info:
        tags = {SYNTHESIZED_TRACE_TAG: "true"}

    class _Trace:
        info = _Info()

    result = _build_groundedness_scorer().run(
        inputs={"query": "q"},
        outputs="an answer that did get produced",
        expectations={"expected_doc_ids": ["tariff_standard_residential"]},
        trace=_Trace(),
    )
    assert result is None, "a lost trace must be unmeasured, never scored 0"


def test_a_real_trace_without_the_tag_is_still_scored() -> None:
    """The escape hatch must not swallow the reward-hacking guard: an ordinary
    trace that simply has no retrieval still scores "no"."""
    from anvil.eval.scorers import _build_groundedness_scorer

    class _Info:
        tags: dict[str, str] = {}

    class _Trace:
        info = _Info()

    result = _build_groundedness_scorer().run(
        inputs={"query": "q"},
        outputs="a",
        expectations={"expected_doc_ids": ["d"]},
        trace=_Trace(),
    )
    assert result is not None
    assert result.value == "no"


def test_the_refusal_judge_suppresses_tracing_only_on_its_own_thread() -> None:
    """``mlflow.tracing.context(enabled=False)`` is ContextVar-scoped, so the
    judge's autolog is silenced without touching the global tracer that
    concurrent predictions depend on. The earlier code used
    ``mlflow.tracing.disable()``, which swaps the provider process-wide.

    Asserted from inside the judge call, on a second thread, because that is the
    only place and moment the old bug was observable.
    """
    import threading

    import mlflow
    from mlflow.tracing.provider import is_tracing_enabled

    from anvil.eval.scorers import _build_refusal_scorer, _JudgeContext

    observed: dict[str, Any] = {}

    class _Completions:
        def create(self, **kwargs: Any) -> Any:
            observed["global_still_enabled"] = is_tracing_enabled()
            with mlflow.start_span(name="judge-own-thread") as own:
                observed["own_thread_suppressed"] = own.trace_id == "MLFLOW_NO_OP_SPAN_TRACE_ID"

            def _other_thread() -> None:
                with mlflow.start_span(name="concurrent-prediction") as span:
                    observed["other_thread_trace_id"] = span.trace_id

            t = threading.Thread(target=_other_thread)
            t.start()
            t.join()
            raise RuntimeError("stop here — the judge's verdict is not what we test")

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    scorer = _build_refusal_scorer(_JudgeContext(client=_Client(), model="m"))  # type: ignore[arg-type]
    scorer.run(inputs={"query": "q"}, outputs="a", expectations={"should_refuse": False})

    assert observed["global_still_enabled"] is True
    assert observed["own_thread_suppressed"] is True, "the judge's own autolog must be silenced"
    assert observed["other_thread_trace_id"] != "MLFLOW_NO_OP_SPAN_TRACE_ID", (
        "a concurrent prediction must still get a real trace"
    )


def test_groundedness_reads_expected_doc_ids_not_should_refuse() -> None:
    """``expected_doc_ids`` is what groundedness is about.

    In the shipped golden set the two columns agree exactly, but they are
    separate columns, and a future row could legitimately expect a refusal to
    cite policy. Keying off ``should_refuse`` would silently stop grading such a
    row.
    """
    from anvil.eval.scorers import _build_groundedness_scorer

    scorer = _build_groundedness_scorer()
    result = scorer.run(
        inputs={"query": "q"},
        outputs="a",
        expectations={"expected_doc_ids": ["policy_doc"], "should_refuse": True},
        trace=object(),
    )
    assert result is not None
    assert result.value == "no"


def test_extraction_failure_is_not_evidence_of_retrieval() -> None:
    """If mlflow's extractor raises, the wrapper must not read that as "the
    agent retrieved" — that would hand back the free pass this fix removes."""
    import anvil.eval.scorers as scorers_mod

    def _boom(trace: Any) -> dict:
        raise RuntimeError("malformed span")

    original = scorers_mod.extract_retrieval_context_from_trace
    try:
        scorers_mod.extract_retrieval_context_from_trace = _boom  # type: ignore[assignment]
        assert scorers_mod._retrieved_chunks(object()) == []
    finally:
        scorers_mod.extract_retrieval_context_from_trace = original  # type: ignore[assignment]


def test_build_scorers_wires_the_wrapper_not_the_bare_mlflow_scorer() -> None:
    """The bare ``RetrievalGroundedness`` must not reach ``mlflow.genai.evaluate``.

    Same scorer name, so the config, the aggregate weights and every consumer are
    unchanged — but the object is anvil's.
    """
    from mlflow.genai.scorers import RetrievalGroundedness

    from anvil.eval.scorers import GROUNDEDNESS_SCORER_NAME, build_scorers

    built = build_scorers(judge_client=object(), judge_model="m")  # type: ignore[arg-type]
    by_name = {s.name: s for s in built}
    assert GROUNDEDNESS_SCORER_NAME in by_name
    assert not isinstance(by_name[GROUNDEDNESS_SCORER_NAME], RetrievalGroundedness)


# ---------------------------------------------------------------------------
# The tracing race
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Scorer errors reach the report
# ---------------------------------------------------------------------------


def test_row_scorer_error_reads_the_error_mlflow_recorded() -> None:
    from anvil.eval.runner import _row_scorer_error

    row = {
        "retrieval_groundedness/value": None,
        "assessments": [
            _value_assessment("correctness", 1.0),
            _scorer_error_assessment("retrieval_groundedness", "No retrieval context found."),
        ],
    }
    assert _row_scorer_error(row, "correctness") is None
    message = _row_scorer_error(row, "retrieval_groundedness")
    assert message is not None
    assert "SCORER_ERROR" in message
    assert "No retrieval context found." in message


def test_a_scored_row_and_an_abstention_are_not_reported_as_errors() -> None:
    """Absence of a value is not an error. Only a recorded ``error`` is."""
    from anvil.eval.runner import _row_scorer_error

    abstained = {"assessments": [{"assessment_name": "retrieval_groundedness", "feedback": {}}]}
    assert _row_scorer_error(abstained, "retrieval_groundedness") is None
    assert _row_scorer_error({"assessments": []}, "retrieval_groundedness") is None
    assert _row_scorer_error({}, "retrieval_groundedness") is None


def test_a_missing_value_in_a_numeric_column_is_unscored_not_nan() -> None:
    """Found by the test below, and worse than what it was testing.

    pandas stores a missing entry in an otherwise-numeric column as NaN, not
    None, so a scorer error on such a column came back from ``_coerce_score`` as
    a *number*. The row counted as assessed and its NaN entered the mean — where
    NaN propagates, so one scorer error turned the judge's mean and the entire
    weighted aggregate into NaN. Every NaN comparison being False, the frontier
    then silently fails its ``>`` check and reverts, or writes NaN to a baseline
    as the bar for every later round.

    Hidden live only because the shipped judges return "yes"/"no", which keeps
    the column ``object`` dtype and the missing value ``None``.
    """
    from anvil.eval.runner import _coerce_score, _row_score

    assert _coerce_score(float("nan")) is None
    assert _coerce_score(float("inf")) is None
    assert _coerce_score("nan") is None

    df = pd.DataFrame({"programmatic_check/value": [1.0, None, 0.5]})
    assert _row_score(df.iloc[1], "programmatic_check") is None, (
        "a NaN cell must read as unscored, or it poisons the aggregate"
    )


def _report_with_groundedness(rows: list[dict[str, Any]]):
    """Build a report over ``rows``, each ``{"correctness": v, "groundedness": ...}``.

    ``groundedness`` may be a float (scored), ``"error"`` (mlflow recorded a
    SCORER_ERROR) or ``None`` (the scorer abstained).
    """
    from anvil.eval.runner import _aggregate_report

    assessments = []
    ground_values: list[Any] = []
    for row in rows:
        row_assessments = [_value_assessment("correctness", row["correctness"])]
        if row["groundedness"] == "error":
            row_assessments.append(
                _scorer_error_assessment("retrieval_groundedness", "No retrieval context found.")
            )
            ground_values.append(None)
        elif row["groundedness"] is None:
            ground_values.append(None)
        else:
            row_assessments.append(_value_assessment("retrieval_groundedness", row["groundedness"]))
            ground_values.append(row["groundedness"])
        assessments.append(row_assessments)

    df = pd.DataFrame(
        {
            "trace_id": [f"t{i}" for i in range(len(rows))],
            "correctness/value": [r["correctness"] for r in rows],
            "retrieval_groundedness/value": ground_values,
            "assessments": assessments,
        }
    )
    return _aggregate_report(
        result_df=df,
        metrics={},
        scorer_names=["correctness", "retrieval_groundedness"],
        aggregate_scorer_names=["correctness", "retrieval_groundedness"],
        weights={"correctness": 1.0, "retrieval_groundedness": 1.0},
        examples=[_gold(f"g{i}", expected_doc_ids=["d"]) for i in range(len(rows))],
        run_id="run-1",
        experiment_id="exp-1",
        mode="quick",
    )


def test_per_judge_counts_separate_scored_errored_and_abstained_rows() -> None:
    report = _report_with_groundedness(
        [
            {"correctness": 1.0, "groundedness": 1.0},
            {"correctness": 1.0, "groundedness": "error"},
            {"correctness": 0.0, "groundedness": None},
            {"correctness": 1.0, "groundedness": 0.0},
        ]
    )
    assert report.per_judge_assessed["retrieval_groundedness"] == 2
    assert report.per_judge_errors["retrieval_groundedness"] == 1
    assert report.per_judge_assessed["correctness"] == 4
    assert report.per_judge_errors["correctness"] == 0
    # The abstained row is in neither count — not applicable is not broken.
    assert (
        report.per_judge_assessed["retrieval_groundedness"]
        + report.per_judge_errors["retrieval_groundedness"]
        == 3
    )
    # And the judge's mean is over the two rows that produced a score.
    assert report.per_judge["retrieval_groundedness"] == pytest.approx(0.5)


def test_a_bucket_omits_a_scorer_that_does_not_apply_within_it() -> None:
    """Found in the regenerated baseline, and the same bug in the output the
    optimizer steers by.

    ``_mean([])`` returns 0.0, so the ``out_of_scope`` bucket published
    ``retrieval_groundedness: 0.0`` — indistinguishable from "completely
    ungrounded on refusals" — for rows that have no groundedness verdict at all.
    ``prompts/anvil-round.md`` tells the optimizer to target a failure cluster it
    reads in ``per_bucket``, so that fabricated zero aims the next mutation at
    making the agent retrieve on questions it is supposed to refuse.
    """
    from anvil.eval.runner import _aggregate_report

    df = pd.DataFrame(
        {
            "trace_id": ["t0", "t1"],
            "correctness/value": [1.0, 0.0],
            "retrieval_groundedness/value": ["yes", None],
            "assessments": [
                [
                    _value_assessment("correctness", 1.0),
                    _value_assessment("retrieval_groundedness", "yes"),
                ],
                [_value_assessment("correctness", 0.0)],
            ],
        }
    )
    examples = [
        _gold("g0", expected_doc_ids=["d"]),
        _gold("g1", expected_doc_ids=[], should_refuse=True),
    ]
    report = _aggregate_report(
        result_df=df,
        metrics={},
        scorer_names=["correctness", "retrieval_groundedness"],
        aggregate_scorer_names=["correctness", "retrieval_groundedness"],
        weights={"correctness": 1.0, "retrieval_groundedness": 1.0},
        examples=examples,
        run_id="run-1",
        experiment_id="exp-1",
        mode="quick",
    )
    assert "retrieval_groundedness" not in report.per_bucket["out_of_scope"], (
        "a scorer with no verdict in a bucket must be absent, not 0.0"
    )
    assert report.per_bucket["out_of_scope"]["correctness"] == pytest.approx(0.0)
    assert report.per_bucket["direct"]["retrieval_groundedness"] == pytest.approx(1.0)


def test_scorer_errors_are_recorded_with_the_case_they_broke_on() -> None:
    """A count alone does not survive contact with debugging six rounds later."""
    report = _report_with_groundedness(
        [
            {"correctness": 1.0, "groundedness": 1.0},
            {"correctness": 1.0, "groundedness": "error"},
        ]
    )
    assert len(report.scorer_errors) == 1
    entry = report.scorer_errors[0]
    assert entry["scorer"] == "retrieval_groundedness"
    assert entry["example_id"] == "g1"
    assert entry["trace_id"] == "t1"
    assert "SCORER_ERROR" in entry["error_message"]


def test_a_prediction_error_is_not_also_counted_as_a_scorer_error() -> None:
    """A row whose prediction failed was never offered to the scorers, so
    counting it against a judge would blame the judge for the gateway."""
    from anvil.eval.runner import _aggregate_report

    df = pd.DataFrame(
        {
            "trace_id": ["t0", "t1"],
            "correctness/value": [1.0, None],
            "assessments": [[_value_assessment("correctness", 1.0)], []],
        }
    )
    report = _aggregate_report(
        result_df=df,
        metrics={},
        scorer_names=["correctness"],
        aggregate_scorer_names=["correctness"],
        weights={"correctness": 1.0},
        examples=[_gold("g0", expected_doc_ids=["d"]), _gold("g1", expected_doc_ids=["d"])],
        run_id="run-1",
        experiment_id="exp-1",
        mode="quick",
        errored={"t1": "gateway 429"},
    )
    assert report.n_errors == 1
    assert report.per_judge_errors["correctness"] == 0
    assert report.per_judge_assessed["correctness"] == 1
    assert report.scorer_errors == []


# ---------------------------------------------------------------------------
# The per-judge floor
# ---------------------------------------------------------------------------


def _report(**overrides: Any):
    from anvil.eval.runner import EvalReport

    kwargs: dict[str, Any] = {
        "aggregate": 0.9,
        "per_judge": {"correctness": 0.9, "retrieval_groundedness": 1.0},
        "per_bucket": {},
        "failures": [],
        "run_id": "run-x",
        "experiment_id": "exp-x",
        "n_rows": 8,
        "mode": "quick",
        "scorers": ["correctness", "retrieval_groundedness"],
        "evaluated_at": "2026-08-22T12:00:00+00:00",
    }
    kwargs.update(overrides)
    return EvalReport(**kwargs)


def test_a_judge_that_broke_on_most_of_its_attempts_is_unjudgeable() -> None:
    """The hole this closes: every prediction succeeded, every row is in the
    frame, the run-level guards see nothing — and one judge's contribution to
    the aggregate is a single row's score."""
    from anvil.eval.judgeability import unjudgeable_reason

    report = _report(
        per_judge_assessed={"correctness": 8, "retrieval_groundedness": 1},
        per_judge_errors={"correctness": 0, "retrieval_groundedness": 5},
    )
    reason = unjudgeable_reason(report, min_scorable_rows=4)
    assert "retrieval_groundedness" in reason
    assert "5 of the 6" in reason


def test_a_judge_that_merely_applies_to_few_rows_is_fine() -> None:
    """The floor must not fire on correct usage.

    Groundedness applies only where the golden set names ``expected_doc_ids``, so
    it legitimately scores 6 of 8 quick-mode rows and abstains on 2. Measuring
    the floor against the run's row count would make a correct eval unjudgeable —
    and a guard that fires on correct usage gets switched off.
    """
    from anvil.eval.judgeability import unjudgeable_reason

    report = _report(
        per_judge_assessed={"correctness": 8, "retrieval_groundedness": 2},
        per_judge_errors={"correctness": 0, "retrieval_groundedness": 0},
    )
    assert unjudgeable_reason(report, min_scorable_rows=4) == ""


def test_a_judge_with_a_few_errors_under_the_ceiling_is_judgeable() -> None:
    """Errors are not automatically disqualifying. One of eight is 0.125, inside
    the 0.20 ceiling, and leaves seven assessed."""
    from anvil.eval.judgeability import unjudgeable_reason

    report = _report(
        per_judge_assessed={"correctness": 8, "retrieval_groundedness": 7},
        per_judge_errors={"correctness": 0, "retrieval_groundedness": 1},
    )
    assert unjudgeable_reason(report, max_error_rate=0.2, min_scorable_rows=4) == ""


def test_a_judge_error_rate_over_the_ceiling_is_unjudgeable_even_above_the_floor() -> None:
    """The floor alone is too blunt one level down, exactly as it is at the run
    level: ``min(4, 8)`` is cleared by 4 assessed rows, so a judge failing half
    its invocations — the live symptom that started all of this — would pass a
    floor-only check. 4 errors of 8 is 0.50 against a 0.20 ceiling.
    """
    from anvil.eval.judgeability import unjudgeable_reason

    report = _report(
        per_judge_assessed={"retrieval_groundedness": 4},
        per_judge_errors={"retrieval_groundedness": 4},
    )
    reason = unjudgeable_reason(report, max_error_rate=0.2, min_scorable_rows=4)
    assert "0.50 exceeds ceiling 0.20" in reason


def test_the_per_judge_floor_is_capped_at_what_the_judge_attempted() -> None:
    """A judge that attempted 3 rows and scored all 3 is fine even though 3 is
    below the floor of 4 — same reasoning as the run-level cap against
    ``n_attempted``."""
    from anvil.eval.judgeability import unjudgeable_reason

    report = _report(
        per_judge_assessed={"retrieval_groundedness": 3},
        per_judge_errors={"retrieval_groundedness": 0},
    )
    assert unjudgeable_reason(report, min_scorable_rows=4) == ""


def test_run_level_checks_still_take_precedence() -> None:
    """Ordering matters for legibility: "the gateway was down" is a more useful
    thing to be told than "one judge is short of cases"."""
    from anvil.eval.judgeability import unjudgeable_reason

    report = _report(
        n_errors=6,
        per_judge_assessed={"retrieval_groundedness": 1},
        per_judge_errors={"retrieval_groundedness": 5},
    )
    reason = unjudgeable_reason(report, max_error_rate=0.2, min_scorable_rows=4)
    assert "unmeasured rate" in reason


# ---------------------------------------------------------------------------
# Fingerprint: semantics changed without the config changing
# ---------------------------------------------------------------------------


def test_semantics_version_is_in_the_groundedness_fingerprint() -> None:
    """A cached baseline measured before the applicability rule is not
    comparable to one measured after it, and every config field is identical
    across the change — so only a semantics version can say so."""
    import json

    from anvil.eval.cache import compute_scorer_fingerprint
    from anvil.eval.scorers import GROUNDEDNESS_SCORER_NAME, SCORER_SEMANTICS_VERSIONS
    from anvil.runtime.models import ScorerConfig

    fp = json.loads(compute_scorer_fingerprint([ScorerConfig(name=GROUNDEDNESS_SCORER_NAME)]))
    assert fp[0]["semantics"] == SCORER_SEMANTICS_VERSIONS[GROUNDEDNESS_SCORER_NAME]


def _cached(**overrides: Any):
    from anvil.eval.cache import CachedBaseline

    kwargs: dict[str, Any] = {
        "scaffold_commit_sha": "abc",
        "evaluated_at": "2026-04-28T03:42:00+00:00",
        "mode": "quick",
        "scorers": ["correctness", "retrieval_groundedness"],
        "runtime_endpoint": "databricks-claude-sonnet-4-6",
        "judge_endpoint": "databricks-claude-sonnet-4-6",
        "aggregate": 0.74,
    }
    kwargs.update(overrides)
    return CachedBaseline(**kwargs)


def test_a_fingerprintless_baseline_is_refused_for_a_versioned_scorer() -> None:
    """The gap the semantics version left open, and the only baseline on disk.

    ``eval/runs/baseline.json`` predates fingerprinting, so the
    backward-compatibility exemption waved it straight through — meaning the
    version bump protected every baseline except the one that actually needed
    invalidating. Its ``per_bucket`` still records ``out_of_scope:
    {retrieval_groundedness: 0.0}``, a bucket that now carries no groundedness
    value at all.
    """
    from anvil.eval.cache import compute_scorer_fingerprint, is_compatible
    from anvil.runtime.models import ScorerConfig

    fingerprint = compute_scorer_fingerprint(
        [ScorerConfig(name="correctness"), ScorerConfig(name="retrieval_groundedness")]
    )
    assert not is_compatible(
        _cached(scorer_fingerprint=""),
        mode="quick",
        scorers=["correctness", "retrieval_groundedness"],
        runtime_endpoint="databricks-claude-sonnet-4-6",
        judge_endpoint="databricks-claude-sonnet-4-6",
        scorer_fingerprint=fingerprint,
    )


def test_a_fingerprintless_baseline_still_works_for_unversioned_scorers() -> None:
    """The exemption survives where it was justified: a scorer whose meaning has
    never changed does not need a fingerprint to be comparable."""
    from anvil.eval.cache import compute_scorer_fingerprint, is_compatible
    from anvil.runtime.models import ScorerConfig

    fingerprint = compute_scorer_fingerprint([ScorerConfig(name="refusal_appropriateness")])
    assert is_compatible(
        _cached(scorers=["refusal_appropriateness"], scorer_fingerprint=""),
        mode="quick",
        scorers=["refusal_appropriateness"],
        runtime_endpoint="databricks-claude-sonnet-4-6",
        judge_endpoint="databricks-claude-sonnet-4-6",
        scorer_fingerprint=fingerprint,
    )


def test_the_gate_and_is_compatible_share_one_definition() -> None:
    """``loop/round.py`` used to inline its own copy of the fingerprint rule, so
    fixing ``is_compatible`` left the check that actually gates keep/revert on the
    old behaviour. Both now call ``scorer_incomparability_reason``, and this
    asserts they cannot disagree again."""
    import inspect

    from anvil.eval.cache import is_compatible, scorer_incomparability_reason
    from anvil.loop import round as round_mod

    assert "scorer_incomparability_reason" in inspect.getsource(round_mod.run_round)
    assert "scorer_incomparability_reason" in inspect.getsource(is_compatible)

    # And the shared rule refuses the fingerprintless baseline both paths see.
    reason = scorer_incomparability_reason(
        _cached(scorer_fingerprint=""),
        scorers=["correctness", "retrieval_groundedness"],
        scorer_fingerprint="anything",
    )
    assert "retrieval_groundedness" in reason


def test_unversioned_scorers_keep_their_old_fingerprint() -> None:
    """Bumping one scorer's semantics must not invalidate baselines for configs
    that do not use it."""
    import json

    from anvil.eval.cache import compute_scorer_fingerprint
    from anvil.runtime.models import ScorerConfig

    fp = json.loads(compute_scorer_fingerprint([ScorerConfig(name="refusal_appropriateness")]))
    assert "semantics" not in fp[0]


def test_the_judge_model_change_is_versioned_into_the_fingerprint() -> None:
    """Issue #13 moved correctness and groundedness onto the configured endpoint.

    The config string did not change, so only a semantics bump stops a
    baseline measured by the old implicit model from silently becoming the
    bar a 50-round run chases.
    """
    import json

    from anvil.eval.cache import compute_scorer_fingerprint
    from anvil.runtime.models import ScorerConfig

    fp = json.loads(compute_scorer_fingerprint([ScorerConfig(name="correctness")]))
    assert fp[0]["semantics"] == 1
    fp = json.loads(compute_scorer_fingerprint([ScorerConfig(name="retrieval_groundedness")]))
    assert fp[0]["semantics"] == 5
