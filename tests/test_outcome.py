"""The failure/error distinction, and the invariants that keep it honest."""

from __future__ import annotations

import pytest

from anvil.eval.outcome import (
    Attempt,
    CaseOutcome,
    CaseRecord,
    summarize,
)

pytestmark = pytest.mark.unit


def test_a_failure_is_scorable_and_an_error_is_not():
    """The whole point: one belongs in the aggregate, the other does not."""
    failure = CaseRecord(case_id="g1", outcome=CaseOutcome.FAILURE, output="wrong answer")
    error = CaseRecord.errored("g2", TimeoutError("gateway timed out"))

    assert failure.scorable
    assert not error.scorable


def test_an_error_record_must_name_its_cause():
    """An error with no type is indistinguishable from a zero, which is the bug."""
    with pytest.raises(ValueError, match="must name its error_type"):
        CaseRecord(case_id="g1", outcome=CaseOutcome.ERROR)


def test_a_scorable_record_cannot_carry_an_error():
    """Makes the conflation unconstructible rather than merely discouraged."""
    with pytest.raises(ValueError, match="cannot carry an error_type"):
        CaseRecord(case_id="g1", outcome=CaseOutcome.OK, error_type="TimeoutError")


def test_errored_captures_the_exception():
    record = CaseRecord.errored("g1", ValueError("bad input"))

    assert record.outcome is CaseOutcome.ERROR
    assert record.error_type == "ValueError"
    assert record.error_message == "bad input"
    assert record.output == ""


def test_an_error_carries_no_output():
    """Distinct from the old behaviour of recording "" as the prediction.

    The empty string is still empty, but the outcome says the case was never
    assessed, so nothing downstream scores it.
    """
    record = CaseRecord.errored("g1", RuntimeError("boom"))
    assert record.output == ""
    assert not record.scorable


def test_retried_attempts_are_retained_even_when_the_case_succeeds():
    """A round that only worked on its third try is worth being able to see."""
    record = CaseRecord(
        case_id="g1",
        outcome=CaseOutcome.OK,
        output="right answer",
        attempts=(
            Attempt(error_type="RateLimitError", error_message="429", duration_ms=10),
            Attempt(error_type="RateLimitError", error_message="429", duration_ms=12),
        ),
    )

    assert record.scorable
    assert len(record.attempts) == 2


# -- the summary a round-level guard reads ---------------------------------


def test_summarize_counts_each_outcome():
    records = [
        CaseRecord(case_id="1", outcome=CaseOutcome.OK, output="a"),
        CaseRecord(case_id="2", outcome=CaseOutcome.FAILURE, output="b"),
        CaseRecord(case_id="3", outcome=CaseOutcome.FAILURE, output="c"),
        CaseRecord.errored("4", TimeoutError("t")),
        CaseRecord(case_id="5", outcome=CaseOutcome.SKIPPED),
        CaseRecord(case_id="6", outcome=CaseOutcome.INTERRUPTED),
    ]

    summary = summarize(records)

    assert summary.total == 6
    assert (summary.ok, summary.failure, summary.error) == (1, 2, 1)
    assert (summary.skipped, summary.interrupted) == (1, 1)
    assert summary.scorable == 3


def test_error_rate_drives_the_round_guard():
    records = [
        CaseRecord(case_id="1", outcome=CaseOutcome.OK, output="a"),
        CaseRecord.errored("2", TimeoutError("t")),
        CaseRecord.errored("3", TimeoutError("t")),
        CaseRecord(case_id="4", outcome=CaseOutcome.FAILURE, output="b"),
    ]

    assert summarize(records).error_rate == 0.5


def test_error_rate_of_an_empty_run_is_zero_not_a_crash():
    """A run with no cases is degenerate, not an infrastructure failure."""
    assert summarize([]).error_rate == 0.0
    assert summarize([]).total == 0


def test_a_run_of_pure_failures_has_no_error_rate():
    """The case that must not trip the guard: the agent is bad, the harness is fine."""
    records = [CaseRecord(case_id=str(i), outcome=CaseOutcome.FAILURE) for i in range(5)]

    summary = summarize(records)

    assert summary.error_rate == 0.0
    assert summary.failure == 5


def test_a_run_of_pure_errors_is_entirely_unscorable():
    """The case that must trip it: nothing was measured at all."""
    records = [CaseRecord.errored(str(i), ConnectionError("down")) for i in range(5)]

    summary = summarize(records)

    assert summary.error_rate == 1.0
    assert summary.scorable == 0
