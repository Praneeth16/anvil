"""Is this eval worth comparing to anything?

Excluding never-assessed cases from the aggregate (see
:mod:`anvil.eval.outcome`) fixes the *direction* of the bias -- a throttled
gateway no longer looks like a bad mutation -- but it does not fix the *sample*.
Three ways an excluded-errors report can still be untrustworthy, and every eval
path has to ask about all three or the exclusion makes things worse rather than
better:

1. **An error that could not be excluded.** The capture is keyed by ``trace_id``;
   an error whose row is not in ``result_df`` cannot have its score removed,
   because there is no row to remove. Its infrastructure zero is still in the
   mean, so the report does not do what it says it does.
2. **Too many unscored cases.** The surviving mean is honest but measured on too
   little. Counted as errors *plus* rows that vanished from the result frame
   without erroring -- both are unscored, and only the first is visible in
   ``error_rate``.
3. **Too few surviving cases.** The dangerous one, and the reason a rate ceiling
   alone is not enough: a rate is relative, so raising the ceiling to ride out a
   flaky afternoon also permits the aggregate to become the score of one
   surviving row. Before exclusion, seven errors in eight rows scored ~0.12 and
   was reverted; after exclusion the same run can read 1.0 and *extend the
   frontier*. An absolute floor is the only instrument that catches that.

Every path that turns a report into a decision -- the round gate, the CLI exit
status, baseline generation, held-out finalization -- routes through
:func:`unjudgeable_reason`, so there is one definition of "judgeable" rather
than four that drift apart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from anvil.eval.runner import EvalReport
    from anvil.runtime.models import EvalConfig


def unjudgeable_reason(
    report: EvalReport,
    *,
    max_error_rate: float = 0.2,
    min_scorable_rows: int = 4,
) -> str:
    """Why ``report`` must not be compared to anything, or ``""`` if it may be.

    A string rather than a bool so the reason reaches the round record and the
    operator's terminal: "reverted" and "could not be measured" are different
    events, and six rounds later the difference is only recoverable if it was
    written down.

    ``min_scorable_rows`` is capped at the number of cases the run *attempted*,
    so the floor asks for "at least 4 assessed cases, or all of them if the run
    attempted fewer". Otherwise a deliberately small mode (a 2-row smoke eval)
    could never be judged, and a guard that fires on correct usage gets switched
    off. Capping against the *surviving* rows instead would let row loss lower
    the very bar it should trip.
    """
    if report.n_unattributed_errors:
        return (
            f"{report.n_unattributed_errors} eval error(s) could not be attributed "
            "to a result row, so their scores are still in the aggregate — the "
            "exclusion this comparison relies on did not happen"
        )
    # Reads unmeasured_rate, not error_rate: a row that vanished from the frame
    # without erroring is just as unscored as one that raised, and error_rate --
    # errors over SURVIVING rows -- cannot see it. Judging on error_rate let a
    # 20-row run lose 16 traces and still pass, because 4 scorable rows clears an
    # absolute floor of 4. A floor alone is too blunt; it is satisfiable by any
    # run large enough, which is every mode above ``quick``.
    if report.unmeasured_rate > max_error_rate:
        unscored = report.n_errors + report.n_dropped_rows
        detail = f"{report.n_errors} errored"
        if report.n_dropped_rows:
            detail += f", {report.n_dropped_rows} lost their trace"
        return (
            f"unmeasured rate {report.unmeasured_rate:.2f} exceeds ceiling "
            f"{max_error_rate:.2f} ({unscored}/{report.n_attempted} cases never "
            f"scored: {detail})"
        )
    # Capped at what the run ATTEMPTED, not at what survived into the frame.
    # Rows dropped for want of a trace are absent from ``n_rows``, so capping
    # against it would let row loss lower the very bar it should trip.
    floor = min(min_scorable_rows, report.n_attempted)
    if report.n_scorable < floor:
        dropped = (
            f", {report.n_dropped_rows} dropped for want of a trace"
            if report.n_dropped_rows
            else ""
        )
        return (
            f"only {report.n_scorable} of {report.n_attempted} cases were assessed"
            f"{dropped}, below the floor of {floor} — the aggregate is a mean over "
            "too few cases to compare"
        )
    return ""


def unjudgeable_reason_for(report: EvalReport, cfg: EvalConfig) -> str:
    """:func:`unjudgeable_reason` with the thresholds read off an eval config."""
    return unjudgeable_reason(
        report,
        max_error_rate=cfg.max_error_rate,
        min_scorable_rows=cfg.min_scorable_rows,
    )
