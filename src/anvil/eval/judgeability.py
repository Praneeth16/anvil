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
2. **Too many errors.** The surviving mean is honest but measured on too little.
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

    ``min_scorable_rows`` is capped at the run's own row count, so the floor
    asks for "at least 4 assessed cases, or all of them if the run has fewer".
    Otherwise a deliberately small mode (a 2-row smoke eval) could never be
    judged, and a guard that fires on correct usage gets switched off.
    """
    if report.n_unattributed_errors:
        return (
            f"{report.n_unattributed_errors} eval error(s) could not be attributed "
            "to a result row, so their scores are still in the aggregate — the "
            "exclusion this comparison relies on did not happen"
        )
    if report.error_rate > max_error_rate:
        return (
            f"error rate {report.error_rate:.2f} exceeds ceiling {max_error_rate:.2f} "
            f"({report.n_errors}/{report.n_rows} cases never assessed)"
        )
    floor = min(min_scorable_rows, report.n_rows)
    if report.n_scorable < floor:
        return (
            f"only {report.n_scorable} of {report.n_rows} cases were assessed, "
            f"below the floor of {floor} — the aggregate is a mean over too few "
            "cases to compare"
        )
    return ""


def unjudgeable_reason_for(report: EvalReport, cfg: EvalConfig) -> str:
    """:func:`unjudgeable_reason` with the thresholds read off an eval config."""
    return unjudgeable_reason(
        report,
        max_error_rate=cfg.max_error_rate,
        min_scorable_rows=cfg.min_scorable_rows,
    )
