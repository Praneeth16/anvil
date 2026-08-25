"""Evaluating a candidate more than once, and combining the results honestly.

The paired sign test in :mod:`anvil.eval.significance` is free but blunt: at
twelve rows, a real improvement has to flip five or six rows before the test can
call it. Most useful mutations do not. The reason is not the test, it is the
measurement — a single per-row score from an LLM judge is one draw from a
distribution nobody has characterised, and the judge disagrees with itself.

Replication attacks that directly. Score the same row *K* times and average, and
the per-row estimate's variance falls by roughly *K*; rows that were flipping
because the judge was inconsistent stop flipping, and rows that flip because the
agent actually changed keep flipping. The test then sees the effect it could not
see before, without changing what the test is.

It is opt-in (``gate.replicates``, default ``1``) because it costs exactly what
it sounds like: *K* times the eval spend per round. At fifty rounds that is a
budget decision, not a default.

**What this deliberately does not do.** It does not average away an
infrastructure failure. A row that errored in one replicate is absent from that
replicate's ``per_row``, and averaging over the replicates that *did* score it
would quietly report a mean over a different sample than the row next to it. So
the merged report carries the per-replicate counts, and every judgeability check
runs against the merged report exactly as it would against a single one.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import replace

from anvil.eval.runner import EvalReport

logger = logging.getLogger(__name__)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def merge_reports(reports: Sequence[EvalReport]) -> EvalReport:
    """Combine replicate reports of the *same* candidate into one.

    Aggregates and per-judge values are averaged across replicates. Per-row
    scores are averaged per ``(example_id, scorer)`` over the replicates that
    produced a value for that pair — which is the point of replicating, and also
    the one place a reader can be misled, so the counts below record it.

    Error and drop counts are **summed**, not averaged. A run where one replicate
    lost six rows is not a run that lost two; the judgeability floor has to see
    the real total or replication becomes a way to launder a degraded run into a
    healthy-looking average.
    """
    if not reports:
        raise ValueError("merge_reports needs at least one report")
    if len(reports) == 1:
        return reports[0]

    base = reports[0]

    per_judge: dict[str, float] = {}
    for name in {n for r in reports for n in r.per_judge}:
        values = [r.per_judge[name] for r in reports if name in r.per_judge]
        per_judge[name] = _mean(values)

    per_bucket: dict[str, dict[str, float]] = {}
    for bucket in {b for r in reports for b in r.per_bucket}:
        merged: dict[str, float] = {}
        for name in {n for r in reports for n in r.per_bucket.get(bucket, {})}:
            values = [
                r.per_bucket[bucket][name]
                for r in reports
                if name in r.per_bucket.get(bucket, {})
            ]
            merged[name] = _mean(values)
        per_bucket[bucket] = merged

    per_row: dict[str, dict[str, float]] = {}
    for example_id in {e for r in reports for e in r.per_row}:
        merged_row: dict[str, float] = {}
        for name in {n for r in reports for n in r.per_row.get(example_id, {})}:
            values = [
                r.per_row[example_id][name]
                for r in reports
                if name in r.per_row.get(example_id, {})
            ]
            merged_row[name] = _mean(values)
        if merged_row:
            per_row[example_id] = merged_row

    per_judge_assessed: dict[str, int] = {}
    per_judge_errors: dict[str, int] = {}
    for name in {n for r in reports for n in r.per_judge_assessed} | {
        n for r in reports for n in r.per_judge_errors
    }:
        per_judge_assessed[name] = sum(r.per_judge_assessed.get(name, 0) for r in reports)
        per_judge_errors[name] = sum(r.per_judge_errors.get(name, 0) for r in reports)

    cost_metrics: dict[str, float] = {}
    for key in {k for r in reports for k in r.cost_metrics}:
        cost_metrics[key] = sum(r.cost_metrics.get(key, 0.0) for r in reports)
    cost_metrics["replicates"] = float(len(reports))

    return replace(
        base,
        aggregate=_mean([r.aggregate for r in reports]),
        per_judge=per_judge,
        per_bucket=per_bucket,
        per_row=per_row,
        # ``n_rows`` stays the row count of one replicate, not the sum. It is the
        # size of the golden-set subset, and the judgeability floor is a statement
        # about how many distinct questions were measured -- replicating twelve
        # rows twice does not make it a twenty-four-question eval.
        n_rows=base.n_rows,
        n_errors=sum(r.n_errors for r in reports),
        n_unattributed_errors=sum(r.n_unattributed_errors for r in reports),
        n_dropped_rows=sum(r.n_dropped_rows for r in reports),
        per_judge_assessed=per_judge_assessed,
        per_judge_errors=per_judge_errors,
        failures=[f for r in reports for f in r.failures],
        errors=[e for r in reports for e in r.errors],
        scorer_errors=[e for r in reports for e in r.scorer_errors],
        trace_ids=[t for r in reports for t in r.trace_ids],
        cost_metrics=cost_metrics,
    )


def evaluate_replicated(
    evaluate: Callable[[], EvalReport],
    *,
    replicates: int,
) -> EvalReport:
    """Call ``evaluate`` ``replicates`` times and merge the reports.

    ``replicates=1`` calls once and returns that report untouched, so the default
    path is byte-identical to not having this module.
    """
    if replicates < 1:
        raise ValueError(f"replicates must be at least 1, got {replicates}")
    if replicates == 1:
        return evaluate()
    reports = []
    for i in range(replicates):
        logger.info("eval replicate %d/%d", i + 1, replicates)
        reports.append(evaluate())
    return merge_reports(reports)
