"""The paired gate: does a delta survive contact with judge noise?

Two healthy runs of the same scaffold, the same rows and the same model scored
0.875 and 0.722. Every per-round gain the loop has produced was 0.03-0.06. So
the old gate -- promote on any positive delta -- was reading noise as signal
about as often as not.

These tests pin the three behaviours that make the fix a gate rather than a
threshold:

* the paired test can only ever *veto* a KEEP, never rescue a REVERT;
* "the test could not run" and "the test ran and says no" are different
  outcomes, because one is a missing baseline field and the other is evidence;
* replication changes the measurement without changing the test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anvil.eval.significance import (
    MIN_DISCORDANT_PAIRS,
    one_sided_sign_test_p,
    paired_sign_test,
    row_aggregate,
)
from anvil.loop.decision import Decision
from anvil.loop.frontier import AGGREGATE_KEY, gate_decision

_W = {"correctness": 1.0}
_NAMES = ["correctness"]


def _rows(values: dict[str, float]) -> dict[str, dict[str, float]]:
    return {k: {"correctness": v} for k, v in values.items()}


# ---------------------------------------------------------------------------
# The exact binomial tail
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("improved", "regressed", "expected"),
    [
        (0, 0, 1.0),
        (1, 0, 0.5),
        (2, 0, 0.25),
        (3, 0, 0.125),
        (4, 0, 0.0625),
        (5, 0, 0.03125),
        (6, 1, 0.0625),
        (3, 3, 0.65625),
        (0, 5, 1.0),
    ],
)
def test_sign_test_p_is_the_exact_binomial_tail(improved, regressed, expected):
    """Exact, not a normal approximation -- which misbehaves at these sizes."""
    assert one_sided_sign_test_p(improved, regressed) == pytest.approx(expected)


@pytest.mark.unit
def test_four_discordant_pairs_cannot_reach_alpha():
    """Why MIN_DISCORDANT_PAIRS is 5 and not a taste-based number.

    A clean sweep of four still gives p = 0.0625 > 0.05, so at four discordant
    pairs no result is reachable and "not significant" would be a verdict the
    data could never have contradicted.
    """
    assert one_sided_sign_test_p(4, 0) > 0.05
    assert one_sided_sign_test_p(5, 0) <= 0.05
    assert MIN_DISCORDANT_PAIRS == 5


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ties_are_excluded_not_counted_as_evidence():
    """A row both runs get right says nothing about the mutation.

    Counting ties would make the verdict depend on how easy the golden set is
    rather than on what changed.
    """
    base = _rows({f"q{i}": 1.0 for i in range(20)} | {f"r{i}": 0.0 for i in range(6)})
    cand = _rows({f"q{i}": 1.0 for i in range(20)} | {f"r{i}": 1.0 for i in range(6)})

    result = paired_sign_test(base, cand, weights=_W, aggregate_scorer_names=_NAMES)

    assert (result.n_improved, result.n_regressed, result.n_tied) == (6, 0, 20)
    assert result.significant


@pytest.mark.unit
def test_a_row_missing_from_either_run_is_dropped_from_both():
    """An errored row is not evidence about the mutation (D7, at the gate).

    Scoring it as zero on the side that lost it would manufacture a difference
    out of an infrastructure failure.
    """
    base = _rows({"a": 0.0, "b": 0.0, "c": 1.0})
    cand = _rows({"a": 1.0, "b": 1.0})

    result = paired_sign_test(base, cand, weights=_W, aggregate_scorer_names=_NAMES)

    assert result.n_pairs == 2
    assert result.n_improved == 2


@pytest.mark.unit
def test_row_aggregate_weights_only_the_scorers_that_applied():
    """An inapplicable scorer is absent, not zero (D10).

    Treating it as zero would make refusal rows -- where groundedness has
    nothing to be grounded in -- look uniformly worse than answerable ones, and
    the gap would move with the bucket mix rather than with the agent.
    """
    weights = {"correctness": 1.0, "retrieval_groundedness": 1.0}
    names = ["correctness", "retrieval_groundedness"]

    both = row_aggregate({"correctness": 1.0, "retrieval_groundedness": 0.0}, weights, names)
    only_one = row_aggregate({"correctness": 1.0}, weights, names)

    assert both == pytest.approx(0.5)
    assert only_one == pytest.approx(1.0)
    assert row_aggregate({}, weights, names) is None


@pytest.mark.unit
def test_regression_is_never_significant_however_lopsided():
    """One-sided, deliberately: the gate asks "better", not "different"."""
    base = _rows({f"q{i}": 1.0 for i in range(8)})
    cand = _rows({f"q{i}": 0.0 for i in range(8)})

    result = paired_sign_test(base, cand, weights=_W, aggregate_scorer_names=_NAMES)

    assert result.n_regressed == 8
    assert not result.significant


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

_BASE_SCORES = {AGGREGATE_KEY: 0.50}


def _gate(tmp_path: Path, *, mutated: float, base_rows, cand_rows, **kw):
    return gate_decision(
        repo_root=tmp_path,
        gate_type="frontier",
        epsilon=0.0,
        pareto=False,
        baseline_scores=_BASE_SCORES,
        baseline_aggregate=_BASE_SCORES[AGGREGATE_KEY],
        mutated_scores={AGGREGATE_KEY: mutated},
        mutated_aggregate=mutated,
        action_kind="add_rule",
        eval_failed=False,
        parse_status="ok",
        baseline_per_row=base_rows,
        mutated_per_row=cand_rows,
        weights=_W,
        aggregate_scorer_names=_NAMES,
        **kw,
    )


@pytest.mark.unit
def test_the_gate_defaults_to_running_the_paired_test():
    """If the default were "none" the feature would ship switched off."""
    import inspect

    from anvil.runtime.models import GateConfig

    assert GateConfig().test == "paired"
    assert inspect.signature(gate_decision).parameters["gate_test"].default == "paired"


@pytest.mark.unit
def test_a_significant_improvement_is_kept(tmp_path: Path):
    base = _rows({f"q{i}": 0.0 for i in range(6)})
    cand = _rows({f"q{i}": 1.0 for i in range(6)})

    decision, _f, paired = _gate(tmp_path, mutated=0.9, base_rows=base, cand_rows=cand)

    assert decision == Decision.KEEP
    assert paired is not None and paired.significant


@pytest.mark.unit
def test_an_aggregate_gain_that_is_only_noise_is_reverted(tmp_path: Path):
    """The whole point: the aggregate rose, the rows say it is a coin flip.

    Three rows improved, two regressed -- p = 0.5. The old gate promoted this on
    the strength of the aggregate delta alone.
    """
    base = _rows({"a": 0.0, "b": 0.0, "c": 0.0, "d": 1.0, "e": 1.0, "f": 1.0, "g": 1.0})
    cand = _rows({"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.0, "e": 0.0, "f": 1.0, "g": 1.0})

    decision, _f, paired = _gate(tmp_path, mutated=0.72, base_rows=base, cand_rows=cand)

    assert decision == Decision.REVERT
    assert paired is not None
    assert not paired.significant
    assert (paired.n_improved, paired.n_regressed) == (3, 2)


@pytest.mark.unit
def test_too_few_disagreements_reverts_and_names_the_knob(tmp_path: Path):
    """Underpowered is not "no", it is "not measurable here" -- and it reverts.

    Promoting on evidence that cannot distinguish the mutation from noise is the
    behaviour being fixed, so the conservative answer is the correct one. The
    reason has to name `gate.replicates`, or an operator watching rounds revert
    has no way to know what to do about it.
    """
    base = _rows({"a": 0.0, "b": 1.0, "c": 1.0, "d": 1.0})
    cand = _rows({"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0})

    decision, _f, paired = _gate(tmp_path, mutated=0.75, base_rows=base, cand_rows=cand)

    assert decision == Decision.REVERT
    assert paired is not None and paired.underpowered
    assert "gate.replicates" in paired.reason


@pytest.mark.unit
def test_no_pairable_rows_leaves_the_frontier_decision_standing(tmp_path: Path):
    """A baseline written before per-row scores must not brick the loop.

    Same "empty means unchecked" rule the scorer and dataset fingerprints use.
    Reverting instead would be a migration disguised as a gate.
    """
    decision, _f, paired = _gate(tmp_path, mutated=0.9, base_rows={}, cand_rows={})

    assert decision == Decision.KEEP
    assert paired is not None
    assert paired.n_pairs == 0
    assert "predates per-row scores" in paired.reason


@pytest.mark.unit
def test_the_paired_test_cannot_rescue_a_regression(tmp_path: Path):
    """A veto, never a promotion.

    The frontier rejects this for regressing; a lopsided row-level win must not
    override that. Direction and significance are different questions.
    """
    base = _rows({f"q{i}": 0.0 for i in range(8)})
    cand = _rows({f"q{i}": 1.0 for i in range(8)})

    decision, _f, paired = _gate(tmp_path, mutated=0.10, base_rows=base, cand_rows=cand)

    assert decision == Decision.REVERT
    assert paired is None  # never even consulted


@pytest.mark.unit
def test_gate_test_none_restores_the_legacy_behaviour(tmp_path: Path):
    """The escape hatch has to actually be an escape hatch."""
    base = _rows({"a": 0.0, "b": 1.0, "c": 1.0, "d": 1.0})
    cand = _rows({"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0})

    decision, _f, paired = _gate(
        tmp_path, mutated=0.75, base_rows=base, cand_rows=cand, gate_test="none"
    )

    assert decision == Decision.KEEP
    assert paired is None


# ---------------------------------------------------------------------------
# Replication
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_replication_lets_the_same_test_see_a_smaller_effect():
    """What `gate.replicates` actually buys, demonstrated rather than asserted.

    Six rows where the candidate is genuinely better, but a noisy judge scores
    each row inconsistently. One replicate leaves most rows tied at the extremes
    and the test underpowered; averaging three replicates separates them.
    """
    from anvil.eval.replication import merge_reports
    from anvil.eval.runner import EvalReport

    def _report(per_row):
        return EvalReport(
            aggregate=0.0,
            per_judge={},
            per_bucket={},
            failures=[],
            run_id="r",
            experiment_id="e",
            n_rows=len(per_row),
            mode="quick",
            scorers=_NAMES,
            evaluated_at="2026-08-24T00:00:00+00:00",
            per_row=per_row,
        )

    ids = [f"q{i}" for i in range(6)]
    # Baseline: the judge says 0 every time. Candidate: the judge says 1 on two
    # of three replicates -- a real but noisily measured improvement.
    base = [_report(_rows(dict.fromkeys(ids, 0.0))) for _ in range(3)]
    # The unlucky replicate is FIRST, deliberately: a merge that took the first
    # replicate instead of the mean would pass this test with the order reversed,
    # and "averaged" is the entire claim replication rests on.
    cand = [
        _report(_rows(dict.fromkeys(ids, 0.0))),
        _report(_rows(dict.fromkeys(ids, 1.0))),
        _report(_rows(dict.fromkeys(ids, 1.0))),
    ]

    single = paired_sign_test(
        base[0].per_row, cand[0].per_row, weights=_W, aggregate_scorer_names=_NAMES
    )
    merged = paired_sign_test(
        merge_reports(base).per_row,
        merge_reports(cand).per_row,
        weights=_W,
        aggregate_scorer_names=_NAMES,
    )

    # The unlucky single replicate sees no difference at all.
    assert single.n_improved == 0
    assert not single.significant
    # Averaged, every row is 0.667 against 0.0 and the test can see it.
    assert merged.n_improved == 6
    assert merged.significant


@pytest.mark.unit
def test_replication_sums_errors_rather_than_averaging_them():
    """Replication must not launder a degraded run into a healthy average.

    A run where one replicate lost six rows is not a run that lost two, and the
    judgeability floor has to see the real total.
    """
    from anvil.eval.replication import merge_reports
    from anvil.eval.runner import EvalReport

    def _report(n_errors, n_dropped):
        return EvalReport(
            aggregate=0.5,
            per_judge={},
            per_bucket={},
            failures=[],
            run_id="r",
            experiment_id="e",
            n_rows=12,
            mode="standard",
            scorers=_NAMES,
            evaluated_at="2026-08-24T00:00:00+00:00",
            n_errors=n_errors,
            n_dropped_rows=n_dropped,
        )

    merged = merge_reports([_report(0, 0), _report(6, 2), _report(0, 0)])

    assert merged.n_errors == 6
    assert merged.n_dropped_rows == 2
    # n_rows stays the size of the golden-set subset: replicating 12 rows three
    # times is not a 36-question eval, and the floor is about distinct questions.
    assert merged.n_rows == 12
    assert merged.cost_metrics["replicates"] == 3.0


@pytest.mark.unit
def test_single_replicate_returns_the_report_untouched():
    """The default path must be identical to not having replication at all."""
    from anvil.eval.replication import evaluate_replicated
    from anvil.eval.runner import EvalReport

    report = EvalReport(
        aggregate=0.5,
        per_judge={},
        per_bucket={},
        failures=[],
        run_id="r",
        experiment_id="e",
        n_rows=8,
        mode="quick",
        scorers=_NAMES,
        evaluated_at="2026-08-24T00:00:00+00:00",
    )
    calls = []

    def _evaluate():
        calls.append(1)
        return report

    assert evaluate_replicated(_evaluate, replicates=1) is report
    assert len(calls) == 1


@pytest.mark.unit
def test_replicates_config_drives_the_number_of_evals():
    from anvil.eval.replication import evaluate_replicated
    from anvil.eval.runner import EvalReport

    calls = []

    def _evaluate():
        calls.append(1)
        return EvalReport(
            aggregate=0.5,
            per_judge={},
            per_bucket={},
            failures=[],
            run_id="r",
            experiment_id="e",
            n_rows=8,
            mode="quick",
            scorers=_NAMES,
            evaluated_at="2026-08-24T00:00:00+00:00",
        )

    evaluate_replicated(_evaluate, replicates=3)
    assert len(calls) == 3

@pytest.mark.unit
def test_replication_averages_the_judge_and_bucket_columns():
    """Averaged, not taken from the first replicate.

    The per-row merge was covered; these two columns were not, and a merge that
    returned the first replicate's numbers passed the whole suite.
    """
    from anvil.eval.replication import merge_reports
    from anvil.eval.runner import EvalReport

    def _report(correctness):
        return EvalReport(
            aggregate=correctness,
            per_judge={"correctness": correctness},
            per_bucket={"direct": {"correctness": correctness}},
            failures=[],
            run_id="r",
            experiment_id="e",
            n_rows=4,
            mode="quick",
            scorers=_NAMES,
            evaluated_at="2026-08-24T00:00:00+00:00",
        )

    merged = merge_reports([_report(0.0), _report(1.0), _report(1.0)])

    assert merged.per_judge["correctness"] == pytest.approx(2 / 3)
    assert merged.per_bucket["direct"]["correctness"] == pytest.approx(2 / 3)
    assert merged.aggregate == pytest.approx(2 / 3)
