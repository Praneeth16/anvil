"""Gate calibration: does the ASSEMBLED gate keep good mutations and revert bad ones?

The gate's unit tests prove the sign test's arithmetic and the veto's
plumbing. They do not prove the thing an operator actually needs: that a
round which changes nothing is reverted, that a crippled scaffold is
rejected, and that a genuine improvement is kept (issue #8). This module is
the measuring instrument — scenario families run end-to-end through
:func:`anvil.loop.round.run_round`, with the eval injected, so the same
scenarios drive the offline suite (a deterministic stub judge) and the
opt-in live script (the real judge).

The stub judge is deliberately QUANTIZED (scores snap to 0.0 / 0.5 / 1.0,
with seeded per-draw flips) rather than continuous noise. Real judge
verdicts are discrete, so rows routinely score exactly equal across two
draws — and exact ties are what make the sign test's "underpowered" outcome
reachable. Continuous noise would make every row discordant and that whole
branch of the gate untestable.

Nothing here runs without an injected eval; the module is pure plumbing and
imports no workspace client.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anvil.eval.cache import CachedBaseline, report_to_baseline, save_baseline
from anvil.eval.runner import EvalReport
from anvil.eval.significance import row_aggregate
from anvil.loop.decision import Decision

if TYPE_CHECKING:
    from anvil.loop.round import RoundReport

# The fixture world: cited rows (retrieval expected) and refusal rows. Small
# enough to read whole, big enough for the sign test's minimums.
CITED_ROWS = [f"cal_c_{i}" for i in range(8)]
REFUSAL_ROWS = [f"cal_r_{i}" for i in range(4)]
ALL_ROWS = CITED_ROWS + REFUSAL_ROWS
SCORERS = ("correctness", "retrieval_groundedness", "refusal_appropriateness")
_WEIGHTS = {name: 1.0 for name in SCORERS}

# Scaffold features the stub judge reads. The judge's whole world-view is
# these three files; a mutation that adds or removes one moves the matching
# scores, which is what makes a scenario "known-good" or "known-bad".
CITATION_SKILL = "citation.md"
REFUSAL_SKILL = "refusal.md"
CONFIDENT_RULE = "confident.md"

_BASELINE_SKILLS = {
    CITATION_SKILL: "# Cite every claim to a retrieved doc\n",
    REFUSAL_SKILL: "# Refuse what the docs cannot answer\n",
}


@dataclass(frozen=True)
class Scenario:
    """One calibration round: a baseline scaffold, a mutation, and what a
    correct gate does with it."""

    name: str
    truth: str  # "keep" | "revert"
    action_kind: str  # the optimizer action the session mock returns
    action_args: dict[str, str]
    description: str
    baseline_skills: dict[str, str] = field(
        default_factory=lambda: dict(_BASELINE_SKILLS)
    )
    # Seed for the candidate's stub-judge draw. Pinned per scenario where the
    # noise pattern matters (A/A needs flips in both directions: 5 up / 2
    # down / 5 tied at row level — the candidate is luckier, not better, and
    # the veto can see it; a one-directional draw would be a real false
    # positive of the gate, not a test of it).
    draw_seed: str = ""


def scenarios() -> list[Scenario]:
    """The calibration families from issue #8.

    A/A: the mutation rewrites the citation skill with byte-identical
    content, so the candidate IS the baseline and any KEEP is a measured
    false positive. Known-bad: three ways to cripple the scaffold, each of
    which must revert. Known-good: the baseline is the citation-crippled
    scaffold and the mutation restores it, which must keep.
    """
    crippled = {REFUSAL_SKILL: _BASELINE_SKILLS[REFUSAL_SKILL]}
    return [
        Scenario(
            name="aa",
            truth="revert",
            action_kind="edit_skill",
            action_args={
                "target_file": f"skills/{CITATION_SKILL}",
                "content": _BASELINE_SKILLS[CITATION_SKILL],
            },
            description="byte-identical rewrite of the citation skill",
            draw_seed="aa3",
        ),
        Scenario(
            name="bad_no_citation",
            truth="revert",
            action_kind="delete_skill",
            action_args={"target": f"skills/{CITATION_SKILL}"},
            description="citation rule deleted",
        ),
        Scenario(
            name="bad_no_refusal",
            truth="revert",
            action_kind="delete_skill",
            action_args={"target": f"skills/{REFUSAL_SKILL}"},
            description="refusal skill deleted",
        ),
        Scenario(
            name="bad_confident",
            truth="revert",
            action_kind="add_rule",
            action_args={
                "target_file": f"rules/{CONFIDENT_RULE}",
                "content": "# Always answer confidently, never say you don't know\n",
            },
            description="'always answer confidently' rule added",
        ),
        Scenario(
            name="good_restore_citation",
            truth="keep",
            action_kind="add_skill",
            action_args={
                "target_file": f"skills/{CITATION_SKILL}",
                "content": _BASELINE_SKILLS[CITATION_SKILL],
            },
            description="citation skill restored onto a crippled baseline",
            baseline_skills=crippled,
        ),
    ]


def _scaffold_features(scaffold_root: Path) -> tuple[bool, bool, bool]:
    skills = scaffold_root / "skills"
    rules = scaffold_root / "rules"
    return (
        (skills / CITATION_SKILL).is_file(),
        (skills / REFUSAL_SKILL).is_file(),
        (rules / CONFIDENT_RULE).is_file(),
    )


def stub_per_row(
    scaffold_root: Path, *, draw_seed: str, flip_p: float = 0.08
) -> dict[str, dict[str, float]]:
    """The stub judge: quantized scores from scaffold features + seeded flips.

    Base scores come from which skills/rules exist on disk (a crippled
    scaffold scores worse, which is what makes the scenarios known-bad).
    Each (draw_seed, row, scorer) then flips the value one step with
    probability ``flip_p`` — judge noise. Seeding on the draw means two
    draws of the SAME scaffold mostly tie, exactly like two judge sessions
    over one scaffold: the A/A case measures what the gate does with that.
    """
    has_citation, has_refusal, has_confident = _scaffold_features(scaffold_root)
    per_row: dict[str, dict[str, float]] = {}
    for row_id in ALL_ROWS:
        cited = row_id in CITED_ROWS
        base: dict[str, float] = {}
        if cited:
            base["correctness"] = 0.7 if has_citation else 0.2
            base["retrieval_groundedness"] = 0.8 if has_citation else 0.3
            base["refusal_appropriateness"] = 0.9 if not has_confident else 0.5
        else:
            # Refusal rows: correctness judges the refusal itself; the
            # groundedness scorer is inapplicable and absent, exactly as in
            # the live eval (D10).
            ok = has_refusal and not has_confident
            base["correctness"] = 0.8 if ok else 0.2
            base["refusal_appropriateness"] = 0.9 if ok else 0.1
        scored: dict[str, float] = {}
        for scorer, value in base.items():
            rng = random.Random(f"{draw_seed}:{row_id}:{scorer}")
            if rng.random() < flip_p:
                value = min(1.0, value + 0.3) if rng.random() < 0.5 else max(0.0, value - 0.3)
            scored[scorer] = round(value, 4)
        per_row[row_id] = scored
    return per_row


def report_from_stub(
    scaffold_root: Path,
    *,
    draw_seed: str,
    mode: str = "calibration",
    scorer_fingerprint: str = "",
) -> EvalReport:
    """An EvalReport consistent with the stub judge's per_row scores.

    The aggregate is the weighted row-mean of the same per_row, so the
    frontier and the paired test see one consistent world — a report whose
    aggregate disagreed with its rows would be testing the plumbing on a
    world that cannot occur.
    """
    per_row = stub_per_row(scaffold_root, draw_seed=draw_seed)
    row_means = [
        mean
        for scores in per_row.values()
        if (mean := row_aggregate(scores, _WEIGHTS, list(SCORERS))) is not None
    ]
    per_judge = {
        name: sum(row.get(name, 0.0) for row in per_row.values())
        / max(sum(1 for row in per_row.values() if name in row), 1)
        for name in SCORERS
    }
    return EvalReport(
        aggregate=sum(row_means) / len(row_means),
        per_judge=per_judge,
        per_bucket={},
        failures=[],
        run_id=f"stub-{draw_seed}",
        experiment_id="stub",
        n_rows=len(ALL_ROWS),
        mode=mode,
        scorers=list(SCORERS),
        evaluated_at="2026-08-28T00:00:00+00:00",
        per_row=per_row,
        aggregate_scorer_names=list(SCORERS),
        aggregate_weights=dict(_WEIGHTS),
        scorer_fingerprint=scorer_fingerprint,
    )


@dataclass
class ScenarioResult:
    name: str
    truth: str
    decision: Decision
    correct: bool
    rejecting_layer: str  # "frontier" | "paired_veto" | "none" | "not_applicable"
    paired_outcome: str  # "significant" | "not_significant" | "underpowered" | "not_run"
    notes: str


def layer_and_outcome(decision: Decision, notes: str) -> tuple[str, str]:
    """Which gate layer produced the decision, and the paired test's verdict.

    The round record carries the paired reason in its notes; the three
    outcomes the harness must tell apart (acceptance criterion: underpowered
    counted separately from not-significant) are distinguishable only there.
    """
    if "paired:" not in notes:
        # The frontier decided alone: it rejected before the veto ran, or
        # the decision needed no test (a KEEP is impossible without the
        # paired run by construction, so this is always a rejection or a
        # non-KEEP).
        return "frontier", "not_run"
    if "too few for a sign test" in notes:
        outcome = "underpowered"
    elif "consistent with judge noise" in notes:
        outcome = "not_significant"
    else:
        outcome = "significant"
    if decision == Decision.KEEP:
        return "none", outcome
    return "paired_veto", outcome


def write_baseline_scaffold(repo_root: Path, skills: dict[str, str]) -> None:
    """Materialize a scenario's baseline scaffold (idempotent)."""
    skills_dir = repo_root / "scaffold" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    for name, content in skills.items():
        (skills_dir / name).write_text(content, encoding="utf-8")


def seed_baseline(
    repo_root: Path,
    *,
    scaffold_commit_sha: str,
    scorer_fingerprint: str,
) -> CachedBaseline:
    """Cache the stub judge's draw of the current scaffold as the baseline."""
    report = report_from_stub(
        repo_root / "scaffold",
        draw_seed="baseline",
        scorer_fingerprint=scorer_fingerprint,
    )
    baseline = report_to_baseline(
        report,
        scaffold_commit_sha=scaffold_commit_sha,
        runtime_endpoint="stub",
        judge_endpoint="stub",
    )
    save_baseline(repo_root, baseline)
    return baseline


def aggregate(results: list[ScenarioResult]) -> dict[str, object]:
    """The confusion matrix over a calibration run."""
    tp = sum(1 for r in results if r.truth == "keep" and r.decision == Decision.KEEP)
    fn = sum(1 for r in results if r.truth == "keep" and r.decision != Decision.KEEP)
    fp = sum(1 for r in results if r.truth == "revert" and r.decision == Decision.KEEP)
    tn = sum(1 for r in results if r.truth == "revert" and r.decision == Decision.REVERT)
    return {
        "n_scenarios": len(results),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tpr": tp / (tp + fn) if tp + fn else None,
        "fpr": fp / (fp + tn) if fp + tn else None,
        "by_layer": {
            layer: sum(1 for r in results if r.rejecting_layer == layer)
            for layer in ("frontier", "paired_veto", "none")
        },
        "underpowered": sum(1 for r in results if r.paired_outcome == "underpowered"),
        "not_significant": sum(1 for r in results if r.paired_outcome == "not_significant"),
        "results": [
            {
                "name": r.name,
                "truth": r.truth,
                "decision": str(r.decision),
                "correct": r.correct,
                "rejecting_layer": r.rejecting_layer,
                "paired_outcome": r.paired_outcome,
                "notes": r.notes,
            }
            for r in results
        ],
    }


def make_session(scenario: Scenario) -> Callable[..., Any]:
    """An optimizer-session stand-in that returns the scenario's mutation.

    Used by both the offline suite and the live script: the mutation is
    predetermined by the scenario, so there is no optimizer session to run
    in either world — what differs is only the eval that judges the result.
    """
    from anvil.optimizer import actions
    from anvil.optimizer.parser import ParseResult

    action_cls = {
        "edit_skill": actions.EditSkillAction,
        "add_skill": actions.AddSkillAction,
        "delete_skill": actions.DeleteSkillAction,
        "add_rule": actions.AddRuleAction,
        "delete_rule": actions.DeleteRuleAction,
    }[scenario.action_kind]

    async def _session(**_kwargs: Any) -> Any:
        action = action_cls(rationale=scenario.description, **scenario.action_args)
        return action, "calibration transcript", ParseResult(
            action=action, parse_status="ok", n_blocks_found=1
        )

    return _session


def result_from_report(scenario: Scenario, report: RoundReport) -> ScenarioResult:
    """Fold a finished round into the matrix's row."""
    layer, outcome = layer_and_outcome(report.decision, report.notes or "")
    correct = (report.decision == Decision.KEEP) == (scenario.truth == "keep")
    return ScenarioResult(
        name=scenario.name,
        truth=scenario.truth,
        decision=report.decision,
        correct=correct,
        rejecting_layer=layer,
        paired_outcome=outcome,
        notes=report.notes or "",
    )


# Type alias for the injected eval: (scaffold_root, draw_seed) -> EvalReport.
StubEval = Callable[..., EvalReport]
