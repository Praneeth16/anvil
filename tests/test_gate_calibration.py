"""Gate calibration: the assembled gate, measured against known truth (issue #8).

The unit tests prove the sign test's arithmetic; these prove the gate's
DECISIONS. Each test drives one scenario end-to-end through ``run_round``
with the eval injected from the deterministic stub judge
(:mod:`anvil.loop.calibration`): the scaffold on disk decides the scores,
so a crippled scaffold measurably scores worse and a restored one better.
Every scenario runs on its own repo — the frontier persists across rounds
and must not leak between scenarios.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from anvil.eval.cache import compute_scorer_fingerprint
from anvil.loop.calibration import (
    SCORERS,
    aggregate,
    layer_and_outcome,
    make_session,
    report_from_stub,
    result_from_report,
    scenarios,
    seed_baseline,
    write_baseline_scaffold,
)
from anvil.loop.decision import Decision
from anvil.runtime.models import ScorerConfig

FINGERPRINT = compute_scorer_fingerprint([ScorerConfig(name=n) for n in SCORERS])
_SCENARIOS = {s.name: s for s in scenarios()}


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


@pytest.fixture
def anvil_repo(tmp_path: Path) -> Path:
    """A committed repo on ``anvil/exp``, ready for one calibration round."""
    repo = tmp_path
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@e.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")

    (repo / "scaffold" / "memory").mkdir(parents=True)
    (repo / "scaffold" / "skills").mkdir()
    (repo / "scaffold" / "rules").mkdir()
    (repo / "scaffold" / "harness.yaml").write_text("skills: []\ntools: []\n")
    (repo / "harness").mkdir()
    (repo / "harness" / "config.yaml").write_text(
        "mode: prompt\n"
        "runtime_endpoint: runtime\n"
        "optimizer_endpoint: optimizer\n"
        "judge_endpoint: judge\n"
        "experiments: {runtime: r, eval: e, optimizer: o}\n"
    )
    (repo / "eval" / "runs").mkdir(parents=True)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    _git(repo, "checkout", "-q", "-b", "anvil/exp")
    return repo


def _run_scenario(anvil_repo: Path, name: str, monkeypatch: pytest.MonkeyPatch):
    """Seed the scenario's baseline, run one round against the stub judge."""
    import anvil.loop.round as round_mod

    scenario = _SCENARIOS[name]
    write_baseline_scaffold(anvil_repo, scenario.baseline_skills)
    _git(anvil_repo, "add", ".")
    _git(anvil_repo, "commit", "-q", "-m", "baseline scaffold")
    seed_baseline(
        anvil_repo,
        scaffold_commit_sha=_git(anvil_repo, "rev-parse", "HEAD"),
        scorer_fingerprint=FINGERPRINT,
    )
    monkeypatch.setattr(round_mod, "run_optimizer_session", make_session(scenario))
    monkeypatch.setattr(
        round_mod,
        "evaluate_branch",
        lambda **_kw: report_from_stub(
            anvil_repo / "scaffold",
            draw_seed=scenario.draw_seed or f"round-{scenario.name}",
            scorer_fingerprint=FINGERPRINT,
        ),
    )
    report = round_mod.run_round(round_id=1, repo_root=anvil_repo)
    return result_from_report(scenario, report)


# ---------------------------------------------------------------------------
# A/A — the same scaffold twice must revert, via the paired veto
# ---------------------------------------------------------------------------


def test_aa_round_reverts_via_the_paired_veto(anvil_repo: Path, monkeypatch) -> None:
    """The mutation rewrites the citation skill byte-identically: the
    candidate IS the baseline, the only differences are stub-judge flips,
    and a gate that KEEPs noise has failed."""
    result = _run_scenario(anvil_repo, "aa", monkeypatch)
    assert result.decision == Decision.REVERT
    assert result.rejecting_layer == "paired_veto"
    # The frontier cannot tell A/A apart (aggregates wobble either way);
    # only the paired test can. Underpowered and not-significant are both
    # honest outcomes — and the harness must count them separately.
    assert result.paired_outcome in ("underpowered", "not_significant")


def test_aa_assertion_is_sensitive_to_the_veto(anvil_repo: Path, monkeypatch) -> None:
    """The acceptance criterion's guard: with ``gate.test: none`` the same
    A/A round must KEEP, proving the test above would go red if the veto
    were ever switched off. The candidate's aggregate wobble exceeds the
    parent's at this fixture's seed, so the frontier alone promotes it."""
    (anvil_repo / "harness" / "config.yaml").write_text(
        (anvil_repo / "harness" / "config.yaml").read_text() + "gate:\n  test: none\n"
    )
    _git(anvil_repo, "commit", "-qam", "disable the paired test")
    result = _run_scenario(anvil_repo, "aa", monkeypatch)
    assert result.decision == Decision.KEEP, (
        "with the veto off, the A/A round must slip through the frontier — "
        "if it reverts anyway, the stub seed no longer exercises the veto and "
        "the calibration test above is testing nothing"
    )


# ---------------------------------------------------------------------------
# Known-bad — crippled scaffolds must revert, layer recorded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["bad_no_citation", "bad_no_refusal", "bad_confident"],
    ids=["citation-deleted", "refusal-deleted", "confident-rule-added"],
)
def test_known_bad_scaffolds_revert(anvil_repo: Path, monkeypatch, name: str) -> None:
    result = _run_scenario(anvil_repo, name, monkeypatch)
    assert result.decision == Decision.REVERT
    # The issue asks the harness to record WHICH layer rejected it; these
    # are loud regressions and the frontier should catch them before the
    # paired test ever runs.
    assert result.rejecting_layer == "frontier", (
        f"{name}: expected the frontier to catch a loud regression, "
        f"got layer={result.rejecting_layer} notes={result.notes}"
    )


# ---------------------------------------------------------------------------
# Known-good — restoring a crippled rule must keep
# ---------------------------------------------------------------------------


def test_known_good_restoration_keeps(anvil_repo: Path, monkeypatch) -> None:
    result = _run_scenario(anvil_repo, "good_restore_citation", monkeypatch)
    assert result.decision == Decision.KEEP, f"notes: {result.notes}"
    assert result.paired_outcome == "significant"


# ---------------------------------------------------------------------------
# The engine itself: determinism, layer attribution, the matrix
# ---------------------------------------------------------------------------


def test_stub_judge_is_deterministic_per_draw(tmp_path: Path) -> None:
    write_baseline_scaffold(tmp_path, _SCENARIOS["aa"].baseline_skills)
    first = report_from_stub(tmp_path / "scaffold", draw_seed="d")
    second = report_from_stub(tmp_path / "scaffold", draw_seed="d")
    assert first.per_row == second.per_row
    assert first.aggregate == second.aggregate


def test_stub_judge_scores_follow_the_scaffold(tmp_path: Path) -> None:
    write_baseline_scaffold(tmp_path, _SCENARIOS["aa"].baseline_skills)
    healthy = report_from_stub(tmp_path / "scaffold", draw_seed="d")
    (tmp_path / "scaffold" / "skills" / "citation.md").unlink()
    crippled = report_from_stub(tmp_path / "scaffold", draw_seed="d")
    assert crippled.aggregate < healthy.aggregate


def test_layer_and_outcome_reads_the_round_record() -> None:
    assert layer_and_outcome(Decision.REVERT, "") == ("frontier", "not_run")
    assert layer_and_outcome(Decision.REVERT, "paired: 2 discordant pair(s) of 12 is too few for a sign test to reach alpha") == (
        "paired_veto",
        "underpowered",
    )
    assert layer_and_outcome(Decision.REVERT, "paired: 1 improved / 5 regressed, p=0.9 > alpha=0.05: consistent with judge noise") == (
        "paired_veto",
        "not_significant",
    )
    assert layer_and_outcome(Decision.KEEP, "paired: 6 improved / 0 regressed") == (
        "none",
        "significant",
    )


def test_aggregate_matrix_counts_and_rates() -> None:
    from anvil.loop.calibration import ScenarioResult

    results = [
        ScenarioResult("g", "keep", Decision.KEEP, True, "none", "significant", ""),
        ScenarioResult("b1", "revert", Decision.REVERT, True, "frontier", "not_run", ""),
        ScenarioResult("b2", "revert", Decision.REVERT, True, "paired_veto", "underpowered", ""),
        ScenarioResult("b3", "revert", Decision.KEEP, False, "none", "significant", ""),
    ]
    matrix = aggregate(results)
    assert (matrix["tp"], matrix["fn"], matrix["fp"], matrix["tn"]) == (1, 0, 1, 2)
    assert matrix["tpr"] == 1.0
    assert matrix["fpr"] == pytest.approx(1 / 3)
    assert matrix["by_layer"] == {"frontier": 1, "paired_veto": 1, "none": 2}
    assert matrix["underpowered"] == 1
    assert matrix["not_significant"] == 0
