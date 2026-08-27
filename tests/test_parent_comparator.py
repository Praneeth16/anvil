"""The paired test pairs against the current parent, not the frozen baseline.

Issue #19: before D13, ``gate_decision`` received the frozen baseline's
``per_row`` forever, so from round two on the veto answered "does the
candidate differ from the original scaffold" rather than "does it improve on
its parent". These tests pin the contract that replaced that:

* a KEEP rewrites ``eval/runs/parent.json`` from the kept candidate's own
  eval report, and the NEXT round's paired test receives it as the
  comparator (the regression test the issue demands);
* a REVERT, NOOP or INFRA_FAIL writes nothing — the parent did not change;
* before the first KEEP the frozen baseline is the comparator, because it is
  the parent of the first candidate by definition;
* a parent comparator that went stale out-of-band (mid-campaign baseline
  regen, hand edit) fails the round pre-flight, before any eval spend;
* ``scripts/make_baseline.py`` re-anchors by deleting ``parent.json``.

No LLM, no workspace: the optimizer session and ``evaluate_branch`` are
monkeypatched, and git runs against a tmp repo.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from anvil.eval.cache import (
    CachedBaseline,
    compute_scorer_fingerprint,
    load_parent,
    parent_path,
    save_baseline,
    save_parent,
)
from anvil.runtime.models import ScorerConfig

REPO_ROOT = Path(__file__).resolve().parent.parent

_BASE_ROWS = {f"q{i}": {"correctness": 0.4} for i in range(6)}


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


@pytest.fixture
def anvil_repo(tmp_path: Path) -> Path:
    """A committed ANVIL repo on ``anvil/exp`` with a per-row baseline."""
    repo = tmp_path
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@e.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")

    (repo / "scaffold" / "memory").mkdir(parents=True)
    (repo / "scaffold" / "skills").mkdir()
    (repo / "scaffold" / "harness.yaml").write_text("skills: []\ntools: []\n")
    (repo / "harness").mkdir()
    # Endpoints + experiments are required RuntimeYAML fields: the KEEP path
    # persists the parent comparator with the endpoints the eval ran under.
    (repo / "harness" / "config.yaml").write_text(
        "mode: prompt\n"
        "runtime_endpoint: runtime\n"
        "optimizer_endpoint: optimizer\n"
        "judge_endpoint: judge\n"
        "experiments: {runtime: r, eval: e, optimizer: o}\n"
    )
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
            aggregate=0.4,
            per_judge={"correctness": 0.4},
            per_bucket={"direct": {"correctness": 0.4}},
            n_examples=6,
            scorer_fingerprint=compute_scorer_fingerprint([ScorerConfig(name="correctness")]),
            per_row={k: dict(v) for k, v in _BASE_ROWS.items()},
        ),
    )

    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    _git(repo, "checkout", "-q", "-b", "anvil/exp")
    return repo


def _mutating_session(skill_name: str):
    """An optimizer session that adds a skill — a real (non-noop) mutation."""
    from anvil.optimizer.actions import AddSkillAction
    from anvil.optimizer.parser import ParseResult

    async def _session(**_kwargs):
        action = AddSkillAction(
            rationale=f"add {skill_name} so the round is scored",
            target_file=f"skills/{skill_name}.md",
            content=f"# {skill_name}\n",
        )
        return action, "transcript", ParseResult(action=action, parse_status="ok", n_blocks_found=1)

    return _session


def _report(score: float):
    """An ``EvalReport`` whose six rows all score ``score`` on correctness."""
    from anvil.eval.runner import EvalReport

    return EvalReport(
        aggregate=score,
        per_judge={"correctness": score},
        per_bucket={"direct": {"correctness": score}},
        failures=[],
        run_id="run-x",
        experiment_id="exp-x",
        n_rows=6,
        mode="test",
        scorers=["correctness"],
        evaluated_at="2026-08-22T13:00:00+00:00",
        per_row={f"q{i}": {"correctness": score} for i in range(6)},
        aggregate_scorer_names=["correctness"],
        aggregate_weights={"correctness": 1.0},
    )


def _run_round(repo: Path, monkeypatch: pytest.MonkeyPatch, round_id: int, score: float):
    """Run one scored round with the session and eval mocked out."""
    import anvil.loop.round as round_mod

    monkeypatch.setattr(round_mod, "run_optimizer_session", _mutating_session(f"s{round_id}"))
    monkeypatch.setattr(round_mod, "evaluate_branch", lambda **_kw: _report(score))
    return round_mod.run_round(round_id=round_id, repo_root=repo)


# ---------------------------------------------------------------------------
# The cache trio
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parent_round_trip_and_absent(tmp_path: Path) -> None:
    assert load_parent(tmp_path) is None
    parent = CachedBaseline(
        scaffold_commit_sha="b" * 40,
        evaluated_at="2026-08-22T12:00:00+00:00",
        mode="test",
        scorers=["correctness"],
        runtime_endpoint="runtime",
        judge_endpoint="judge",
        aggregate=0.6,
        per_row={"q0": {"correctness": 0.6}},
    )
    save_parent(tmp_path, parent)
    loaded = load_parent(tmp_path)
    assert loaded == parent
    assert loaded is not parent  # a copy: mutating it must not leak


# ---------------------------------------------------------------------------
# The regression test the issue asks for
# ---------------------------------------------------------------------------


def test_after_a_keep_the_next_round_pairs_against_the_kept_scaffold(
    anvil_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 1 KEEPs at 0.6 over the 0.4 baseline. Round 2 must be paired
    against ROUND 1's per-row scores — its actual parent — not against the
    frozen baseline's."""
    import anvil.loop.frontier as frontier_mod
    from anvil.loop.decision import Decision

    report1 = _run_round(anvil_repo, monkeypatch, round_id=1, score=0.6)
    assert report1.decision == Decision.KEEP

    parent = load_parent(anvil_repo)
    assert parent is not None
    assert parent.aggregate == 0.6
    assert parent.per_row == {f"q{i}": {"correctness": 0.6} for i in range(6)}
    # The comparator names the commit that produced it, so a reader can trace
    # the draw back to the kept scaffold -- and it is not the baseline's SHA.
    assert parent.scaffold_commit_sha != "a" * 40
    assert len(parent.scaffold_commit_sha) == 40

    seen: dict[str, object] = {}
    real = frontier_mod.paired_sign_test

    def _spy(baseline_per_row, candidate_per_row, **kw):
        seen["comparator"] = baseline_per_row
        seen["candidate"] = candidate_per_row
        return real(baseline_per_row, candidate_per_row, **kw)

    monkeypatch.setattr(frontier_mod, "paired_sign_test", _spy)

    report2 = _run_round(anvil_repo, monkeypatch, round_id=2, score=0.8)
    assert report2.decision == Decision.KEEP
    assert seen["comparator"] == {f"q{i}": {"correctness": 0.6} for i in range(6)}
    assert seen["candidate"] == {f"q{i}": {"correctness": 0.8} for i in range(6)}
    # And the chain continues: round 2's KEEP replaced the comparator again.
    assert load_parent(anvil_repo).aggregate == 0.8


def test_before_the_first_keep_the_comparator_is_the_frozen_baseline(
    anvil_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 1 has no parent.json; the baseline is the parent by definition."""
    import anvil.loop.frontier as frontier_mod

    seen: dict[str, object] = {}
    real = frontier_mod.paired_sign_test

    def _spy(baseline_per_row, candidate_per_row, **kw):
        seen["comparator"] = baseline_per_row
        return real(baseline_per_row, candidate_per_row, **kw)

    monkeypatch.setattr(frontier_mod, "paired_sign_test", _spy)

    report = _run_round(anvil_repo, monkeypatch, round_id=1, score=0.6)
    from anvil.loop.decision import Decision

    assert report.decision == Decision.KEEP
    assert seen["comparator"] == _BASE_ROWS


# ---------------------------------------------------------------------------
# Non-KEEP decisions write nothing
# ---------------------------------------------------------------------------


def test_a_revert_leaves_the_parent_comparator_untouched(
    anvil_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from anvil.loop.decision import Decision

    report1 = _run_round(anvil_repo, monkeypatch, round_id=1, score=0.6)
    assert report1.decision == Decision.KEEP
    before = parent_path(anvil_repo).read_bytes()

    # 0.5 < the 0.6 frontier: a plain regression, reverted by the frontier
    # itself. The parent is still round 1's scaffold, so the comparator must
    # be byte-identical -- reuse across a revert streak is correct because
    # the parent genuinely did not change.
    report2 = _run_round(anvil_repo, monkeypatch, round_id=2, score=0.5)
    assert report2.decision == Decision.REVERT
    assert parent_path(anvil_repo).read_bytes() == before


def test_noop_and_infra_fail_write_no_parent(
    anvil_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anvil.loop.round as round_mod
    from anvil.loop.decision import Decision
    from anvil.optimizer.actions import NoopAction
    from anvil.optimizer.parser import ParseResult

    async def _noop(**_kwargs):
        action = NoopAction(rationale="nothing worth changing")
        return action, "transcript", ParseResult(action=action, parse_status="ok", n_blocks_found=1)

    monkeypatch.setattr(round_mod, "run_optimizer_session", _noop)
    report = round_mod.run_round(round_id=1, repo_root=anvil_repo)
    assert report.decision == Decision.NOOP
    assert not parent_path(anvil_repo).exists()

    monkeypatch.setattr(round_mod, "run_optimizer_session", _mutating_session("boom"))

    def _failing_eval(**_kw):
        raise RuntimeError("gateway down")

    monkeypatch.setattr(round_mod, "evaluate_branch", _failing_eval)
    report = round_mod.run_round(round_id=2, repo_root=anvil_repo)
    assert report.decision == Decision.INFRA_FAIL
    assert not parent_path(anvil_repo).exists()


# ---------------------------------------------------------------------------
# Staleness fails closed, before any spend
# ---------------------------------------------------------------------------


def test_a_stale_parent_comparator_fails_before_any_spend(
    anvil_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """parent.json outlived its world: a baseline regenerated on a new domain
    (or a hand edit) left it pointing at rows the current dataset is not.
    Pairing against it would silently shrink to zero shared rows and leave the
    frontier unchecked, so the round refuses before the eval is paid for."""
    import anvil.loop.round as round_mod

    # A real dataset on disk so the fingerprint is non-empty and the mismatch
    # is checked rather than waved through as "absent". Passed explicitly:
    # the defaults are CWD-relative and would fingerprint the real repo's
    # data instead of this fixture's.
    (anvil_repo / "data" / "kb").mkdir(parents=True)
    (anvil_repo / "data" / "kb" / "doc.md").write_text("# doc\n")
    (anvil_repo / "data" / "golden_set.jsonl").write_text("{}\n")

    save_parent(
        anvil_repo,
        CachedBaseline(
            scaffold_commit_sha="c" * 40,
            evaluated_at="2026-08-22T12:00:00+00:00",
            mode="test",
            scorers=["correctness"],
            runtime_endpoint="runtime",
            judge_endpoint="judge",
            aggregate=0.6,
            dataset_fingerprint="sha256:from-another-domain",
            per_row={"q0": {"correctness": 0.6}},
        ),
    )

    def _fail_if_called(**_kw):
        raise AssertionError("a stale comparator must refuse before any eval spend")

    monkeypatch.setattr(round_mod, "evaluate_branch", _fail_if_called)

    with pytest.raises(RuntimeError, match="parent comparator is stale"):
        round_mod.run_round(
            round_id=1,
            repo_root=anvil_repo,
            kb_dir=anvil_repo / "data" / "kb",
            golden_set_path=anvil_repo / "data" / "golden_set.jsonl",
        )


# ---------------------------------------------------------------------------
# Re-anchoring
# ---------------------------------------------------------------------------


def test_make_baseline_reanchors_the_parent_comparator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A regenerated baseline is the parent of whatever runs next, so the old
    comparator is deleted -- the next round falls back to the fresh baseline
    rather than pairing against a draw from a superseded world."""
    spec = importlib.util.spec_from_file_location(
        "make_baseline_script", REPO_ROOT / "scripts" / "make_baseline.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)

    parent = CachedBaseline(
        scaffold_commit_sha="d" * 40,
        evaluated_at="2026-08-22T12:00:00+00:00",
        mode="test",
        scorers=["correctness"],
        runtime_endpoint="runtime",
        judge_endpoint="judge",
        aggregate=0.6,
        per_row={"q0": {"correctness": 0.6}},
    )
    save_parent(tmp_path, parent)
    assert parent_path(tmp_path).exists()

    fresh = CachedBaseline(
        scaffold_commit_sha="e" * 40,
        evaluated_at="2026-08-22T13:00:00+00:00",
        mode="test",
        scorers=["correctness"],
        runtime_endpoint="runtime",
        judge_endpoint="judge",
        aggregate=0.5,
    )
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "build_baseline", lambda **_kw: fresh)

    assert module.main(["--out", str(tmp_path / "baseline.json")]) == 0
    assert not parent_path(tmp_path).exists()
    assert (tmp_path / "baseline.json").exists()
