"""Run one ANVIL round end-to-end.

The orchestrator. Reads cached baseline → spawns optimizer session →
applies the action → runs eval on the mutated branch → computes
score_delta → writes critique md + round JSON + mutations log row →
applies the keep/revert decision via the configured gate
(``harness/config.yaml > gate``; see :mod:`anvil.loop.frontier`).

Sync function. Internally calls ``asyncio.run`` to bridge to the
async ``run_optimizer_session``. Everything else is plain Python.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from anvil.eval import evaluate_branch, load_baseline
from anvil.loop.builder import build_round_prompt
from anvil.loop.decision import Decision
from anvil.loop.frontier import (
    gate_decision,
    load_gate_config,
    scores_from_baseline,
    scores_from_eval,
)
from anvil.loop.git_ops import (
    commit_all,
    create_round_branch,
    current_branch,
    current_sha,
    delete_branch,
    ff_merge,
)
from anvil.loop.mutations_log import MutationRecord, append_mutation
from anvil.optimizer import (
    NoopAction,
    apply_action,
    run_optimizer_session,
)
from anvil.runtime.loader import default_runtime_config_path


@dataclass
class RoundReport:
    round_id: int
    branch: str
    decision: Decision
    action_kind: str
    parse_status: str
    diff_summary: str
    mode: str = "prompt"
    files_added: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    files_removed: list[str] = field(default_factory=list)
    baseline_score: float | None = None
    mutated_score: float | None = None
    score_delta: float | None = None
    eval_run_id: str | None = None
    git_commit_sha: str | None = None
    notes: str = ""


def run_round(
    *,
    round_id: int,
    repo_root: Path | str,
    scaffold_root: Path | str | None = None,
    profile: str = "DEFAULT",
    parent_branch: str = "anvil/exp",
    eval_mode: str | None = None,
    max_turns: int = 30,
) -> RoundReport:
    """Run round ``round_id`` end-to-end. Returns a report.

    Side effects:
      * Creates ``anvil/exp-round-<N>`` from ``parent_branch``.
      * May write under ``scaffold/`` (the applier).
      * Commits the change to the round branch.
      * Runs eval against the mutated branch.
      * Writes ``scaffold/memory/round_<N>_critique.md``,
        ``eval/runs/round_<N>.json``, appends to
        ``eval/mutations.jsonl``.
      * On KEEP: ff-merges round branch into ``parent_branch``.
        On REVERT/INFRA_FAIL: deletes the round branch.
    """
    repo_root = Path(repo_root).resolve()
    scaffold_root = Path(scaffold_root or (repo_root / "scaffold")).resolve()

    mode = _read_optimization_mode(scaffold_root)
    print(f"[round {round_id}] mode={mode}")

    _starting_branch = current_branch(repo_root)
    parent_sha = current_sha(repo_root)
    baseline = load_baseline(repo_root)

    # 1. Branch off the parent.
    branch = create_round_branch(repo_root, round_id, parent_branch=parent_branch)

    # 2. Build prompt and run the optimizer session.
    prompt = build_round_prompt(
        repo_root=repo_root,
        round_id=round_id,
        baseline=asdict_baseline(baseline) if baseline else None,
    )

    # NOTE: optimizer-side MLflow tracing is intentionally disabled for
    # now. The Databricks docs for Claude Code + MLflow GenAI require an
    # async-context-manager wrapping pattern that interacts subtly with
    # the SDK; we'd rather have rounds execute reliably than have a
    # half-working trace. The transcript still lives at
    # ``scaffold/memory/round_NNN_transcript.md`` for diagnostic.
    # Re-enable by passing experiment_name + round_id when the
    # async-tracing pattern is validated. See:
    # https://docs.databricks.com/aws/en/mlflow3/genai/tracing/integrations/claude-code
    action, transcript, parse_result = asyncio.run(
        run_optimizer_session(
            prompt=prompt,
            cwd=str(repo_root),
            max_turns=max_turns,
            profile=profile,
        )
    )

    # 3. Apply the action (writes scaffold files, edits harness.yaml).
    apply_result = apply_action(action, scaffold_root, mode=mode, repo_root=repo_root)

    # 4. Commit (no-op if nothing changed, e.g. noop action).
    commit_message = f"round {round_id:03d}: {apply_result.action_summary or 'noop'}"
    commit_sha = commit_all(repo_root, message=commit_message)

    # 5. Persist transcript (debug aid; not the critique md yet).
    transcript_path = repo_root / "scaffold" / "memory" / f"round_{round_id:03d}_transcript.md"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(transcript or "(empty)\n", encoding="utf-8")

    # 6. Run eval on the mutated branch (or skip if noop).
    mutated_score: float | None = None
    eval_run_id: str | None = None
    eval_failed = False
    eval_report = None
    if isinstance(action, NoopAction):
        # No need to evaluate for a noop; the score is the parent's.
        if baseline:
            mutated_score = float(baseline.aggregate)
    else:
        try:
            eval_report = evaluate_branch(
                scaffold_root=scaffold_root,
                profile=profile,
                mode=eval_mode,
            )
            mutated_score = eval_report.aggregate
            eval_run_id = eval_report.run_id
        except Exception as exc:  # noqa: BLE001 — surface any eval failure
            eval_failed = True
            mutated_score = None
            notes = f"eval failed: {exc.__class__.__name__}: {exc}"
            print(f"[round {round_id}] eval failure: {notes}")

    # 7. Compute score delta + decision.
    #
    # ``score_delta`` is reported vs the cached (frozen) baseline for the
    # human-facing artifacts (round JSON, mutations log) — it shows how the
    # mutation compares to the *original* baseline. The keep/revert DECISION
    # is delegated to the configured gate (``harness/config.yaml > gate``):
    #
    # * ``gate.type: frontier`` (default) — Pareto frontier. The decision
    #   compares the mutation against the best-so-far per objective
    #   (per-judge scores + aggregate), persisted to
    #   ``eval/runs/frontier.json``. A round that scores worse than a
    #   previously KEPT round is REVERTED even if it still beats the frozen
    #   baseline — the fix for the silent regression the frozen-baseline
    #   delta gate allowed. On the first scored round the frontier is
    #   initialized from the baseline.
    # * ``gate.type: delta`` — legacy frozen-baseline behavior (kept for
    #   backward compatibility); the decision reproduces the old
    #   ``score_delta > 0`` check exactly and does not touch the frontier.
    baseline_aggregate = float(baseline.aggregate) if baseline else None
    score_delta = (
        (mutated_score - baseline_aggregate)
        if (mutated_score is not None and baseline_aggregate is not None)
        else None
    )

    # Validate scorer-config compatibility between the cached baseline
    # and the current eval run. A weight or check_function change
    # invalidates the comparison even when scorer names are unchanged —
    # the cached aggregate has a different meaning under a different
    # weighting, so the frontier gate could make an invalid decision.
    # An empty fingerprint on either side (e.g. a baseline written
    # before this field existed) skips the check for backward compat.
    if (
        baseline is not None
        and eval_report is not None
        and baseline.scorer_fingerprint
        and eval_report.scorer_fingerprint
        and baseline.scorer_fingerprint != eval_report.scorer_fingerprint
    ):
        raise RuntimeError(
            "scorer configuration has changed since baseline was cached — "
            "regenerate the baseline with scripts/make_baseline.py"
        )

    gate_cfg = load_gate_config(scaffold_root)
    configured_objectives = gate_cfg.pareto.objectives if gate_cfg.pareto.enabled else None
    if gate_cfg.pareto.enabled and not configured_objectives:
        configured_objectives = None
    baseline_scores = (
        scores_from_baseline(baseline, configured_objectives) if baseline else None
    )
    mutated_scores = (
        scores_from_eval(eval_report, configured_objectives) if eval_report is not None else None
    )
    decision, frontier = gate_decision(
        repo_root=repo_root,
        gate_type=gate_cfg.type,
        epsilon=gate_cfg.epsilon,
        pareto=gate_cfg.pareto,
        baseline_scores=baseline_scores,
        baseline_aggregate=baseline_aggregate,
        mutated_scores=mutated_scores,
        mutated_aggregate=mutated_score,
        action_kind=action.action,
        eval_failed=eval_failed,
        parse_status=parse_result.parse_status,
    )

    # 8. Write critique md.
    critique_path = repo_root / "scaffold" / "memory" / f"round_{round_id:03d}_critique.md"
    critique_path.write_text(
        _build_critique_md(
            round_id=round_id,
            branch=branch,
            decision=decision,
            action_kind=action.action,
            apply_summary=apply_result.action_summary,
            rationale=action.rationale,
            baseline_score=baseline_aggregate,
            mutated_score=mutated_score,
            score_delta=score_delta,
            parse_status=parse_result.parse_status,
        ),
        encoding="utf-8",
    )

    # 9. Write round JSON (combines aggregate + decision + delta).
    round_json_path = repo_root / "eval" / "runs" / f"round_{round_id:03d}.json"
    round_json_path.parent.mkdir(parents=True, exist_ok=True)
    round_json_path.write_text(
        json.dumps(
            _build_round_json(
                round_id=round_id,
                branch=branch,
                commit_sha=commit_sha,
                parent_sha=parent_sha,
                decision=decision,
                action_kind=action.action,
                eval_report=eval_report,
                baseline_score=baseline_aggregate,
                score_delta=score_delta,
                parse_status=parse_result.parse_status,
                notes="",
                frontier_best=frontier.best if frontier else None,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _save_dashboard_round(repo_root, round_id, round_json_path.read_text(encoding="utf-8"))

    # 10. Append mutations log.
    record = MutationRecord.new(
        round_id=round_id,
        git_branch=branch,
        git_commit_sha=commit_sha,
        parent_commit_sha=parent_sha,
        files_added=apply_result.files_added,
        files_changed=apply_result.files_changed,
        files_removed=apply_result.files_removed,
        diff_summary=apply_result.action_summary,
        baseline_score=baseline_aggregate,
        mutated_score=mutated_score,
        score_delta=score_delta,
        decision=str(decision),
        mlflow_eval_run_id=eval_run_id,
        parse_status=parse_result.parse_status,
    )
    append_mutation(repo_root, record)

    # 10b. Commit the round artifacts (critique md, transcript md,
    # round JSON) onto the round branch BEFORE we ff-merge or delete
    # it. Without this step the critique md was orphaned: the only
    # commit_all (step 4) ran before the critique was written, so the
    # file lived as untracked in the working tree and got clobbered
    # by the next round's git checkout. Fixed in this commit; the
    # legacy artifacts for R4/R6/R8 were reconstructed via
    # ``scripts/reconstruct_critiques.py``.
    try:
        commit_all(repo_root, message=f"round {round_id:03d}: critique + eval artifacts")
    except Exception as exc:  # noqa: BLE001 — defensive, don't break the round
        print(f"[round {round_id}] warning: artifacts commit failed: {exc}")

    # 11. Apply git verdict.
    if decision == Decision.KEEP:
        ff_merge(repo_root, branch=branch, target=parent_branch)
    elif decision in (Decision.REVERT, Decision.INFRA_FAIL):
        delete_branch(repo_root, branch=branch, target=parent_branch)
    else:
        # NOOP: keep the (empty) branch around? Cheaper to delete.
        delete_branch(repo_root, branch=branch, target=parent_branch)

    # We don't hard-restore the starting branch — caller's session is now on
    # parent_branch, which is correct: that's where the kept mutation
    # lives. Worth printing for clarity.
    print(
        f"[round {round_id}] {decision} · action={action.action} · "
        f"baseline={baseline_aggregate} · mutated={mutated_score} · Δ={score_delta}"
    )

    return RoundReport(
        round_id=round_id,
        branch=branch,
        decision=decision,
        action_kind=action.action,
        parse_status=parse_result.parse_status,
        diff_summary=apply_result.action_summary,
        mode=mode,
        files_added=apply_result.files_added,
        files_changed=apply_result.files_changed,
        files_removed=apply_result.files_removed,
        baseline_score=baseline_aggregate,
        mutated_score=mutated_score,
        score_delta=score_delta,
        eval_run_id=eval_run_id,
        git_commit_sha=commit_sha,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def asdict_baseline(baseline) -> dict:
    """Convert a CachedBaseline to a plain dict for the prompt builder."""
    return baseline.to_dict()


def _save_dashboard_round(repo_root: Path | str, round_id: int, payload: str) -> Path:
    """Persist serialized round JSON in the dashboard data directory."""
    path = Path(repo_root) / "data" / f"round_{round_id:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def _read_optimization_mode(scaffold_root: Path | str) -> str:
    """Read the optimization mode from ``harness/config.yaml``.

    Returns ``"prompt"`` (the default) when the file or field is absent,
    so the loop keeps running on a repo that predates the mode field.
    Only the ``mode`` key is read here; the full-file ``extra="forbid"``
    check is enforced by the runtime loader.

    Invalid values (e.g. ``hybrid``, a typo, or an empty string) raise
    :class:`ValueError` so the loop fails closed at the source rather
    than silently permitting every action downstream.
    """
    path = default_runtime_config_path(Path(scaffold_root))
    if not path.is_file():
        return "prompt"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mode = raw.get("mode", "prompt")
    if mode not in ("prompt", "code"):
        raise ValueError(
            f"unknown optimization mode {mode!r}; expected 'prompt' or 'code'"
        )
    return mode


def _build_critique_md(
    *,
    round_id: int,
    branch: str,
    decision: Decision,
    action_kind: str,
    apply_summary: str,
    rationale: str,
    baseline_score: float | None,
    mutated_score: float | None,
    score_delta: float | None,
    parse_status: str,
) -> str:
    bs = f"{baseline_score:.4f}" if baseline_score is not None else "null"
    ms = f"{mutated_score:.4f}" if mutated_score is not None else "null"
    sd = f"{score_delta:+.4f}" if score_delta is not None else "null"
    return f"""---
round: {round_id}
branch: {branch}
decision: {decision}
action_kind: {action_kind}
parse_status: {parse_status}
baseline_score: {bs}
mutated_score: {ms}
score_delta: {sd}
---

# Round {round_id} critique

## Action applied
{apply_summary or "(no change)"}

## Rationale (from optimizer)
{rationale}

## Outcome
Decision: **{str(decision).upper()}**. Score delta vs cached baseline:
{sd}.
"""


def _build_round_json(
    *,
    round_id: int,
    branch: str,
    commit_sha: str,
    parent_sha: str,
    decision: Decision,
    action_kind: str,
    eval_report,
    baseline_score: float | None,
    score_delta: float | None,
    parse_status: str,
    notes: str,
    frontier_best: dict[str, float] | None = None,
) -> dict:
    payload: dict = {
        "round_id": round_id,
        "branch": branch,
        "scaffold_commit_sha": commit_sha,
        "parent_commit_sha": parent_sha,
        "decision": str(decision),
        "action_kind": action_kind,
        "parse_status": parse_status,
        "baseline_score": baseline_score,
        "score_delta_vs_parent": score_delta,
        "notes": notes,
        # Best-so-far per objective after this round's decision (frontier
        # gate only; None for the legacy delta gate / noop / infra-fail).
        # The decision is driven by this, not by ``score_delta_vs_parent``.
        "frontier_best": frontier_best,
    }
    if eval_report is not None:
        payload.update(
            {
                "evaluated_at": eval_report.evaluated_at,
                "n_examples": eval_report.n_rows,
                "mode": eval_report.mode,
                "scorers": list(eval_report.scorers),
                "aggregate": eval_report.aggregate,
                "per_judge": dict(eval_report.per_judge),
                "per_bucket": {k: dict(v) for k, v in eval_report.per_bucket.items()},
                "cost_metrics": dict(getattr(eval_report, "cost_metrics", {})),
                "failures": list(eval_report.failures),
                "trace_ids": list(eval_report.trace_ids),
                "mlflow": {
                    "run_id": eval_report.run_id,
                    "experiment_id": eval_report.experiment_id,
                },
            }
        )
    return payload
