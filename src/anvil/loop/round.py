"""Run one ANVIL round end-to-end.

The orchestrator. Reads cached baseline → spawns optimizer session →
applies the action → runs eval on the mutated branch → computes
score_delta → writes critique md + round JSON + mutations log row →
applies the keep/revert decision.

Sync function. Internally calls ``asyncio.run`` to bridge to the
async ``run_optimizer_session``. Everything else is plain Python.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from anvil.eval import evaluate_branch, load_baseline
from anvil.loop.builder import build_round_prompt
from anvil.loop.decision import Decision, decide
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


@dataclass
class RoundReport:
    round_id: int
    branch: str
    decision: Decision
    action_kind: str
    parse_status: str
    diff_summary: str
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

    starting_branch = current_branch(repo_root)
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
    apply_result = apply_action(action, scaffold_root)

    # 4. Commit (no-op if nothing changed, e.g. noop action).
    commit_message = (
        f"round {round_id:03d}: {apply_result.action_summary or 'noop'}"
    )
    commit_sha = commit_all(repo_root, message=commit_message)

    # 5. Persist transcript (debug aid; not the critique md yet).
    transcript_path = (
        repo_root / "scaffold" / "memory" / f"round_{round_id:03d}_transcript.md"
    )
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
    baseline_aggregate = float(baseline.aggregate) if baseline else None
    score_delta = (
        (mutated_score - baseline_aggregate)
        if (mutated_score is not None and baseline_aggregate is not None)
        else None
    )
    decision = decide(
        score_delta=score_delta,
        action_kind=action.action,
        parse_status=parse_result.parse_status,
        eval_failed=eval_failed,
    )

    # 8. Write critique md.
    critique_path = (
        repo_root / "scaffold" / "memory" / f"round_{round_id:03d}_critique.md"
    )
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
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

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

    # 11. Apply git verdict.
    if decision == Decision.KEEP:
        ff_merge(repo_root, branch=branch, target=parent_branch)
    elif decision in (Decision.REVERT, Decision.INFRA_FAIL):
        delete_branch(repo_root, branch=branch, target=parent_branch)
    else:
        # NOOP: keep the (empty) branch around? Cheaper to delete.
        delete_branch(repo_root, branch=branch, target=parent_branch)

    # We don't hard-restore starting_branch — caller's session is now on
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
{apply_summary or '(no change)'}

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
                "failures": list(eval_report.failures),
                "trace_ids": list(eval_report.trace_ids),
                "mlflow": {
                    "run_id": eval_report.run_id,
                    "experiment_id": eval_report.experiment_id,
                },
            }
        )
    return payload
