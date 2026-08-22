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
from anvil.eval.judgeability import unjudgeable_reason_for
from anvil.loop.builder import build_round_prompt
from anvil.loop.decision import Decision
from anvil.loop.frontier import (
    gate_decision,
    load_gate_config,
    scores_from_baseline,
    scores_from_eval,
)
from anvil.loop.git_ops import (
    changed_paths,
    commit_all,
    create_round_branch,
    current_branch,
    current_sha,
    delete_branch,
    ff_merge,
    restore_paths,
)
from anvil.loop.mutations_log import MutationRecord, append_mutation
from anvil.optimizer import (
    NoopAction,
    apply_action,
    run_optimizer_session,
)
from anvil.optimizer.policy import ToolPolicy
from anvil.runtime.loader import default_runtime_config_path, load_eval_config


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
    optimizer_endpoint = _read_optimizer_endpoint(scaffold_root)
    cost_budget_usd = _read_cost_budget_usd(scaffold_root)
    eval_cfg = load_eval_config(scaffold_root)
    policy = ToolPolicy(root=repo_root)
    print(f"[round {round_id}] mode={mode}")

    _starting_branch = current_branch(repo_root)
    parent_sha = current_sha(repo_root)
    baseline = load_baseline(repo_root)

    # Snapshot the tree's existing dirt so the scope check in step 3b can
    # attribute writes to *this round* rather than to whatever was already
    # lying around. Runs are normally started on a clean tree, but
    # `--allow-dirty` exists and leftover round artifacts are common; blaming
    # the optimizer for a file it never touched would fail rounds at random.
    pre_session_dirt = set(changed_paths(repo_root))

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
            optimizer_endpoint=optimizer_endpoint,
            policy=policy,
            max_budget_usd=cost_budget_usd,
        )
    )

    # 3. Apply the action (writes scaffold files, edits harness.yaml).
    apply_result = apply_action(action, scaffold_root, mode=mode, repo_root=repo_root)

    # 3b. Verify the session stayed inside its writable scope.
    #
    # The permission callback in optimizer/policy.py denies out-of-scope writes
    # as they are attempted, but it runs inside the Claude Agent SDK and is
    # therefore only as reliable as the SDK's willingness to call it. This check
    # depends on nothing but git: whatever mechanism produced a write, the write
    # shows up here. A violation means the round is unjudgeable -- the mutation
    # may have altered the very thing about to grade it -- so it is failed rather
    # than scored, and the offending edits are reverted so they cannot ride the
    # branch deletion onto the parent.
    touched_this_round = [p for p in changed_paths(repo_root) if p not in pre_session_dirt]
    scope_violations = policy.verify_changed_paths(touched_this_round)
    if scope_violations:
        restore_paths(repo_root, scope_violations)
        print(
            f"[round {round_id}] scope violation, reverted: {', '.join(scope_violations)}"
        )

    # 4. Commit the mutation — but only when the applier actually wrote
    # something. A parse-failure noop (parse_status=no_block) collapses
    # to a NoopAction whose applier writes no files; committing on the
    # resulting empty index exits non-zero ("no changes added to commit")
    # and aborts the whole multi-round run. Detect no-change explicitly
    # from the applier's file lists (the robust signal) rather than
    # relying on the commit failing, and record the parent SHA instead.
    # ``commit_all`` is itself hardened against an empty index, so a
    # real mutation whose written content is byte-identical to HEAD
    # (files_changed populated but nothing staged) still returns the
    # current SHA instead of raising.
    applied_change = bool(
        apply_result.files_added
        or apply_result.files_changed
        or apply_result.files_removed
    )
    if applied_change:
        commit_message = f"round {round_id:03d}: {apply_result.action_summary or 'noop'}"
        commit_sha = commit_all(repo_root, message=commit_message)
    else:
        commit_sha = current_sha(repo_root)

    # 5. Persist transcript (debug aid; not the critique md yet).
    transcript_path = repo_root / "scaffold" / "memory" / f"round_{round_id:03d}_transcript.md"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(transcript or "(empty)\n", encoding="utf-8")

    # 6. Run eval on the mutated branch (or skip if noop).
    mutated_score: float | None = None
    eval_run_id: str | None = None
    eval_failed = False
    eval_report = None
    notes = ""
    if scope_violations:
        # Do not spend an eval on a round whose grader integrity is in doubt.
        eval_failed = True
        notes = "scope violation: " + ", ".join(scope_violations)
    elif isinstance(action, NoopAction):
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
        else:
            # An errored case is excluded from the aggregate, so a degraded
            # endpoint does not drag the score down -- it shrinks the sample the
            # score was measured on. Past a point that sample is not worth
            # comparing: the round is unjudgeable, so it is failed rather than
            # reverted. Without this, a throttled gateway silently discards good
            # mutations and the round record cannot say why.
            #
            # ``eval_failed`` short-circuits ``gate_decision`` to INFRA_FAIL
            # before any frontier I/O, so the frontier is not advanced by a
            # number that was never trustworthy. ``mutated_score`` is kept on
            # the record: "0.41, but 40% of cases never ran" is a more useful
            # thing to read six rounds later than a null.
            reason = unjudgeable_reason_for(eval_report, eval_cfg)
            if reason:
                eval_failed = True
                notes = reason
                print(f"[round {round_id}] unjudgeable: {notes}")

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

    # A round that wrote outside its scope does not reach the gate at all.
    # Routing it through `gate_decision` would be wrong twice over: `decide()`
    # returns NOOP for a noop action before it ever looks at `eval_failed`, so a
    # violation on a noop round would be recorded as a clean noop; and a round
    # that may have edited the frontier, the baseline, or the evaluator must not
    # be allowed to advance the frontier it just touched.
    decision: Decision
    frontier = None
    if scope_violations:
        # Note this also avoids *parsing* a config the round may have rewritten:
        # `load_gate_config` validates strictly and would raise on a tampered
        # epsilon, turning a handled failure into a crashed loop.
        decision = Decision.INFRA_FAIL
    else:
        gate_cfg = load_gate_config(scaffold_root)
        configured_objectives = gate_cfg.pareto.objectives if gate_cfg.pareto.enabled else None
        if gate_cfg.pareto.enabled and not configured_objectives:
            configured_objectives = None
        baseline_scores = (
            scores_from_baseline(baseline, configured_objectives) if baseline else None
        )
        mutated_scores = (
            scores_from_eval(eval_report, configured_objectives)
            if eval_report is not None
            else None
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
    round_payload = (
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
                notes=notes,
                frontier_best=frontier.best if frontier else None,
            ),
            indent=2,
        )
        + "\n"
    )
    round_json_path.write_text(round_payload, encoding="utf-8")
    _save_dashboard_round(repo_root, round_id, round_payload)

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
        notes=notes,
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


def _read_cost_budget_usd(scaffold_root: Path | str) -> float | None:
    """Read ``loop.cost_budget_usd_per_round`` from harness/config.yaml.

    The field has existed since the loop was written and was never enforced.
    The Claude Agent SDK takes a ``max_budget_usd`` ceiling directly, so the
    declared budget can now be the real one.
    """
    path = default_runtime_config_path(Path(scaffold_root))
    if not path.is_file():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    budget = (raw.get("loop") or {}).get("cost_budget_usd_per_round")
    return float(budget) if budget is not None else None


def _read_optimizer_endpoint(scaffold_root: Path | str) -> str | None:
    """Read the ``optimizer_endpoint`` FMAPI model from harness/config.yaml.

    Returns None when the file or field is absent, so the optimizer
    session falls back to its built-in default model. Only this field is
    read here; the full-file ``extra="forbid"`` validation is enforced
    by the runtime loader.
    """
    path = default_runtime_config_path(Path(scaffold_root))
    if not path.is_file():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw.get("optimizer_endpoint")


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
                # How much of the eval actually happened. A reader six rounds
                # later needs this next to the aggregate: it is the difference
                # between "the agent scored 0.41" and "0.41 was measured on
                # three of eight cases".
                "n_errors": eval_report.n_errors,
                "error_rate": eval_report.error_rate,
                "errors": list(eval_report.errors),
                "per_judge": dict(eval_report.per_judge),
                # The denominators behind ``per_judge``. Each per-judge value is a
                # mean over only the rows that produced a score, so the value
                # alone cannot distinguish "1.0 across eight cases" from "1.0 on
                # the one case this judge did not break on".
                "per_judge_assessed": dict(eval_report.per_judge_assessed),
                "per_judge_errors": dict(eval_report.per_judge_errors),
                "scorer_errors": list(eval_report.scorer_errors),
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
