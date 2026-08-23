#!/usr/bin/env python3
"""CLI driver for ``anvil.loop.round.run_round``.

Usage::

    # Run one round (auto-detects round_id from existing branches)
    uv run python scripts/run_round.py

    # Run a specific round id
    uv run python scripts/run_round.py --round-id 1

    # Run N consecutive rounds
    uv run python scripts/run_round.py --rounds 3

    # Choose eval mode (default = harness/config.yaml > eval.default_mode)
    uv run python scripts/run_round.py --eval-mode quick

The runner expects:

* The repo's parent branch (default ``anvil/exp``) to exist.
* ``eval/runs/baseline.json`` to be cached (run
  ``scripts/evaluate.py --mode full`` first if missing).
* Optimizer auth is handled automatically by Claude Code via the
  Databricks CLI (``CLAUDE_CODE_USE_GATEWAY=1``), using the active
  profile / ``DATABRICKS_HOST`` — no secret is required. An operator
  may set ``ANTHROPIC_AUTH_TOKEN`` in the env as an optional override.

Exit status (see :mod:`anvil.cli`): ``0`` every requested round ran to a
verdict, ``2`` a round could not be judged (``INFRA_FAIL``: the eval broke, the
error rate was above ``eval.max_error_rate``, or the session wrote outside its
scope) or the invocation itself was wrong, ``130`` interrupted.

A REVERT is not a failure of this script — a mutation that does not improve the
agent is the loop working — so a run of 50 rounds that keeps two of them exits
``0``. Only a round that could not be *measured* is an error.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from anvil.cli import ExitCode, run_cli  # noqa: E402
from anvil.loop.decision import Decision  # noqa: E402
from anvil.loop.git_ops import check_clean_worktree  # noqa: E402
from anvil.loop.round import run_round  # noqa: E402


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--round-id", type=int, default=None, help="explicit round id")
    p.add_argument("--rounds", type=int, default=1, help="number of consecutive rounds")
    p.add_argument("--profile", default="DEFAULT", help="Databricks CLI profile")
    p.add_argument("--parent-branch", default="anvil/exp", help="branch to fork from + ff-merge to")
    p.add_argument(
        "--eval-mode",
        choices=["quick", "standard", "full"],
        default=None,
        help="eval mode for the post-mutation eval",
    )
    # The domain the round is optimized against. These must match whatever
    # eval/runs/baseline.json was generated from -- the gate compares the
    # round's aggregate against that cached number without re-checking which
    # domain produced it.
    p.add_argument(
        "--kb-dir",
        default=str(REPO_ROOT / "data" / "kb"),
        help="path to the knowledge-base directory of *.md docs",
    )
    p.add_argument(
        "--golden-set-path",
        default=str(REPO_ROOT / "data" / "golden_set.jsonl"),
        help="path to the golden set JSONL",
    )
    p.add_argument(
        "--evaluator-path",
        default=None,
        help="path to the programmatic check-function module (default: data/evaluator.py)",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=30,
        help="hard cap on optimizer CLI turns per round",
    )
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="skip the clean-worktree safety check",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="run even when eval/runs/finalized.json exists",
    )
    return p


def _next_round_id(repo_root: Path) -> int:
    """Highest existing eval/runs/round_NNN.json + 1, or 1 if none."""
    runs = (repo_root / "eval" / "runs").glob("round_*.json")
    nums = []
    for p in runs:
        m = re.search(r"round_(\d+)\.json$", p.name)
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1


def main(argv: list[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)

    finalized_path = REPO_ROOT / "eval" / "runs" / "finalized.json"
    if finalized_path.is_file() and not args.force:
        print(f"ERROR: optimization is finalized ({finalized_path}); pass --force to override.")
        # 2, not 1: refusing to run is a malfunction of the invocation, not a
        # result about the agent. Exit 1 is reserved for "measured, and not
        # good enough".
        return ExitCode.ERROR

    if args.allow_dirty:
        print("WARNING: --allow-dirty specified; skipping clean-worktree safety check.")
    else:
        check_clean_worktree()

    # Verify parent branch exists.
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--verify", args.parent_branch],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(f"ERROR: parent branch {args.parent_branch!r} does not exist.")
        print(f"Create it first: git -C {REPO_ROOT} checkout -b {args.parent_branch} main")
        return ExitCode.ERROR

    # Validate the domain before the first billed session. These paths are not
    # touched until step 6 of the round, deep inside the try/except that records
    # INFRA_FAIL and keeps looping -- so `--rounds 50` with a mistyped --kb-dir
    # spends fifty optimizer sessions to produce fifty identical failures.
    kb_dir = Path(args.kb_dir)
    if not kb_dir.is_dir() or not any(kb_dir.glob("*.md")):
        print(f"ERROR: --kb-dir {kb_dir} is not a directory containing *.md documents.")
        return ExitCode.ERROR
    golden_set_path = Path(args.golden_set_path)
    if not golden_set_path.is_file():
        print(f"ERROR: --golden-set-path {golden_set_path} does not exist.")
        return ExitCode.ERROR
    if args.evaluator_path is not None and not Path(args.evaluator_path).is_file():
        print(f"ERROR: --evaluator-path {args.evaluator_path} does not exist.")
        return ExitCode.ERROR

    next_id = args.round_id if args.round_id is not None else _next_round_id(REPO_ROOT)

    unjudged: list[str] = []
    escaped: list[str] = []
    for i in range(args.rounds):
        rid = next_id + i
        print(f"\n=== round {rid} ===")
        report = run_round(
            round_id=rid,
            repo_root=REPO_ROOT,
            kb_dir=args.kb_dir,
            golden_set_path=args.golden_set_path,
            evaluator_path=args.evaluator_path,
            profile=args.profile,
            parent_branch=args.parent_branch,
            eval_mode=args.eval_mode,
            max_turns=args.max_turns,
        )
        if report.decision == Decision.INFRA_FAIL:
            # Both a throttled gateway and an optimizer writing outside its
            # writable scope land as INFRA_FAIL, and they are not remotely the
            # same event: one is a bad afternoon, the other is the session
            # reaching for its own grader, which this repo treats as a security
            # property. A caller reading only the exit status cannot tell them
            # apart, so at minimum they are separated on stderr.
            if report.notes.startswith("scope violation"):
                escaped.append(f"round {rid}: {report.notes}")
            else:
                unjudged.append(f"round {rid}: {report.notes or 'eval failed'}")
        print(
            f"=== round {rid} done · {report.decision} · "
            f"action={report.action_kind} · Δ={report.score_delta}\n"
        )

    if escaped:
        print(
            f"\nSCOPE VIOLATION in {len(escaped)}/{args.rounds} round(s) — the optimizer "
            "wrote outside its writable scope. This is not a flaky endpoint; treat it as "
            "a containment failure and read the round records before running more rounds:",
            file=sys.stderr,
        )
        for line in escaped:
            print(f"  {line}", file=sys.stderr)
    if unjudged:
        # A round that could not be measured is a malfunction worth surfacing to
        # whatever launched this, even though the loop kept going: it means a
        # round was spent without producing evidence about the agent.
        print(
            f"\nUNJUDGEABLE: {len(unjudged)}/{args.rounds} round(s) produced no usable "
            "measurement:",
            file=sys.stderr,
        )
        for line in unjudged:
            print(f"  {line}", file=sys.stderr)
    if escaped or unjudged:
        return ExitCode.ERROR
    return ExitCode.OK


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
