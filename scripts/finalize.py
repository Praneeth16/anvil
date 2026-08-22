#!/usr/bin/env python3
"""Run the one-time held-out evaluation and lock further optimization.

Usage::

    uv run python scripts/finalize.py [--mode test]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from anvil.cli import ExitCode, run_cli  # noqa: E402
from anvil.eval.judgeability import unjudgeable_reason_for  # noqa: E402
from anvil.eval.runner import EvalReport, evaluate_branch  # noqa: E402
from anvil.loop.frontier import Frontier, load_frontier  # noqa: E402
from anvil.runtime.models import RuntimeYAML  # noqa: E402

DEFAULT_OUT_REL = "eval/runs/finalized.json"


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["test"], default="test")
    parser.add_argument("--profile", default="DEFAULT", help="Databricks CLI profile")
    parser.add_argument("--scaffold", default=str(REPO_ROOT / "scaffold"), help="path to scaffold/")
    parser.add_argument("--out", default=None, help=f"default: {DEFAULT_OUT_REL}")
    parser.add_argument("--include-safety", action="store_true")
    return parser


def _git_head_sha(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return "unknown"
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def finalize(
    *,
    repo_root: Path | str,
    scaffold_root: Path | str,
    profile: str = "DEFAULT",
    mode: str = "test",
    include_safety: bool = False,
) -> dict:
    """Evaluate HEAD on the held-out set and return the finalized payload."""
    root = Path(repo_root)
    config_path = root / "harness" / "config.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = RuntimeYAML.model_validate(raw)
    if not config.eval.held_out_test:
        raise RuntimeError("held-out finalization is disabled; set eval.held_out_test: true")

    frontier: Frontier | None = load_frontier(root)
    if frontier is None:
        raise RuntimeError("cannot finalize without eval/runs/frontier.json")

    report: EvalReport = evaluate_branch(
        scaffold_root=scaffold_root,
        runtime_config_path=config_path,
        profile=profile,
        mode=mode,
        allow_test=True,
        include_safety=include_safety,
    )

    # This is the highest-stakes number the harness produces: the held-out score
    # of the finished agent, run once, on the split nothing else is allowed to
    # touch. And ``main`` refuses to overwrite the file afterwards, so a
    # degraded run does not merely mislead -- it locks in and has to be deleted
    # by hand. Refuse before writing rather than after.
    reason = unjudgeable_reason_for(report, config.eval)
    if reason:
        raise RuntimeError(
            f"refusing to finalize on an eval that cannot be judged: {reason}. "
            "The held-out set is single-use by design, so rerun this only once "
            "the endpoint is healthy."
        )

    return {
        **asdict(report),
        "scaffold_commit_sha": _git_head_sha(root),
        "finalized_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "frontier": frontier.to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    out_path = Path(args.out) if args.out else REPO_ROOT / DEFAULT_OUT_REL
    if out_path.is_file():
        print(f"ERROR: finalization already exists at {out_path}")
        return ExitCode.ERROR

    try:
        payload = finalize(
            repo_root=REPO_ROOT,
            scaffold_root=args.scaffold,
            profile=args.profile,
            mode=args.mode,
            include_safety=args.include_safety,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return ExitCode.ERROR

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Finalized {payload['scaffold_commit_sha']} on {payload['n_rows']} held-out "
        f"rows. Aggregate: {payload['aggregate']:.4f}."
    )
    for name, score in payload["per_judge"].items():
        print(f"  {name}: {score:.4f}")
    print(f"Written to {out_path}")
    return ExitCode.OK


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
