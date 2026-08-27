#!/usr/bin/env python3
"""Generate ``eval/runs/baseline.json`` for the current scaffold.

``scripts/evaluate.py --out`` writes an ``EvalReport`` — the eval
runner's own schema. The ANVIL loop's keep/revert gate, however,
reads a ``CachedBaseline`` from ``eval/runs/baseline.json`` (see
:func:`anvil.eval.load_baseline`). There was no supported path from a
fresh eval to that cache file, which blocked running ANVIL on a new
agent domain: without a baseline the gate has nothing to delta
against.

This script closes that gap. It runs the eval on the current
scaffold (the same logic as ``evaluate.py --mode quick``) and converts
the resulting ``EvalReport`` into a ``CachedBaseline``, sourcing the
three fields the eval does not know itself — ``scaffold_commit_sha``
from ``git rev-parse HEAD`` and ``runtime_endpoint`` /
``judge_endpoint`` from ``harness/config.yaml``.

Usage::

    uv run python scripts/make_baseline.py --mode quick
    uv run python scripts/make_baseline.py --mode standard --out eval/runs/baseline.json
    uv run python scripts/make_baseline.py --profile my-workspace --include-safety
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from anvil.cli import ExitCode, run_cli  # noqa: E402
from anvil.eval.cache import CachedBaseline, parent_path, report_to_baseline  # noqa: E402
from anvil.eval.judgeability import unjudgeable_reason_for  # noqa: E402
from anvil.eval.runner import evaluate_branch  # noqa: E402
from anvil.runtime.loader import (  # noqa: E402
    default_runtime_config_path,
    load_endpoints,
    load_eval_config,
)

# Default cache location the loop's gate reads. Relative to the repo
# root so ``--help`` shows a stable, machine-independent string.
DEFAULT_OUT_REL = "eval/runs/baseline.json"


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["quick", "standard", "full"],
        default=None,
        help="row sub-set size (default: harness/config.yaml > eval.default_mode)",
    )
    p.add_argument(
        "--out",
        default=None,
        help=f"output JSON path. Default: {DEFAULT_OUT_REL}",
    )
    p.add_argument("--profile", default="DEFAULT", help="Databricks CLI profile")
    p.add_argument(
        "--scaffold",
        default=str(REPO_ROOT / "scaffold"),
        help="path to scaffold/ directory",
    )
    # Must accept the same three data paths as scripts/evaluate.py: a baseline
    # cached from one domain is meaningless as the bar for rounds evaluated
    # against another.
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
        "--include-safety",
        action="store_true",
        help="evaluate Safety per-row (still excluded from aggregate)",
    )
    return p


def _git_head_sha(scaffold_root: Path) -> str:
    """``git rev-parse HEAD`` of the scaffold's repo (full 40-char SHA).

    Falls back to ``"unknown"`` if git is unavailable or the repo has
    no commits yet, mirroring ``evaluate.py``'s defensive SHA lookup.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(scaffold_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip()


def build_baseline(
    *,
    scaffold_root: Path | str,
    runtime_config_path: Path | str | None = None,
    kb_dir: Path | str = "data/kb",
    golden_set_path: Path | str = "data/golden_set.jsonl",
    evaluator_path: Path | str | None = None,
    profile: str = "DEFAULT",
    mode: str | None = None,
    include_safety: bool = False,
) -> CachedBaseline:
    """Run the eval on the current scaffold and build a ``CachedBaseline``.

    This is the programmatic entry point ``main()`` wraps; tests call
    it with ``evaluate_branch`` monkeypatched so no LLM is invoked.
    """
    scaffold_path = Path(scaffold_root)
    runtime_path = (
        Path(runtime_config_path)
        if runtime_config_path is not None
        else default_runtime_config_path(scaffold_path)
    )
    runtime_endpoint, judge_endpoint = load_endpoints(runtime_path)

    # Forward the resolved config path so the eval runs against the SAME
    # harness/config.yaml the endpoints above were read from. Without this
    # evaluate_branch() falls back to default_runtime_config_path() and the
    # recorded endpoints can diverge from the config that produced the eval
    # — a misleading baseline cache.
    report = evaluate_branch(
        scaffold_root=scaffold_path,
        runtime_config_path=runtime_path,
        kb_dir=kb_dir,
        golden_set_path=golden_set_path,
        evaluator_path=evaluator_path,
        profile=profile,
        mode=mode,
        include_safety=include_safety,
    )

    # The baseline is the bar every round is compared against and the frontier's
    # seed, so a degraded run must never be frozen into it. This matters MORE
    # since errored cases stopped being scored as zeros: a baseline run that
    # 429'd on six of eight rows used to produce a visibly broken ~0.25 that an
    # operator would rerun; now those rows are excluded and the same run reads
    # the mean of the two that survived -- higher than a healthy baseline, and
    # indistinguishable from one. Refuse instead.
    # Thresholds from ``runtime_path``, the same file the eval above ran under.
    # Resolving the default path here instead would judge the report against a
    # ceiling it was never measured under whenever a custom config is passed --
    # the same divergence the comment above guards the endpoints from.
    reason = unjudgeable_reason_for(report, load_eval_config(scaffold_path, runtime_path))
    if reason:
        raise RuntimeError(
            f"refusing to cache a baseline that cannot be judged: {reason}. "
            "Rerun once the endpoint is healthy."
        )

    scaffold_commit_sha = _git_head_sha(scaffold_path)
    return report_to_baseline(
        report,
        scaffold_commit_sha=scaffold_commit_sha,
        runtime_endpoint=runtime_endpoint,
        judge_endpoint=judge_endpoint,
    )


def main(argv: list[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)

    baseline = build_baseline(
        scaffold_root=args.scaffold,
        kb_dir=args.kb_dir,
        golden_set_path=args.golden_set_path,
        evaluator_path=args.evaluator_path,
        profile=args.profile,
        mode=args.mode,
        include_safety=args.include_safety,
    )

    out_path = Path(args.out) if args.out else REPO_ROOT / DEFAULT_OUT_REL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(baseline.to_dict(), indent=2) + "\n", encoding="utf-8")

    # Re-anchor the paired test's comparator: whatever parent.json holds was
    # drawn against the OLD baseline's world, and after a regen the parent of
    # the next round is the scaffold this baseline just measured. Deleting
    # (not overwriting) is what makes the next round fall back to the fresh
    # baseline rather than pairing against a draw from a superseded domain.
    stale_parent = parent_path(REPO_ROOT)
    if stale_parent.is_file():
        stale_parent.unlink()
        print("Cleared eval/runs/parent.json — the next round pairs against this baseline.")

    print(
        f"Baseline written to {out_path}. Aggregate: {baseline.aggregate:.4f}. "
        f"{baseline.n_examples} examples, {baseline.mode} mode."
    )
    return ExitCode.OK


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
