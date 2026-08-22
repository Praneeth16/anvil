#!/usr/bin/env python3
"""CLI driver for ``anvil.eval.runner.evaluate_branch``.

Usage::

    uv run python scripts/evaluate.py --mode quick
    uv run python scripts/evaluate.py --mode standard --profile my-workspace
    uv run python scripts/evaluate.py --mode full --include-safety

Persists the report to ``eval/runs/round_NNN.json`` (or to the path
given by ``--out``).

Exit status (see :mod:`anvil.cli`): ``0`` every case assessed and met its
expectations, ``1`` cases assessed and some did not, ``2`` the error rate was
above ``eval.max_error_rate`` so the run did not measure the agent, ``130``
interrupted.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from anvil.cli import exit_code_for_report, run_cli  # noqa: E402
from anvil.eval.runner import evaluate_branch  # noqa: E402
from anvil.runtime.loader import load_eval_config  # noqa: E402


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=["quick", "standard", "full"],
        default=None,
        help="row sub-set size (default: harness/config.yaml > eval.default_mode)",
    )
    p.add_argument("--profile", default="DEFAULT", help="Databricks CLI profile")
    p.add_argument(
        "--scaffold",
        default=str(REPO_ROOT / "scaffold"),
        help="path to scaffold/ directory",
    )
    p.add_argument(
        "--out",
        default=None,
        help="output JSON path. Default: eval/runs/eval_<mode>_<sha8>.json",
    )
    p.add_argument(
        "--include-safety",
        action="store_true",
        help="evaluate Safety per-row (still excluded from aggregate)",
    )
    p.add_argument(
        "--label",
        default="",
        help="free-form label to embed in the output JSON",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)

    report = evaluate_branch(
        scaffold_root=args.scaffold,
        profile=args.profile,
        mode=args.mode,
        include_safety=args.include_safety,
    )

    payload: dict = asdict(report)
    if args.label:
        payload["label"] = args.label

    if args.out:
        out_path = Path(args.out)
    else:
        sha = ""
        try:
            import subprocess

            sha = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            ).stdout.strip()[:8]
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            sha = "unknown"
        out_path = REPO_ROOT / "eval" / "runs" / f"eval_{report.mode}_{sha}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"== eval done: mode={report.mode} n={report.n_rows}")
    print(f"   aggregate: {report.aggregate:.3f}")
    for name, value in report.per_judge.items():
        print(f"   {name:>26}: {value:.3f}")
    if report.n_errors:
        # Printed next to the aggregate, never folded into it: the aggregate is
        # the mean over the cases that ran, and how many did not is what says
        # whether it is worth reading.
        print(f"   {'unassessed (errors)':>26}: {report.n_errors} ({report.error_rate:.0%})")
    print(f"== written to: {out_path}")
    print(f"== mlflow run: {report.run_id}")

    max_error_rate = load_eval_config(args.scaffold).max_error_rate
    code = exit_code_for_report(report, max_error_rate=max_error_rate)
    if code:
        print(f"== exit {int(code)}: {len(report.failures)} failed, {report.n_errors} errored")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
