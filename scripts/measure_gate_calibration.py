"""Live gate calibration: the real judge, the real agent, known-truth scaffolds (issue #8).

Runs the calibration scenario families from
:mod:`anvil.loop.calibration` end-to-end through ``run_round`` with the REAL
eval path — the real agent answering the real golden set, scored by the real
judges — and prints the confusion matrix the offline suite only simulates:
does the assembled gate KEEP what it should and REVERT what it should?

Each scenario materializes its baseline scaffold in a scratch git repo,
evaluates it once for the baseline, applies the scenario's mutation, and
runs one round. That is two evals per scenario (ten total at the default
five), so the default ``--mode quick`` (9 rows) is the sensible smoke; run
``--mode full`` when the result matters.

Opt-in by construction: nothing here runs in CI. Writes
``eval/runs/gate_calibration.json`` (or ``--out``).

Usage::

    .venv/bin/python scripts/measure_gate_calibration.py \
        --profile fe-vm-lakebase-praneeth --mode quick
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from anvil.cli import ExitCode, run_cli  # noqa: E402
from anvil.eval import evaluate_branch  # noqa: E402
from anvil.eval.cache import report_to_baseline, save_baseline  # noqa: E402
from anvil.loop import round as round_mod  # noqa: E402
from anvil.loop.calibration import (  # noqa: E402
    aggregate,
    make_session,
    result_from_report,
    scenarios,
)
from anvil.runtime.loader import default_runtime_config_path, load_endpoints  # noqa: E402

OUT_PATH = REPO_ROOT / "eval" / "runs" / "gate_calibration.json"


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def _make_repo(repo: Path, config_text: str) -> None:
    """A scratch repo carrying the REAL scaffold.

    The live run composes the real prompt, which requires a registered
    identity skill (and every registered file to exist) — the offline
    suite's stub scaffold world cannot drive it. Scenarios therefore mutate
    the real scaffold: the applier keeps ``harness.yaml`` registrations in
    sync on their mutations, and the one scenario whose BASELINE is crippled
    is prepared by hand below.
    """
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "calibration@anvil")
    _git(repo, "config", "user.name", "anvil-calibration")
    _git(repo, "config", "commit.gpgsign", "false")
    shutil.copytree(REPO_ROOT / "scaffold", repo / "scaffold")
    (repo / "harness").mkdir()
    (repo / "harness" / "config.yaml").write_text(config_text)
    (repo / "eval" / "runs").mkdir(parents=True)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    _git(repo, "checkout", "-q", "-b", "anvil/exp")


def _prepare_baseline(repo: Path, scenario) -> None:
    """Materialize the scenario's baseline scaffold onto the real one."""
    if scenario.name == "good_restore_citation":
        # The crippled world this scenario restores from: citation skill
        # gone, from the registry as well as the tree.
        (repo / "scaffold" / "skills" / "citation.md").unlink()
        harness = repo / "scaffold" / "harness.yaml"
        harness.write_text(harness.read_text().replace("- file: citation.md\n", ""))
        _git(repo, "add", ".")
        _git(repo, "commit", "-q", "-m", "baseline scaffold")
    # Other scenarios use the real scaffold as-is; the init commit already
    # covers it, and committing an unchanged tree exits non-zero.


def _live_scenarios(selected: list, real_citation: str) -> list:
    """Scenario definitions adjusted for the live world.

    The offline A/A rewrites the citation skill with stub content; live, it
    must rewrite it with the skill's REAL content — byte-identical is the
    whole point, and the stub text is not what is on disk.
    """
    import dataclasses

    return [
        dataclasses.replace(s, action_args={**s.action_args, "content": real_citation})
        if s.name == "aa"
        else s
        for s in selected
    ]


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default="DEFAULT")
    p.add_argument("--mode", default="quick", help="eval mode (quick is the smoke; full when it matters)")
    p.add_argument("--out", default=str(OUT_PATH))
    p.add_argument(
        "--scenarios",
        default="all",
        help="comma-separated scenario names, or 'all'",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    config_text = (REPO_ROOT / "harness" / "config.yaml").read_text(encoding="utf-8")
    runtime_endpoint, judge_endpoint = load_endpoints(
        default_runtime_config_path(REPO_ROOT / "scaffold")
    )
    all_scenarios = scenarios()
    if args.scenarios != "all":
        wanted = set(args.scenarios.split(","))
        all_scenarios = [s for s in all_scenarios if s.name in wanted]
        missing = wanted - {s.name for s in all_scenarios}
        if missing:
            raise SystemExit(f"unknown scenarios: {sorted(missing)}")

    real_citation = (REPO_ROOT / "scaffold" / "skills" / "citation.md").read_text(
        encoding="utf-8"
    )
    results = []
    for scenario in _live_scenarios(all_scenarios, real_citation):
        print(f"[{scenario.name}] {scenario.description} (truth: {scenario.truth})")
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            _make_repo(repo, config_text)
            _prepare_baseline(repo, scenario)

            baseline_report = evaluate_branch(
                scaffold_root=repo / "scaffold",
                profile=args.profile,
                mode=args.mode,
            )
            save_baseline(
                repo,
                report_to_baseline(
                    baseline_report,
                    scaffold_commit_sha=_git(repo, "rev-parse", "HEAD"),
                    runtime_endpoint=runtime_endpoint,
                    judge_endpoint=judge_endpoint,
                ),
            )
            print(f"  baseline aggregate {baseline_report.aggregate:.3f}; running round")

            with mock.patch.object(
                round_mod, "run_optimizer_session", make_session(scenario)
            ):
                report = round_mod.run_round(
                    round_id=1,
                    repo_root=repo,
                    profile=args.profile,
                    eval_mode=args.mode,
                )
            result = result_from_report(scenario, report)
            print(f"  -> {result.decision} (layer={result.rejecting_layer}, {result.paired_outcome})")
            results.append(result)

    matrix = aggregate(results)
    matrix["measured_at"] = datetime.now(UTC).isoformat()
    matrix["mode"] = args.mode
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")

    print("\n=== confusion matrix ===")
    print(f"TP={matrix['tp']} FN={matrix['fn']} FP={matrix['fp']} TN={matrix['tn']}")
    print(f"TPR={matrix['tpr']} FPR={matrix['fpr']}")
    print(f"by layer: {matrix['by_layer']}")
    print(f"underpowered: {matrix['underpowered']}  not-significant: {matrix['not_significant']}")
    print(f"written to {out}")
    return ExitCode.OK


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
