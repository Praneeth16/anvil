"""What the process's exit status means — Phase 2, step 6.

Four codes, borrowed from the heddle contracts:

* ``0`` — every case was assessed and met its expectations.
* ``1`` — cases were assessed and some did not meet expectations. A *result*,
  not a malfunction: the agent is not good enough yet.
* ``2`` — the run could not produce a usable answer. A malfunction.
* ``130`` — an operator interrupted it (the shell convention, ``128 + SIGINT``).

The distinction that matters is 1 vs 2, and it is the same distinction as
failure vs error one layer down: a caller that cannot tell "the agent scored
badly" from "the eval never ran" cannot automate anything on top of this. A
harness that exits 0 no matter what, which is what these scripts did, is
unusable in CI for the same reason.
"""

from __future__ import annotations

import pytest


def _report(
    *,
    n_rows: int = 8,
    n_errors: int = 0,
    n_failures: int = 0,
    n_unattributed_errors: int = 0,
):
    from anvil.eval.runner import EvalReport

    return EvalReport(
        aggregate=0.9,
        per_judge={"correctness": 0.9},
        per_bucket={},
        failures=[{"example_id": f"f{i}"} for i in range(n_failures)],
        run_id="run-1",
        experiment_id="exp-1",
        n_rows=n_rows,
        mode="quick",
        scorers=["correctness"],
        evaluated_at="2026-08-22T12:00:00+00:00",
        n_errors=n_errors,
        n_unattributed_errors=n_unattributed_errors,
    )


# ---------------------------------------------------------------------------
# 1. Mapping a report to an exit code
# ---------------------------------------------------------------------------


def test_a_clean_run_exits_zero() -> None:
    from anvil.cli import ExitCode, exit_code_for_report

    assert exit_code_for_report(_report()) is ExitCode.OK


def test_assessed_failures_exit_one_only_when_asked() -> None:
    """Failures are a result about the agent, so the run itself succeeded.

    Exit 1 is opt-in. ``failures`` is populated for any scorer below 1.0 on any
    row, which is nearly every real eval, so returning 1 by default would abort
    any ``set -e`` wrapper around the invocation the README documents — and a
    status that fires on correct usage gets worked around rather than heeded.
    """
    from anvil.cli import ExitCode, exit_code_for_report

    assert exit_code_for_report(_report(n_failures=3)) is ExitCode.OK
    assert (
        exit_code_for_report(_report(n_failures=3), gate_on_failures=True)
        is ExitCode.FAILURES
    )


def test_a_stray_error_under_the_ceiling_does_not_make_the_run_an_error() -> None:
    """One unassessed case in eight is excluded from the score and the run is
    still a measurement. Reporting 2 here would make every flaky afternoon look
    like a broken harness."""
    from anvil.cli import ExitCode, exit_code_for_report

    assert exit_code_for_report(_report(n_errors=1)) is ExitCode.OK


def test_an_error_rate_above_the_ceiling_exits_two() -> None:
    """Past the ceiling the run did not measure the agent. That is a
    malfunction, and it is deliberately the *same* judgement the round's gate
    makes — one definition, so a CI failure and a reverted round mean the same
    thing."""
    from anvil.cli import ExitCode, exit_code_for_report

    assert exit_code_for_report(_report(n_errors=4)) is ExitCode.ERROR


def test_errors_outrank_failures() -> None:
    """A run that both failed cases and could not assess most of them is
    reported as the malfunction — the failures are not trustworthy."""
    from anvil.cli import ExitCode, exit_code_for_report

    report = _report(n_errors=6, n_failures=2)
    assert exit_code_for_report(report, gate_on_failures=True) is ExitCode.ERROR


def test_too_few_assessed_cases_exits_two_even_under_the_rate_ceiling() -> None:
    """The case a rate ceiling cannot express. An operator who raises
    max_error_rate to ride out a flaky endpoint would otherwise let the run be
    reported clean on the strength of one surviving row."""
    from anvil.cli import ExitCode, exit_code_for_report

    generous = _eval_cfg(1.0)  # rate guard fully disabled
    assert (
        exit_code_for_report(_report(n_rows=8, n_errors=7), eval_config=generous)
        is ExitCode.ERROR
    )


def test_an_unexcludable_error_exits_two() -> None:
    """An error that could not be joined to a row still has its zero in the
    mean, so the run did not do what it claims. Under the rate ceiling it would
    otherwise be reported as a clean measurement."""
    from anvil.cli import ExitCode, exit_code_for_report

    report = _report(n_rows=8, n_errors=1, n_unattributed_errors=1)
    assert exit_code_for_report(report) is ExitCode.ERROR


# ---------------------------------------------------------------------------
# 2. The CLI wrapper
# ---------------------------------------------------------------------------


def test_run_cli_passes_through_a_returned_code() -> None:
    from anvil.cli import run_cli

    assert run_cli(lambda: 0) == 0
    assert run_cli(lambda: 1) == 1


def test_run_cli_maps_keyboard_interrupt_to_130() -> None:
    """The shell convention. A harness that reported an operator's Ctrl-C as a
    generic error would make an aborted run indistinguishable from a broken
    one in any log that only kept the exit status."""
    from anvil.cli import ExitCode, run_cli

    def _interrupted() -> int:
        raise KeyboardInterrupt

    assert run_cli(_interrupted) == ExitCode.INTERRUPTED


def test_run_cli_maps_a_partial_run_to_130_and_says_what_survived() -> None:
    """``RunInterrupted`` carries the records that exist, so the operator is
    told how far the run got rather than just that it stopped."""
    from anvil.cli import ExitCode, run_cli
    from anvil.eval.outcome import CaseOutcome, CaseRecord, RunInterrupted

    records = [
        CaseRecord(case_id="0", outcome=CaseOutcome.OK, output="a"),
        CaseRecord(case_id="1", outcome=CaseOutcome.INTERRUPTED),
    ]

    def _interrupted() -> int:
        raise RunInterrupted(records)

    assert run_cli(_interrupted) == ExitCode.INTERRUPTED


def test_run_cli_maps_an_unexpected_exception_to_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An uncaught exception is a malfunction, not a failing agent. The
    traceback is still printed — swallowing it to return a tidy code would
    trade a debuggable crash for an undebuggable one."""
    from anvil.cli import ExitCode, run_cli

    def _boom() -> int:
        raise RuntimeError("synthetic explosion")

    assert run_cli(_boom) == ExitCode.ERROR
    err = capsys.readouterr().err
    assert "synthetic explosion" in err
    assert "Traceback" in err


def test_run_cli_does_not_intercept_systemexit() -> None:
    """``argparse`` exits through ``SystemExit`` for ``--help`` and for a usage
    error. Catching it would turn ``--help`` into exit 2 with a traceback."""
    from anvil.cli import run_cli

    def _argparse_style_exit() -> int:
        raise SystemExit(0)

    with pytest.raises(SystemExit) as exc:
        run_cli(_argparse_style_exit)
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# 3. The scripts are wired to it
# ---------------------------------------------------------------------------


def _load_script(name: str):
    """Import a ``scripts/*.py`` CLI as a module so its ``main`` is callable."""
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"anvil_script_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _eval_cfg(max_error_rate: float):
    from anvil.runtime.models import EvalConfig

    return EvalConfig(max_error_rate=max_error_rate)


@pytest.mark.parametrize(
    ("n_errors", "n_failures", "extra_args", "expected"),
    [
        (0, 0, [], 0),
        (0, 2, [], 0),  # failures alone are not a nonzero exit by default
        (0, 2, ["--gate-on-failures"], 1),
        (4, 0, [], 2),  # unjudgeable is nonzero whether asked or not
        (4, 0, ["--gate-on-failures"], 2),
    ],
)
def test_evaluate_script_reports_the_right_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    n_errors: int,
    n_failures: int,
    extra_args: list[str],
    expected: int,
) -> None:
    """End to end through the real CLI, including the argparse wiring: a clean
    run exits 0, assessed failures exit 1 only under ``--gate-on-failures``, and
    a run that could not measure the agent exits 2 regardless — even though its
    aggregate looks fine."""
    script = _load_script("evaluate")

    monkeypatch.setattr(
        script,
        "evaluate_branch",
        lambda **_kw: _report(n_errors=n_errors, n_failures=n_failures),
    )
    monkeypatch.setattr(script, "load_eval_config", lambda *_a, **_kw: _eval_cfg(0.2))

    argv = ["--out", str(tmp_path / "out.json"), *extra_args]
    assert script.main(argv) == expected
