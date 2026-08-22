"""Exit-status contract shared by the ``scripts/*.py`` CLIs.

Four codes, borrowed from the heddle contracts:

======  =========================================================
``0``   Every case was assessed and met its expectations.
``1``   Cases were assessed and some did not meet expectations.
``2``   The run could not produce a usable answer.
``130`` An operator interrupted it (``128 + SIGINT``).
======  =========================================================

The load-bearing distinction is 1 vs 2, and it is the same one as failure vs
error a layer down (:mod:`anvil.eval.outcome`): a caller that cannot tell "the
agent scored badly" from "the eval never ran" cannot automate anything on top of
this harness. Every one of these scripts used to return ``0`` unless it hit an
argument error, which is why none of them could be used as a CI gate.

What makes a run unusable is deliberately the round gate's own definition
(:func:`anvil.eval.judgeability.unjudgeable_reason`) rather than a second one
invented here, so a red CI run and a reverted round mean the same thing.

``1`` is **opt-in** for an eval, via ``--gate-on-failures``. Almost every real
eval has some case scoring below 1.0, so returning ``1`` by default would make
the documented ``scripts/evaluate.py --mode quick`` invocation abort any
``set -e`` wrapper and read as broken to an operator running it by hand. A status
that fires on correct usage gets ignored, or worse, worked around. ``2`` is not
opt-in: a run that could not measure the agent is always an error.
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable
from enum import IntEnum
from typing import TYPE_CHECKING

from anvil.eval.outcome import RunInterrupted, summarize

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance, typing only
    from anvil.eval.runner import EvalReport
    from anvil.runtime.models import EvalConfig


class ExitCode(IntEnum):
    """Process exit statuses. See the module docstring."""

    OK = 0
    FAILURES = 1
    ERROR = 2
    INTERRUPTED = 130


def exit_code_for_report(
    report: EvalReport,
    *,
    eval_config: EvalConfig | None = None,
    gate_on_failures: bool = False,
) -> ExitCode:
    """Map an :class:`~anvil.eval.runner.EvalReport` to an exit status.

    Errors outrank failures: a run that could not assess enough of its cases has
    not measured the agent, so the failures it did record are not worth gating
    on. A run that stays inside the judgeability thresholds is still a
    measurement even if a case errored -- reporting :attr:`ExitCode.ERROR` for
    every stray timeout would make a flaky afternoon indistinguishable from a
    broken harness, and a status that cries wolf gets ignored.

    ``gate_on_failures`` is off by default; see the module docstring for why
    ``1`` is opt-in and ``2`` is not.
    """
    from anvil.eval.judgeability import unjudgeable_reason, unjudgeable_reason_for

    reason = (
        unjudgeable_reason_for(report, eval_config)
        if eval_config is not None
        else unjudgeable_reason(report)
    )
    if reason:
        return ExitCode.ERROR
    if gate_on_failures and report.failures:
        return ExitCode.FAILURES
    return ExitCode.OK


def run_cli(main: Callable[[], int]) -> int:
    """Call ``main`` and translate what escapes it into an exit status.

    ``SystemExit`` passes through untouched: ``argparse`` raises it for
    ``--help`` and for a usage error, and catching it would turn ``--help`` into
    a traceback.

    An unexpected exception still prints its traceback. Swallowing it to return
    a tidy code would trade a debuggable crash for an undebuggable one, and the
    exit status is for the caller, not for the person reading the terminal.
    """
    try:
        return int(main())
    except RunInterrupted as interrupted:
        summary = summarize(interrupted.records)
        print(
            f"\nINTERRUPTED: {summary.scorable}/{summary.total} cases completed, "
            f"{summary.interrupted} not reached.",
            file=sys.stderr,
        )
        return ExitCode.INTERRUPTED
    except KeyboardInterrupt:
        print("\nINTERRUPTED.", file=sys.stderr)
        return ExitCode.INTERRUPTED
    except Exception:
        traceback.print_exc()
        return ExitCode.ERROR
