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

Where 1 becomes 2 is deliberately the round's own ``eval.max_error_rate``
ceiling rather than a second threshold invented here, so a red CI run and a
reverted round mean the same thing.
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


class ExitCode(IntEnum):
    """Process exit statuses. See the module docstring."""

    OK = 0
    FAILURES = 1
    ERROR = 2
    INTERRUPTED = 130


def exit_code_for_report(report: EvalReport, *, max_error_rate: float = 0.2) -> ExitCode:
    """Map an :class:`~anvil.eval.runner.EvalReport` to an exit status.

    Errors outrank failures: a run that could not assess most of its cases has
    not measured the agent, so the failures it did record are not worth gating
    on. Below the ceiling an errored case is excluded from the score and the run
    is still a measurement -- reporting :attr:`ExitCode.ERROR` for every stray
    timeout would make a flaky afternoon indistinguishable from a broken
    harness, and a status that cries wolf gets ignored.
    """
    if report.error_rate > max_error_rate:
        return ExitCode.ERROR
    if report.failures:
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
