"""Whether a case was judged, and if not, why not.

The distinction this module exists to hold:

* A **failure** is an expectation that was assessed and not met. It is a fact
  about the agent, and it is signal.
* An **error** is an expectation that was never assessed. It is a fact about the
  infrastructure, and it is noise.

Today the harness cannot tell them apart. A prediction that raises is recorded as
an empty string (``eval/runner.py``), the judges score the absence of an answer
as a wrong answer, and the resulting near-zero moves the promotion gate exactly
as a genuinely bad answer would. So a rate-limited gateway reverts a good
mutation, and nothing in the round record says that is what happened.

The rule, from which everything else follows: **failures never become errors and
errors never become failures.** An error is excluded from the aggregate rather
than scored as zero, because the alternative is to assert that the agent answered
badly when in fact it never answered.

See ``docs/design/failure-vs-error.md`` for what the current code does, read out
of the mlflow source rather than inferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class CaseOutcome(StrEnum):
    """Terminal state of one evaluated case."""

    OK = "ok"
    """Assessed, and every expectation met."""

    FAILURE = "failure"
    """Assessed, and at least one expectation not met. Signal -- keep it."""

    ERROR = "error"
    """Not assessed: the prediction or the scorer raised. Excluded from scores."""

    SKIPPED = "skipped"
    """Deliberately not run (e.g. filtered out of the mode's row budget)."""

    INTERRUPTED = "interrupted"
    """Not run because the run was cancelled before reaching it."""


#: Outcomes that carry a score the aggregate may use. Anything else is excluded,
#: which is the whole point of the enum.
SCORABLE: frozenset[CaseOutcome] = frozenset({CaseOutcome.OK, CaseOutcome.FAILURE})


@dataclass(frozen=True)
class Attempt:
    """One invocation that raised, retained even when a later attempt succeeded.

    A round that only succeeded on its third try is not the same round as one
    that succeeded immediately -- the difference is a degrading endpoint, and it
    is worth being able to see it in the evidence rather than inferring it from
    a latency graph.
    """

    error_type: str
    error_message: str
    duration_ms: int = 0


@dataclass(frozen=True)
class CaseRecord:
    """The outcome of one case, with enough detail to explain itself.

    ``output`` is empty for an error. That is deliberate and is *not* the same as
    the old behaviour of recording ``""`` as the prediction: the outcome field
    says the case was never assessed, so nothing downstream will score the empty
    string.
    """

    case_id: str
    outcome: CaseOutcome
    output: str = ""
    error_type: str = ""
    error_message: str = ""
    attempts: tuple[Attempt, ...] = field(default_factory=tuple)
    duration_ms: int = 0

    def __post_init__(self) -> None:
        if self.outcome is CaseOutcome.ERROR and not self.error_type:
            raise ValueError("an error record must name its error_type")
        if self.outcome in SCORABLE and self.error_type:
            raise ValueError(
                f"{self.outcome} cannot carry an error_type -- a failure that was "
                "assessed is not an error, and conflating them is the bug this "
                "type exists to prevent"
            )

    @property
    def scorable(self) -> bool:
        return self.outcome in SCORABLE

    @classmethod
    def errored(
        cls,
        case_id: str,
        exc: BaseException,
        *,
        attempts: tuple[Attempt, ...] = (),
        duration_ms: int = 0,
    ) -> CaseRecord:
        """Build an error record from a caught exception."""
        return cls(
            case_id=case_id,
            outcome=CaseOutcome.ERROR,
            error_type=type(exc).__name__,
            error_message=str(exc),
            attempts=attempts,
            duration_ms=duration_ms,
        )


@dataclass(frozen=True)
class OutcomeSummary:
    """Counts per outcome, plus the error rate a round-level guard reads."""

    total: int
    ok: int
    failure: int
    error: int
    skipped: int
    interrupted: int

    @property
    def scorable(self) -> int:
        return self.ok + self.failure

    @property
    def error_rate(self) -> float:
        """Errors as a fraction of all cases. ``0.0`` for an empty run.

        A round whose error rate is high has not measured the agent, so the
        promotion gate must not read its aggregate -- a degraded gateway would
        otherwise be recorded as a bad mutation and revert good work.
        """
        return self.error / self.total if self.total else 0.0


def summarize(records: list[CaseRecord]) -> OutcomeSummary:
    """Tally ``records`` by outcome.

    Derived at read time rather than accumulated, so a partially written run
    summarises correctly from whatever records exist.
    """
    counts = dict.fromkeys(CaseOutcome, 0)
    for record in records:
        counts[record.outcome] += 1
    return OutcomeSummary(
        total=len(records),
        ok=counts[CaseOutcome.OK],
        failure=counts[CaseOutcome.FAILURE],
        error=counts[CaseOutcome.ERROR],
        skipped=counts[CaseOutcome.SKIPPED],
        interrupted=counts[CaseOutcome.INTERRUPTED],
    )
