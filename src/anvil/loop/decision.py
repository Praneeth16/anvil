"""Keep / revert / noop / infra_fail — the round's terminal verdict.

Pure function: takes the score delta, the action kind, and the
parser status, and returns one of four states. The loop runner
applies the verdict (ff-merge | branch -D | leave-as-noop |
mark-infra-fail).

Strict positive deltas are kept. A tie counts as a revert (zero
gradient is not an improvement and a kept tie clutters the loop's
search history).
"""

from __future__ import annotations

from enum import StrEnum


class Decision(StrEnum):
    KEEP = "keep"
    REVERT = "revert"
    NOOP = "noop"
    INFRA_FAIL = "infra_fail"


def decide(
    *,
    score_delta: float | None,
    action_kind: str,
    parse_status: str,
    eval_failed: bool = False,
) -> Decision:
    """Compute the round's terminal decision.

    Order matters: an explicit ``noop`` from the optimizer wins over
    everything (we trust the optimizer's choice not to mutate); an
    eval-side infrastructure failure beats any score consideration;
    only then do we look at the score delta.
    """
    if action_kind == "noop":
        return Decision.NOOP
    if eval_failed or score_delta is None:
        return Decision.INFRA_FAIL
    if score_delta > 0:
        return Decision.KEEP
    return Decision.REVERT
