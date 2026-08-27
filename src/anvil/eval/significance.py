"""Is a score difference a real improvement, or is it the judge?

Two healthy runs of the **same** scaffold, the same rows, and the same model
scored `0.875` and `0.722` — about 0.15 of aggregate, from judge noise alone.
Every per-round gain the loop has actually produced was 0.03–0.06. So the gate,
which promotes on any strictly positive delta, was reading noise as signal about
as often as not, and fifty rounds of that is a random walk with a plausible
story attached.

The fix is not a bigger threshold. Raising ``gate.epsilon`` to 0.15 would reject
every real gain ever observed here, which trades a loop that promotes noise for
a loop that promotes nothing. What makes the difference measurable instead is
that the two runs answer **the same questions**: the scores pair row by row, and
a per-row difference cancels the row-difficulty variance that dominates the
aggregate. A question both runs get right contributes nothing either way; only
the rows where they *disagree* carry information.

That is a sign test. It asks: of the rows where the two runs disagreed, how many
went the candidate's way, and would a coin have done that as easily? It assumes
almost nothing — no normality, no equal variances, no known noise scale — which
is the right posture when the noise source is an LLM judge whose distribution
nobody has characterised.

**What the two runs are.** The candidate's draw, and the *current parent*
scaffold's most recent draw — ``eval/runs/parent.json``, rewritten on every
KEEP, with the frozen baseline standing in only until the first KEEP. The
comparison used to be against the frozen original baseline forever, which
from round two on answered "does the candidate differ from the original
scaffold" rather than "does it improve on its parent" (issue #19). Pairing
cancels row difficulty because both runs answer the same questions; it does
**not** cancel cross-session judge drift, because the parent's draw comes
from an earlier judge session. That limitation is accepted and recorded in
``docs/decisions.md``: the contemporaneous alternative (re-evaluate the
parent every round) doubles eval spend to control a drift the frontier gate
already tolerates by comparing best-so-far scores across sessions.

**One-sided, deliberately.** The question the gate asks is "is this better",
not "is this different". A two-sided test would spend half its significance
budget on the possibility that the mutation made things worse, which is not a
hypothesis the gate needs to distinguish from "no change" — both revert.

**What this cannot do.** On the 50-row dev partition a typical round yields
~30 discordant pairs, which detects a q=0.65–0.70 mutation with power
0.5–0.7 — and still misses smaller effects. On a smaller mode the test will
say "not significant" for many mutations that did help slightly. That is the
honest trade, and it is why ``gate.replicates`` exists — replication tightens
the per-row estimates so the same test can see smaller effects. A test that
cannot detect an effect reports exactly that, rather than a verdict it has
not earned.

``mlflow.genai.evaluate`` offers no ``seed``, no repetition count and no paired
mode (`docs/verified-api-surface.md`), so this is ANVIL's to build. It is pure,
dependency-free (an exact binomial tail via :func:`math.comb`, not a normal
approximation that misbehaves at these sample sizes), and therefore unit
testable without a workspace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Below this many discordant pairs, no sign test can reach any conventional
# alpha: with 4 discordant pairs a clean sweep is p = 0.0625. Reporting
# "underpowered" is a different statement from "not significant", and the gate
# needs to tell them apart -- one means the evidence is against the mutation,
# the other means there is not enough evidence to say.
MIN_DISCORDANT_PAIRS = 5

DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class PairedResult:
    """The outcome of one paired comparison."""

    n_pairs: int
    """Rows scored in both runs."""

    n_improved: int
    n_regressed: int
    n_tied: int

    p_value: float
    """One-sided exact binomial tail. ``1.0`` when nothing is comparable."""

    significant: bool
    """The improvement cleared ``alpha`` on enough discordant pairs."""

    underpowered: bool
    """Too few discordant pairs for any verdict to be reachable."""

    reason: str
    """Human-readable summary, for the round record and the operator."""

    @property
    def n_discordant(self) -> int:
        return self.n_improved + self.n_regressed


def one_sided_sign_test_p(n_improved: int, n_regressed: int) -> float:
    """P(at least ``n_improved`` of the discordant pairs by chance alone).

    Exact binomial tail at p=0.5 over the discordant pairs. Ties carry no
    directional information and are excluded — including them would dilute the
    test toward "no difference" in proportion to how easy the golden set is,
    making the verdict depend on the rows rather than on the mutation.
    """
    n = n_improved + n_regressed
    if n <= 0:
        return 1.0
    tail: int = sum(math.comb(n, k) for k in range(n_improved, n + 1))
    # ``1 << n`` rather than ``2**n``: identical for n >= 0, and it stays an
    # ``int`` for the type checker instead of the ``Any`` that ``**`` yields.
    return tail / (1 << n)


def row_aggregate(
    scores: dict[str, float],
    weights: dict[str, float],
    aggregate_scorer_names: list[str],
) -> float | None:
    """One row's weighted score, or ``None`` if no scorer applied to it.

    Weighted over the scorers that produced a value *for this row*, not over all
    configured scorers. Treating an inapplicable scorer as 0 here would make a
    refusal row -- where groundedness has nothing to be grounded in -- look
    uniformly worse than an answerable one, and the difference would move with
    the bucket mix rather than with the agent (D10).
    """
    present = [n for n in aggregate_scorer_names if n in scores]
    total_weight = sum(weights.get(n, 1.0) for n in present)
    if not present or total_weight <= 0:
        return None
    return sum(scores[n] * weights.get(n, 1.0) for n in present) / total_weight


def paired_sign_test(
    baseline_per_row: dict[str, dict[str, float]],
    candidate_per_row: dict[str, dict[str, float]],
    *,
    weights: dict[str, float],
    aggregate_scorer_names: list[str],
    alpha: float = DEFAULT_ALPHA,
    min_discordant_pairs: int = MIN_DISCORDANT_PAIRS,
) -> PairedResult:
    """Compare two runs row by row on the weighted per-row aggregate.

    Only ``example_id``s present in both runs and scorable in both are paired. A
    row that errored in one run is not evidence about the mutation, and dropping
    it from *both* sides is what keeps the comparison paired -- scoring it as
    zero on one side would manufacture a difference out of an infrastructure
    failure, which is the whole failure-vs-error argument (D7) applied to the
    gate.
    """
    improved = regressed = tied = 0
    for example_id, base_scores in baseline_per_row.items():
        cand_scores = candidate_per_row.get(example_id)
        if cand_scores is None:
            continue
        base = row_aggregate(base_scores, weights, aggregate_scorer_names)
        cand = row_aggregate(cand_scores, weights, aggregate_scorer_names)
        if base is None or cand is None:
            continue
        if cand > base:
            improved += 1
        elif cand < base:
            regressed += 1
        else:
            tied += 1

    n_pairs = improved + regressed + tied
    n_discordant = improved + regressed
    p_value = one_sided_sign_test_p(improved, regressed)

    if n_pairs == 0:
        return PairedResult(
            n_pairs=0,
            n_improved=0,
            n_regressed=0,
            n_tied=0,
            p_value=1.0,
            significant=False,
            underpowered=True,
            reason=(
                "no rows could be paired between the two runs: either the "
                "baseline predates per-row scores or the two runs share no "
                "scorable example_id"
            ),
        )

    underpowered = n_discordant < min_discordant_pairs
    # ``p_value <= alpha`` already implies ``improved > regressed``: the tail is
    # measured upward from ``improved``, so an even split gives p > 0.5. An extra
    # ``improved > regressed`` conjunct would be a branch no input can exercise,
    # which is worse than nothing -- it reads as a guard while testing nothing.
    significant = (not underpowered) and p_value <= alpha

    if underpowered:
        reason = (
            f"{n_discordant} discordant pair(s) of {n_pairs} is too few for a "
            f"sign test to reach alpha={alpha:g} (a clean sweep of "
            f"{n_discordant} gives p={one_sided_sign_test_p(n_discordant, 0):.4f}); "
            f"the comparison cannot distinguish this mutation from noise. "
            f"Raise gate.replicates to tighten the per-row estimates, or use a "
            f"mode with more rows"
        )
    elif significant:
        reason = (
            f"{improved} improved / {regressed} regressed / {tied} tied over "
            f"{n_pairs} paired rows, p={p_value:.4f} <= alpha={alpha:g}"
        )
    else:
        reason = (
            f"{improved} improved / {regressed} regressed / {tied} tied over "
            f"{n_pairs} paired rows, p={p_value:.4f} > alpha={alpha:g}: "
            f"consistent with judge noise"
        )

    return PairedResult(
        n_pairs=n_pairs,
        n_improved=improved,
        n_regressed=regressed,
        n_tied=tied,
        p_value=p_value,
        significant=significant,
        underpowered=underpowered,
        reason=reason,
    )
