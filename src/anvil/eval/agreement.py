"""Do the judges agree with humans, beyond chance?

Every gate mechanism — the frontier, the paired test, replication — assumes
the LLM judges are noisy but unbiased. Replication makes a biased judge
*worse*, not better: averaging shrinks variance around the wrong value, so a
systematically wrong judge produces more consistently-signed discordant
pairs and a *more* significant result (issue #16). The only way to see bias
is to score a human-labeled dataset and measure agreement.

That measurement is Cohen's kappa here, not accuracy. Accuracy hides class
imbalance: a judge that passes everything scores 96% on a dataset where 4%
of rows are inappropriate, while learning nothing. Kappa subtracts the
agreement chance alone would produce from what was observed, so a judge that
rubber-stamps gets ~0 no matter how lopsided the rows are.

The functions are pure and dependency-free (the bootstrap uses
``random.Random(seed)``, not numpy), so they are unit testable without a
workspace — the same posture as :mod:`anvil.eval.significance`.

Krippendorff's alpha (nominal, any number of raters) is here for the human
ceiling study: a judge cannot fairly be held to better agreement with
humans than humans reach with each other, so the ceiling comes first and
the judge number is read as a fraction of it.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Hashable, Sequence


def confusion(pairs: Sequence[tuple[Hashable, Hashable]]) -> dict[tuple[Hashable, Hashable], int]:
    """Count ``(reference, judged)`` verdict pairs.

    Reference first: ``(human, judge)``. The false-positive rate a judge
    report quotes is then ``pairs[(False, True)] / human-negatives`` — the
    judge said pass where the human said fail.
    """
    counts: dict[tuple[Hashable, Hashable], int] = Counter()
    for reference, judged in pairs:
        counts[(reference, judged)] += 1
    return dict(counts)


def cohens_kappa(pairs: Sequence[tuple[Hashable, Hashable]]) -> float:
    """Observed-minus-chance agreement over one minus chance.

    ``pairs`` are ``(reference, judged)`` verdicts over the same items, in
    any category set. Returns 1.0 on perfect agreement, ~0 at chance, and
    negative when the judge does worse than chance. Raises on an empty
    slice: a kappa over no rows is not 0, it is nothing.
    """
    n = len(pairs)
    if n == 0:
        raise ValueError("kappa over zero pairs is undefined, not zero")
    observed = sum(1 for reference, judged in pairs if reference == judged) / n
    reference_counts = Counter(reference for reference, _ in pairs)
    judged_counts = Counter(judged for _, judged in pairs)
    expected = sum(
        (reference_counts[cat] / n) * (judged_counts[cat] / n)
        for cat in set(reference_counts) | set(judged_counts)
    )
    if expected == 1.0:
        # expected == 1 only when both sides concentrated on the SAME single
        # category -- and then observed is 1 too, so the 1-expected
        # denominator never actually divides by zero.
        return 1.0
    return (observed - expected) / (1 - expected)


def bootstrap_kappa_ci(
    pairs: Sequence[tuple[Hashable, Hashable]],
    *,
    seed: int = 42,
    n_resamples: int = 1000,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for :func:`cohens_kappa`.

    Resamples items with replacement and recomputes kappa, so the interval
    carries the slice's own class balance and size rather than a normal
    approximation that misbehaves near kappa's bounds. Resamples that land
    on a single-category slice are dropped (their kappa is undefined);
    with fewer than half the resamples usable the slice is too degenerate
    for an interval at all, and this raises.
    """
    n = len(pairs)
    if n == 0:
        raise ValueError("bootstrap over zero pairs is undefined, not zero")
    rng = random.Random(seed)
    estimates: list[float] = []
    attempts = 0
    while len(estimates) < n_resamples and attempts < n_resamples * 3:
        attempts += 1
        resample = [pairs[rng.randrange(n)] for _ in range(n)]
        try:
            estimates.append(cohens_kappa(resample))
        except ValueError:
            continue
    if len(estimates) < n_resamples / 2:
        raise ValueError(
            f"slice too degenerate for a bootstrap interval: {len(estimates)} usable "
            f"resamples of {attempts} attempts"
        )
    estimates.sort()
    lo = estimates[int(0.025 * len(estimates))]
    hi = estimates[min(int(0.975 * len(estimates)), len(estimates) - 1)]
    return lo, hi


def rates(
    pairs: Sequence[tuple[bool, bool]],
) -> dict[str, float | int]:
    """The binary breakdown a judge report quotes.

    ``pairs`` are ``(human, judge)`` booleans where ``True`` means
    "pass / supported / appropriate". A false positive is the judge saying
    pass where the human said fail; a false negative is the reverse. Rates
    are over the corresponding human class, so they answer the operator's
    question directly: "when the row was bad, how often did the judge miss
    it" (FN), and "when the row was good, how often did the judge invent a
    problem" (FP).
    """
    n = len(pairs)
    if n == 0:
        raise ValueError("rates over zero pairs are undefined, not zero")
    human_pos = sum(1 for human, _ in pairs if human)
    human_neg = n - human_pos
    fp = sum(1 for human, judge in pairs if not human and judge)
    fn = sum(1 for human, judge in pairs if human and not judge)
    return {
        "n": n,
        "human_pos": human_pos,
        "human_neg": human_neg,
        "false_positives": fp,
        "false_negatives": fn,
        "fp_rate": fp / human_neg if human_neg else 0.0,
        "fn_rate": fn / human_pos if human_pos else 0.0,
        "observed_agreement": sum(1 for human, judge in pairs if human == judge) / n,
    }


def krippendorffs_alpha_nominal(
    ratings: Sequence[Sequence[Hashable | None]],
) -> float:
    """Krippendorff's alpha for nominal categories, any number of raters.

    ``ratings`` is one sequence per item, one value per rater; ``None``
    marks a rater who did not score that item (the overlap design needs
    this: not every annotator sees every item). Returns 1.0 on perfect
    agreement, ~0 at chance.

    Computed from the coincidence formulation: observed disagreement is the
    within-item pairwise disagreement rate; expected disagreement treats
    every (category, category) pair's pooled frequency as chance. Items
    scored by fewer than two raters carry no disagreement information and
    are skipped.
    """
    pair_counts: dict[tuple[Hashable, Hashable], int] = Counter()
    n_pairs_total = 0
    category_counts: dict[Hashable, int] = Counter()
    for item in ratings:
        present = [rating for rating in item if rating is not None]
        for category in present:
            category_counts[category] += 1
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                pair_counts[(present[i], present[j])] += 1
                pair_counts[(present[j], present[i])] += 1
                n_pairs_total += 2
    if n_pairs_total == 0:
        raise ValueError("alpha needs at least one item scored by two raters")

    observed_disagreement = (
        sum(count for (a, b), count in pair_counts.items() if a != b) / n_pairs_total
    )
    n_ratings = sum(category_counts.values())
    expected_disagreement = 1.0 - sum(
        (count / n_ratings) ** 2 for count in category_counts.values()
    )
    if expected_disagreement == 0.0:
        # A single category in the whole study: agreement is trivially
        # perfect, and alpha is undefined rather than 1.0 — same posture as
        # kappa's single-category case, but here there is nothing to
        # distinguish, so refuse to print a number that looks meaningful.
        raise ValueError("alpha is undefined when every rating is the same category")
    return 1.0 - observed_disagreement / expected_disagreement
