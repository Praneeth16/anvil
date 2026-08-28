"""Known-answer tests for anvil.eval.agreement.

The statistics here feed the judge-validation report (issue #16): a kappa
that computes the wrong number would make a biased judge look healthy or a
healthy judge look biased, and nothing downstream would notice. Every test
is a hand-computed table, not a property the implementation also believes.
"""

from __future__ import annotations

import pytest

from anvil.eval.agreement import (
    bootstrap_kappa_ci,
    cohens_kappa,
    confusion,
    krippendorffs_alpha_nominal,
    rates,
)


@pytest.mark.unit
def test_kappa_is_one_on_perfect_agreement() -> None:
    pairs = [(True, True), (False, False), (True, True), (False, False)]
    assert cohens_kappa(pairs) == 1.0


@pytest.mark.unit
def test_kappa_is_zero_at_chance() -> None:
    """A judge that passes everything on a half-positive slice agrees 50%,
    and chance expects exactly 50%: kappa 0, rubber-stamping detected."""
    pairs = [(True, True), (True, True), (False, True), (False, True)]
    assert cohens_kappa(pairs) == pytest.approx(0.0)


@pytest.mark.unit
def test_kappa_hand_computed_2x2() -> None:
    """The canonical worked example: 20 agree-yes, 15 agree-no, 10 yes/no,
    5 no/yes. observed = 35/50 = 0.7; marginals (25, 25) x (30, 20) give
    expected = 0.5*0.6 + 0.5*0.4 = 0.5; kappa = 0.2/0.5 = 0.4."""
    pairs = (
        [(True, True)] * 20
        + [(False, False)] * 15
        + [(True, False)] * 10
        + [(False, True)] * 5
    )
    assert cohens_kappa(pairs) == pytest.approx(0.4)


@pytest.mark.unit
def test_kappa_is_negative_below_chance() -> None:
    pairs = [(True, False), (False, True)] * 5
    assert cohens_kappa(pairs) < 0


@pytest.mark.unit
def test_kappa_refuses_an_empty_slice() -> None:
    with pytest.raises(ValueError, match="zero pairs"):
        cohens_kappa([])


@pytest.mark.unit
def test_kappa_single_category_is_one() -> None:
    """Both sides on the same single category: expected agreement IS 1, and
    so is observed — the 1-expected denominator never divides by zero."""
    assert cohens_kappa([(True, True)] * 4) == 1.0
    # Opposite single categories: chance expects 0 agreement and gets it.
    assert cohens_kappa([(True, False)] * 4) == 0.0


@pytest.mark.unit
def test_confusion_counts_reference_first() -> None:
    counts = confusion([(True, True), (False, True), (True, False), (True, True)])
    assert counts == {(True, True): 2, (False, True): 1, (True, False): 1}


@pytest.mark.unit
def test_rates_break_out_fp_and_fn_over_the_human_classes() -> None:
    """4 human-bad rows, judge misses 1 (FP=1/4); 6 human-good rows, judge
    fails 2 (FN=2/6). Rates are over the human class, not the whole slice."""
    pairs = [(False, False)] * 3 + [(False, True)] + [(True, True)] * 4 + [(True, False)] * 2
    out = rates(pairs)
    assert out["n"] == 10
    assert out["fp_rate"] == pytest.approx(1 / 4)
    assert out["fn_rate"] == pytest.approx(2 / 6)
    assert out["observed_agreement"] == pytest.approx(0.7)


@pytest.mark.unit
def test_bootstrap_ci_contains_the_point_estimate_and_is_reproducible() -> None:
    pairs = [(True, True)] * 20 + [(False, False)] * 15 + [(True, False)] * 10 + [
        (False, True)
    ] * 5
    lo, hi = bootstrap_kappa_ci(pairs, seed=7)
    assert lo <= cohens_kappa(pairs) <= hi
    assert (lo, hi) == bootstrap_kappa_ci(pairs, seed=7)


@pytest.mark.unit
def test_alpha_is_one_on_perfect_agreement() -> None:
    ratings = [("pass", "pass"), ("fail", "fail"), ("pass", "pass"), ("fail", "fail")]
    assert krippendorffs_alpha_nominal(ratings) == 1.0


@pytest.mark.unit
def test_alpha_hand_computed_two_raters() -> None:
    """Six items, two raters: agree on 4, disagree on 2, balanced categories.

    observed disagreement = 4 ordered disagreeing pairs / 12 = 1/3.
    categories are balanced, so expected disagreement = 1 - (0.5^2 + 0.5^2) = 0.5.
    alpha = 1 - (1/3)/0.5 = 1/3.
    """
    ratings = [
        ("a", "a"),
        ("b", "b"),
        ("a", "a"),
        ("b", "b"),
        ("a", "b"),
        ("b", "a"),
    ]
    assert krippendorffs_alpha_nominal(ratings) == pytest.approx(1 / 3)


@pytest.mark.unit
def test_alpha_skips_items_fewer_than_two_raters() -> None:
    ratings = [("a", None), ("a", "a"), (None, None), ("b", "b")]
    assert krippendorffs_alpha_nominal(ratings) == 1.0


@pytest.mark.unit
def test_alpha_refuses_a_single_category_study() -> None:
    with pytest.raises(ValueError, match="same category"):
        krippendorffs_alpha_nominal([("a", "a"), ("a", "a")])


@pytest.mark.unit
def test_alpha_refuses_a_study_with_no_overlap() -> None:
    with pytest.raises(ValueError, match="two raters"):
        krippendorffs_alpha_nominal([("a", None), (None, "b")])
