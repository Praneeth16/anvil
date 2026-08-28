"""Offline tests for scripts/measure_judge_agreement.py (issue #16).

The live script scores RAGTruth rows with the real judges; what these tests
pin is everything around the calls — verdict extraction, human/judge
pairing, inapplicability, error accounting, the report's shape — with fake
scorers, so a green run proves the pairing logic without a workspace.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def module():
    spec = importlib.util.spec_from_file_location(
        "measure_judge_agreement_script",
        REPO_ROOT / "scripts" / "measure_judge_agreement.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    return mod


def _row(rid: str, *, supported: bool, refusal_incorrect: bool = False) -> dict:
    return {
        "example_id": rid,
        "ragtruth_id": rid,
        "task_type": "QA",
        "query": f"question {rid}",
        "response": f"response {rid}",
        "passages": [f"passage {rid}"],
        "labels": {
            "supported": supported,
            "should_refuse": False,
            "refusal_incorrect": refusal_incorrect,
        },
        "stratum": "QA/supported" if supported else "QA/unsupported",
    }


class _FakeGroundedness:
    """Agrees with the human label except on 'u2' (a false negative)."""

    name = "retrieval_groundedness"

    def run(self, *, inputs, outputs, expectations, trace):
        if expectations["expected_doc_ids"] == ["ragtruth:u2"]:
            return SimpleNamespace(value="yes")
        if expectations["expected_doc_ids"] in (["ragtruth:u1"], ["ragtruth:s2"]):
            return SimpleNamespace(value="no")
        return SimpleNamespace(value="yes")


class _FakeRefusal:
    """Fails the incorrect-refusal row (correct) and one answer row (FP)."""

    name = "refusal_appropriateness"

    def run(self, *, inputs, outputs, expectations, trace):
        if inputs["query"] == "question ref1":
            return SimpleNamespace(value=False)
        if inputs["query"] == "question s2":
            return SimpleNamespace(value=False)
        return SimpleNamespace(value=True)


def _scorers():
    return [_FakeGroundedness(), _FakeRefusal()]


@pytest.fixture
def no_trace(module, monkeypatch: pytest.MonkeyPatch):
    """Traces are scaffolding for the groundedness extraction; fake them."""
    monkeypatch.setattr(module, "_synthesize_trace", lambda *a, **kw: object())


@pytest.mark.unit
def test_measure_pairs_verdicts_with_human_labels(module, no_trace) -> None:
    rows = [
        _row("s1", supported=True),
        _row("s2", supported=True),
        _row("u1", supported=False),
        _row("u2", supported=False),
        _row("ref1", supported=True, refusal_incorrect=True),
    ]
    report = module.measure(rows, _scorers())

    grounded = report["judges"]["retrieval_groundedness"]
    # ref1 is an incorrect-refusal row: excluded from groundedness, like the
    # eval excludes rows with no expected docs.
    assert grounded["n"] == 4
    # s1, u1 agree; s2 disagrees (fake says fail on a supported row);
    # u2 disagrees (fake says grounded on an unsupported row).
    assert grounded["false_positives"] == 1  # u2: judge grounded, human not
    assert grounded["false_negatives"] == 1  # s2: judge ungrounded, human supported

    refusal = report["judges"]["refusal_appropriateness"]
    assert refusal["n"] == 5
    # s2: judge failed an appropriate answer — a false NEGATIVE (the judge
    # missed a good row). ref1: judge fail vs human inappropriate agrees.
    assert refusal["false_negatives"] == 1
    assert refusal["false_positives"] == 0

    correctness = report["judges"]["correctness"]
    assert correctness["status"].startswith("not measurable")
    assert report["n_errors"] == 0


@pytest.mark.unit
def test_groundedness_verdict_mapping(module) -> None:
    assert module._groundedness_verdict(None) is None
    assert module._groundedness_verdict(SimpleNamespace(value="yes")) is True
    assert module._groundedness_verdict(SimpleNamespace(value="no")) is False
    with pytest.raises(RuntimeError, match="unexpected groundedness verdict"):
        module._groundedness_verdict(SimpleNamespace(value=1.0))


@pytest.mark.unit
def test_judge_errors_are_counted_and_excluded_but_do_not_pair(
    module, no_trace
) -> None:
    class FlakyRefusal:
        name = "refusal_appropriateness"

        def run(self, *, inputs, outputs, expectations, trace):
            raise ConnectionError("429 from gateway")

    rows = [_row(f"s{i}", supported=True) for i in range(40)]
    # 40 errors / (40 refusal + 40 groundedness attempts) = 0.5 > 0.05 ceiling.
    with pytest.raises(RuntimeError, match="exceeds the .* ceiling"):
        module.measure(rows, [_FakeGroundedness(), FlakyRefusal()])

    # Under the ceiling: 1 error in 40 rows is reported, and the errored row
    # simply has no refusal pair.
    class OnceFlaky:
        name = "refusal_appropriateness"
        calls = 0

        def run(self, *, inputs, outputs, expectations, trace):
            self.__class__.calls += 1
            if self.__class__.calls == 1:
                raise ConnectionError("429")
            return SimpleNamespace(value=True)

    report = module.measure(rows, [_FakeGroundedness(), OnceFlaky()])
    assert report["n_errors"] == 1
    assert report["judges"]["refusal_appropriateness"]["n"] == 39


@pytest.mark.unit
def test_named_finds_scorers_by_name(module) -> None:
    scorers = _scorers()
    assert module._named(scorers, "retrieval_groundedness") is scorers[0]
    assert module._named(scorers, "refusal_appropriateness") is scorers[1]
    with pytest.raises(RuntimeError, match="not found"):
        module._named(scorers, "correctness")
