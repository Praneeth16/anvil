"""Programmatic check functions for ANVIL's ``programmatic`` scorers.

A *programmatic* scorer evaluates an agent's answer deterministically —
no LLM call — by invoking one of the functions below (or a user-supplied
one in this same file). Each check function has the signature::

    def check_answer(prediction: str, ground_truth: dict) -> float

and returns a score in ``[0.0, 1.0]``. The ``ground_truth`` dict is the
eval row's ``expectations`` projection (built by
:func:`anvil.eval.runner._build_dataset`), so it carries the golden-set
fields — ``must_include``, ``should_refuse``, ``reference_answer``,
``expected_facts``, ``must_not_include``, and friends.

This file follows the meta-harness pattern: the user can append their
own check functions here and reference them by name from
``harness/config.yaml`` via a scorer's ``check_function`` field. The
eval runner loads this module dynamically (importlib) so edits take
effect without touching the harness source.

The built-ins are intentionally dependency-free (only the stdlib
:mod:`json`): they must run inside the eval worker without extra
installs, and they must be unit-testable in isolation.
"""

from __future__ import annotations

import json
from typing import Any


def _clamp(score: float) -> float:
    """Clamp a raw score to the valid ``[0.0, 1.0]`` range.

    A misconfigured custom check that returns 1.2 or -0.1 should not
    poison the aggregate or trip mlflow's Feedback validation.
    """
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return float(score)


def exact_match(prediction: str, ground_truth: dict) -> float:
    """1.0 if ``prediction`` exactly matches ``reference_answer``, else 0.0.

    Comparison is whitespace-normalized (stripped) so a trailing newline
    from the agent does not flip a correct answer to a failure. Rows
    without a ``reference_answer`` score 0.0 — exact match is undefined
    without a reference, and silently passing would mask a misconfigured
    scorer.
    """
    reference = ground_truth.get("reference_answer")
    if reference is None:
        return 0.0
    return 1.0 if str(prediction).strip() == str(reference).strip() else 0.0


def must_include_check(prediction: str, ground_truth: dict) -> float:
    """Fraction of ``must_include`` items found in ``prediction``.

    ``must_include`` is the golden-set list of substrings the answer
    must contain (e.g. ``["$0.142", "kWh"]``). Falls back to
    ``expected_facts`` (the mlflow-facing alias) for robustness. With no
    required items the check vacuously passes (1.0) — there is nothing
    to miss.
    """
    items = ground_truth.get("must_include")
    if items is None:
        items = ground_truth.get("expected_facts")
    if not items:
        return 1.0
    pred = str(prediction)
    found = sum(1 for item in items if str(item) in pred)
    return _clamp(found / len(items))


def json_schema_validity(prediction: str, ground_truth: dict) -> float:
    """1.0 if ``prediction`` is valid JSON, 0.0 otherwise.

    If ``ground_truth`` carries a ``json_schema`` dict, a lightweight
    structural check is applied on top: the parsed object must be of the
    declared ``type`` (when specified) and contain every key listed in
    ``required``. Full JSON Schema validation is intentionally not pulled
    in — ``jsonschema`` is not a harness dependency, and for an
    extraction agent the structural check is the meaningful signal.
    """
    try:
        obj = json.loads(str(prediction))
    except (json.JSONDecodeError, TypeError):
        return 0.0

    schema = ground_truth.get("json_schema")
    if isinstance(schema, dict) and not _matches_schema(obj, schema):
        return 0.0
    return 1.0


_TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


def _matches_schema(obj: Any, schema: dict) -> bool:
    """Lightweight JSON-Schema-ish structural check (no external deps)."""
    expected_type = schema.get("type")
    if expected_type is not None:
        py_type = _TYPE_MAP.get(expected_type)
        if py_type is not None and not isinstance(obj, py_type):
            return False
    required = schema.get("required")
    if isinstance(required, list) and isinstance(obj, dict):
        for key in required:
            if key not in obj:
                return False
    return True


def field_exact_match(prediction: str, ground_truth: dict) -> float:
    """For JSON predictions: fraction of expected fields that match exactly.

    Expected fields are read from ``ground_truth["expected_fields"]``
    (a ``{field: value}`` dict). If absent, the check falls back to
    parsing ``reference_answer`` as JSON and using its top-level fields.
    With no expected fields at all the check passes iff the prediction
    is a JSON object (1.0) — there is nothing to compare. A prediction
    that is not valid JSON scores 0.0.
    """
    try:
        pred_obj = json.loads(str(prediction))
    except (json.JSONDecodeError, TypeError):
        return 0.0

    expected = ground_truth.get("expected_fields")
    if not isinstance(expected, dict):
        reference = ground_truth.get("reference_answer")
        if isinstance(reference, dict):
            expected = reference
        elif isinstance(reference, str):
            try:
                parsed = json.loads(reference)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if isinstance(parsed, dict):
                expected = parsed

    if not isinstance(expected, dict) or not expected:
        return 1.0 if isinstance(pred_obj, dict) else 0.0
    if not isinstance(pred_obj, dict):
        return 0.0

    matched = sum(1 for key, value in expected.items() if pred_obj.get(key) == value)
    return _clamp(matched / len(expected))


__all__ = [
    "exact_match",
    "must_include_check",
    "json_schema_validity",
    "field_exact_match",
]
