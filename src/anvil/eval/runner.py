"""End-to-end evaluation runner — driver of ``mlflow.genai.evaluate``.

Wraps :func:`mlflow.genai.evaluate` with the active scorers (3 by
default; Safety opt-in), the golden set sub-set per mode
(``quick``/``standard``/``full``), and an :class:`AnvilAgent`
constructed with ``source=SOURCE_EVAL``.

Parallel predict execution: ``mlflow.genai.evaluate`` already runs
``predict_fn`` per row in a ``ThreadPoolExecutor`` sized by the
``MLFLOW_GENAI_EVAL_MAX_WORKERS`` env var (default 10). The harness
wires ``eval.n_workers`` from ``harness/config.yaml`` into that env
var so the configured value actually controls concurrency — and keeps
passing ``predict_fn`` (not pre-computed ``outputs``) so mlflow builds
a per-row trace carrying the ``RETRIEVER`` span that
``RetrievalGroundedness`` scores against. :func:`_run_predictions_parallel`
is anvil's own tested thread-pool primitive for direct/pre-compute
paths that do not need traces.

Public surface:

* :func:`evaluate_branch` — driver function callable from
  ``scripts/evaluate.py`` or another module.
* :class:`EvalReport` — aggregate / per-judge / per-bucket / failures.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import logging
import os
import sys
import time
import warnings
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.types.responses import ResponsesAgentRequest
from openai import OpenAI

try:
    from mlflow.environment_variables import MLFLOW_ENABLE_ASYNC_TRACE_LOGGING

    _MLFLOW_ASYNC_TRACE_LOGGING_ENV = MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.name
except ImportError:
    # Compatibility with MLflow versions that predate the env-var constant.
    _MLFLOW_ASYNC_TRACE_LOGGING_ENV = "MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"

from anvil.agents.memory_system import MemorySystem
from anvil.data import load_golden_set, select_subset
from anvil.eval.cache import compute_scorer_fingerprint
from anvil.eval.outcome import (
    Attempt,
    CaseOutcome,
    CaseRecord,
    RunInterrupted,
)
from anvil.eval.scorers import build_scorers
from anvil.observability import SOURCE_EVAL, enable_runtime_tracing
from anvil.runtime.agent import AnvilAgent
from anvil.runtime.client import build_gateway_client
from anvil.runtime.loader import default_runtime_config_path, load_harness
from anvil.runtime.models import EvalConfig, ScorerConfig, SplitConfig
from anvil.tools.search_knowledge_base import make_kb_executor

logger = logging.getLogger(__name__)


@dataclass
class EvalReport:
    """Summary of one ``mlflow.genai.evaluate`` run."""

    aggregate: float
    per_judge: dict[str, float]
    per_bucket: dict[str, dict[str, float]]
    failures: list[dict[str, Any]]
    run_id: str
    experiment_id: str
    n_rows: int
    mode: str
    scorers: list[str]
    evaluated_at: str
    trace_ids: list[str] = field(default_factory=list)
    # Rows that were never assessed: the prediction raised, so mlflow left
    # ``outputs`` None and recorded an ``error_message``. These are EXCLUDED
    # from ``aggregate``, ``per_judge``, and ``per_bucket`` rather than scored
    # as the near-zero an absent answer earns. ``n_errors`` counts every
    # captured error, including any that could not be joined back to a row.
    # See ``docs/design/failure-vs-error.md``.
    n_errors: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    # Errors that could NOT be joined back to a result row, and whose scores
    # therefore could not be excluded. A report with any of these cannot honour
    # its own contract -- the infrastructure zero is still in the mean -- so it
    # is unjudgeable regardless of the error rate. See
    # :func:`anvil.eval.judgeability.unjudgeable_reason`.
    n_unattributed_errors: int = 0
    # Rows that ran but never made it into ``result_df`` because their trace was
    # not retrievable. NOT errors -- the prediction succeeded -- but not measured
    # either, and invisible to ``error_rate`` because they are absent from
    # ``n_rows``. Without counting them, losing six of eight rows this way leaves
    # error_rate 0.0 and shrinks the floor along with the sample, so no guard
    # fires. The judgeability floor is computed against ``n_attempted``.
    n_dropped_rows: int = 0
    # Always-available eval cost proxies. Token usage may be added when
    # supplied by MLflow traces; context characters and row count do not
    # require another service call.
    cost_metrics: dict[str, float] = field(default_factory=dict)
    # JSON fingerprint of the aggregate scorer configs (name, type,
    # weight, check_function) that produced this report's aggregate.
    # Carried into ``CachedBaseline`` so the frontier gate can detect a
    # weight/check_function change that invalidates a cross-run
    # comparison even when scorer names are unchanged. Empty when the
    # report is built by code that predates this field.
    scorer_fingerprint: str = ""

    @property
    def error_rate(self) -> float:
        """Errored cases as a fraction of all cases.

        The round guard reads this instead of the score: a round with a high
        error rate has not measured the agent at all, so comparing its
        aggregate to the frontier would record a degraded gateway as a bad
        mutation and revert good work.

        A run with no rows reads ``1.0`` when anything errored and ``0.0``
        otherwise. ``0.0`` for the errored case would be a fail-*open* sentinel
        in a guard: an unmeasurable run would pass every check. Nothing reaches
        this today (``_aggregate_report`` needs a DataFrame), but a guard's
        degenerate case should point at "refuse", not at "fine".

        Clamped to ``1.0`` because ``n_errors`` counts captured errors, which can
        exceed the rows that made it into ``result_df`` -- not hypothetically:
        a row can error AND then lose its trace, leaving it counted here and
        absent from ``n_rows``.

        This is for reporting. Guards should read :attr:`unmeasured_rate`, which
        also sees rows that vanished without erroring.
        """
        if not self.n_rows:
            return 1.0 if self.n_errors else 0.0
        return min(1.0, self.n_errors / self.n_rows)

    @property
    def unmeasured_rate(self) -> float:
        """Fraction of ATTEMPTED cases that produced no usable score.

        The number a guard should read, because it is the union of the two ways a
        case goes unscored: it errored, or it vanished from the frame. Neither is
        visible in :attr:`error_rate`, which divides errors by the *surviving*
        rows and so cannot see a dropped one at all.

        Without this, a 20-row run that loses 16 traces reports ``error_rate``
        0.0, four scorable rows, and a floor of 4 -- judgeable, and a four-row
        mean extends the frontier. An absolute floor alone is too blunt: it is
        satisfiable by any run big enough, which is every mode above ``quick``.
        """
        attempted = self.n_attempted
        if not attempted:
            return 1.0 if (self.n_errors or self.n_dropped_rows) else 0.0
        return min(1.0, (self.n_errors + self.n_dropped_rows) / attempted)

    @property
    def n_attempted(self) -> int:
        """Rows the eval actually tried, including any dropped for want of a trace.

        The denominator a sample-size floor has to use. ``n_rows`` counts what
        survived into the frame, so comparing a floor against it lets row loss
        lower the bar it was supposed to trip.
        """
        return self.n_rows + self.n_dropped_rows

    @property
    def n_scorable(self) -> int:
        """Cases that were actually assessed, i.e. that the scores rest on.

        Exact wherever it is consulted: an unattributed error makes the report
        unjudgeable outright, so on any path that gets as far as reading this,
        every error corresponds to one excluded row.
        """
        return max(0, self.n_rows - self.n_errors)


def partition_dataset(
    examples: list[dict],
    split: SplitConfig,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Partition examples into (train, dev, test) by hash of example_id.

    Uses a deterministic hash plus seed so membership is stable across runs
    and independent of input ordering.
    """
    train: list[dict] = []
    dev: list[dict] = []
    test: list[dict] = []
    train_cutoff = split.train_ratio
    dev_cutoff = train_cutoff + split.dev_ratio

    for example in examples:
        digest = hashlib.md5(  # noqa: S324 - deterministic partitioning, not security
            f"{split.seed}:{example['example_id']}".encode(), usedforsecurity=False
        ).hexdigest()
        fraction = int(digest, 16) / (2**128)
        if fraction < train_cutoff:
            train.append(example)
        elif fraction < dev_cutoff:
            dev.append(example)
        else:
            test.append(example)

    return train, dev, test


def _verify_no_overlap(train: list[dict], dev: list[dict], test: list[dict]) -> None:
    """Assert no example_id appears in multiple partitions."""
    train_ids = {example["example_id"] for example in train}
    dev_ids = {example["example_id"] for example in dev}
    test_ids = {example["example_id"] for example in test}
    overlap = (train_ids & dev_ids) | (train_ids & test_ids) | (dev_ids & test_ids)
    if overlap:
        raise RuntimeError(f"partition overlap detected: {overlap}")


def _select_mode_examples(
    examples: list[dict], *, cfg: EvalConfig, selected_mode: str
) -> list[dict]:
    """Select a mode's rows while enforcing configured partition boundaries."""
    mode_config = cfg.modes[selected_mode]
    if not cfg.split.enabled:
        return select_subset(examples, buckets=mode_config.buckets)

    train, dev, test = partition_dataset(examples, cfg.split)
    _verify_no_overlap(train, dev, test)
    if selected_mode == "test":
        return test[: mode_config.rows]

    scaled_buckets = {
        bucket: max(1, round(count * cfg.split.dev_ratio))
        for bucket, count in mode_config.buckets.items()
    }
    if scaled_buckets != mode_config.buckets:
        warnings.warn(
            f"scaled {selected_mode!r} bucket counts for dev_ratio="
            f"{cfg.split.dev_ratio}: {mode_config.buckets} -> {scaled_buckets}",
            UserWarning,
            stacklevel=2,
        )
    return select_subset(dev, buckets=scaled_buckets)


def _extract_final_text(response: Any) -> str:
    """Walk ``response.output`` for the last ``message`` and concat its content."""
    output = getattr(response, "output", None) or []
    last_message: dict[str, Any] | None = None
    for item in output:
        data = item.model_dump() if hasattr(item, "model_dump") else item
        if isinstance(data, dict) and data.get("type") == "message":
            last_message = data
    if last_message is None:
        return ""
    parts = last_message.get("content", [])
    if isinstance(parts, str):
        return parts
    if not isinstance(parts, list):
        return ""
    chunks: list[str] = []
    for part in parts:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
        elif isinstance(part, str):
            chunks.append(part)
    return "".join(chunks)


def _build_dataset(examples: list[dict]) -> list[dict]:
    """Project golden-set rows into mlflow's inputs/expectations/tags shape."""
    # Correctness rejects rows that pass BOTH expected_response and
    # expected_facts. We use must_include as expected_facts; the
    # reference_answer stays in the row for human debugging via
    # mlflow.search_traces.
    #
    # ``must_include`` is ALSO projected under its golden-set name so
    # programmatic check functions (data/evaluator.py) can read the
    # familiar key directly from the expectations dict they receive as
    # ``ground_truth``. This is additive — Correctness still reads
    # ``expected_facts`` and ignores the alias.
    rows: list[dict] = []
    for ex in examples:
        expectations: dict[str, Any] = {
            "expected_facts": ex["must_include"],
            "must_include": ex["must_include"],
            "should_refuse": ex["should_refuse"],
            "expected_doc_ids": ex["expected_doc_ids"],
            "expected_citations": ex["expected_citations"],
            "must_not_include": ex["must_not_include"],
            "notes_for_judge": ex["notes_for_judge"],
            "reference_answer": ex["reference_answer"],
        }
        # Pass through json_schema, expected_fields, and any other
        # extension fields prefixed with ``json_`` or ``expected_`` so
        # programmatic check functions (json_schema_validity,
        # field_exact_match) receive their documented primary inputs
        # through the real runner. This is additive — existing scorers
        # ignore unknown keys in the expectations dict.
        for key, val in ex.items():
            if key not in expectations and (key.startswith("json_") or key.startswith("expected_")):
                expectations[key] = val
        rows.append(
            {
                "inputs": {
                    "query": ex["query"],
                    "category": ex["category"],
                },
                "expectations": expectations,
                "tags": {"example_id": ex["example_id"]},
            }
        )
    return rows


def _coerce_score(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("yes", "pass", "true", "ok"):
            return 1.0
        if s in ("no", "fail", "false"):
            return 0.0
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _row_score(row: Any, scorer_name: str) -> float | None:
    if hasattr(row, "get"):
        flat = row.get(f"{scorer_name}/value")
        coerced = _coerce_score(flat)
        if coerced is not None:
            return coerced
    assessments = row.get("assessments") if hasattr(row, "get") else None
    if not isinstance(assessments, list):
        return None
    for a in assessments:
        if not isinstance(a, dict):
            continue
        if a.get("assessment_name") != scorer_name:
            continue
        feedback = a.get("feedback")
        if isinstance(feedback, dict):
            coerced = _coerce_score(feedback.get("value"))
            if coerced is not None:
                return coerced
    return None


def _row_trace_id(row: Any) -> str:
    """The row's ``trace_id`` as a string, or ``""`` when absent."""
    if not hasattr(row, "get"):
        return ""
    trace_id = row.get("trace_id")
    return str(trace_id) if trace_id else ""


def _category_for_row(row: Any, examples: list[dict], idx: int) -> str:
    if hasattr(row, "get"):
        request = row.get("request")
        if isinstance(request, dict):
            cat = request.get("category")
            if isinstance(cat, str):
                return cat
    if idx < len(examples):
        cat = examples[idx].get("category")
        if isinstance(cat, str):
            return cat
    return ""


def _aggregate_report(
    *,
    result_df,
    metrics: dict[str, float],
    scorer_names: list[str],
    aggregate_scorer_names: list[str],
    weights: dict[str, float],
    examples: list[dict],
    run_id: str,
    experiment_id: str,
    mode: str,
    scorer_fingerprint: str = "",
    errored: dict[str, str] | None = None,
    n_dropped_rows: int = 0,
    attempted_examples: list[dict] | None = None,
) -> EvalReport:
    n_rows = len(result_df)

    # Rows whose prediction raised. ``errored`` maps trace_id -> message, as
    # captured from mlflow's own ``eval_item.error_message`` by
    # ``_resilient_eval_harness``. trace_id is the join key because it is the
    # only row identifier that appears in both places.
    errored = errored or {}
    row_trace_ids = [_row_trace_id(result_df.iloc[i]) for i in range(n_rows)]
    errored_rows = {i for i, tid in enumerate(row_trace_ids) if tid and tid in errored}
    unattributed = sorted(set(errored) - {tid for tid in row_trace_ids if tid})
    if unattributed:
        # The row is not in result_df (or carries no trace_id), so its score
        # cannot be excluded -- there is nothing to exclude. Count it anyway so
        # the round guard still sees the degradation, and say so out loud
        # rather than dropping it because it would not join.
        logger.warning(
            "%s eval error(s) could not be attributed to a result row and so "
            "cannot be excluded from the scores: %s",
            len(unattributed),
            ", ".join(unattributed),
        )

    per_judge_rows: dict[str, list[float | None]] = {
        name: [
            None if i in errored_rows else _row_score(result_df.iloc[i], name)
            for i in range(n_rows)
        ]
        for name in scorer_names
    }

    def _mean(values: list[float | None]) -> float:
        nums = [v for v in values if v is not None]
        return sum(nums) / len(nums) if nums else 0.0

    per_judge: dict[str, float] = {}
    for name in scorer_names:
        metric_key = f"{name}/mean"
        # mlflow's own mean is computed over every row, errored ones included,
        # so the moment anything errored it is exactly the number this
        # exclusion exists to stop trusting. Fall back to anvil's mean, which
        # sees the Nones written above.
        if metric_key in metrics and not errored:
            per_judge[name] = float(metrics[metric_key])
            continue
        contributing = [v for v in per_judge_rows[name] if v is not None]
        if not contributing and metric_key in metrics:
            # Switching away from mlflow's mean must not silently turn a real
            # score into 0.0. That happens when a scorer's per-row
            # ``{name}/value`` is absent from result_df while its mean is in
            # ``metrics``: ``_mean`` has nothing to average and returns 0.0,
            # which would drag the aggregate down and revert a good mutation
            # for exactly the reason this exclusion exists to prevent. Say so
            # rather than emitting the zero quietly.
            logger.warning(
                "scorer %r has no per-row scores in result_df, so excluding "
                "errored rows leaves nothing to average; its mean reads 0.0 "
                "instead of mlflow's %.4f. The aggregate is understated.",
                name,
                float(metrics[metric_key]),
            )
        per_judge[name] = _mean(per_judge_rows[name])

    # Weighted average across the configured scorers. ``weights`` maps a
    # scorer name to its config weight (defaulting to 1.0); with uniform
    # weights this collapses to the legacy unweighted mean, so a shipped
    # scaffold that lists scorers as bare strings scores identically.
    total_weight = sum(weights.get(name, 1.0) for name in aggregate_scorer_names)
    if aggregate_scorer_names and total_weight > 0:
        aggregate = (
            sum(per_judge[name] * weights.get(name, 1.0) for name in aggregate_scorer_names)
            / total_weight
        )
    else:
        aggregate = 0.0

    bucket_rows: dict[str, list[int]] = defaultdict(list)
    for i in range(n_rows):
        if i in errored_rows:
            # A bucket whose only rows errored disappears rather than reading
            # 0.0. The per-bucket means steer where the next mutation aims, and
            # 0.0 would assert the agent is weak at a category the round has no
            # evidence about.
            continue
        category = _category_for_row(result_df.iloc[i], examples, i)
        if category:
            bucket_rows[category].append(i)
    per_bucket: dict[str, dict[str, float]] = {}
    for bucket, idxs in bucket_rows.items():
        per_bucket[bucket] = {
            name: _mean([per_judge_rows[name][i] for i in idxs]) for name in scorer_names
        }

    failures: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    trace_ids: list[str] = []
    for i in range(n_rows):
        row = result_df.iloc[i]
        trace_id = row.get("trace_id") if hasattr(row, "get") else None
        if trace_id:
            trace_ids.append(str(trace_id))
        example_id = examples[i]["example_id"] if i < len(examples) else ""
        query = examples[i]["query"] if i < len(examples) else ""
        if i in errored_rows:
            # Reported as an error, never as a failure. A failure list that
            # includes never-assessed rows sends the optimizer chasing a bad
            # answer that was never given.
            errors.append(
                {
                    "example_id": example_id,
                    "query": query,
                    "category": _category_for_row(row, examples, i),
                    "trace_id": trace_id,
                    "error_message": errored[str(trace_id)],
                }
            )
            continue
        judge_failures = [
            name for name in scorer_names if (s := per_judge_rows[name][i]) is not None and s < 1.0
        ]
        if not judge_failures:
            continue
        failures.append(
            {
                "example_id": example_id,
                "query": query,
                "category": _category_for_row(row, examples, i),
                "judge_failures": judge_failures,
                "trace_id": trace_id,
            }
        )
    for trace_id_str in unattributed:
        errors.append(
            {
                "example_id": "",
                "query": "",
                "category": "",
                "trace_id": trace_id_str,
                "error_message": errored[trace_id_str],
            }
        )

    return EvalReport(
        aggregate=aggregate,
        per_judge=per_judge,
        per_bucket=per_bucket,
        failures=failures,
        run_id=run_id,
        experiment_id=experiment_id,
        n_rows=n_rows,
        mode=mode,
        scorers=list(scorer_names),
        evaluated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        trace_ids=trace_ids,
        n_errors=len(errored),
        errors=errors,
        n_unattributed_errors=len(unattributed),
        n_dropped_rows=n_dropped_rows,
        # Cost is what the run SPENT, so it is measured over the rows it
        # attempted. Measuring it over the survivors would make losing rows look
        # like a cost improvement to a minimising Pareto objective on ``n_rows``
        # or ``context_chars`` -- row loss registering as a win.
        cost_metrics={
            "total_context_chars": float(
                sum(len(str(ex.get("query", ""))) for ex in (attempted_examples or examples))
            ),
            "n_rows": float(n_rows + n_dropped_rows),
        },
        scorer_fingerprint=scorer_fingerprint,
    )


# ---------------------------------------------------------------------------
# Code-mode agent loading
# ---------------------------------------------------------------------------


def _import_agent_module(module_path: str) -> ModuleType:
    """Import an agent module from a dotted path or a ``.py`` file path.

    A dotted path (e.g. ``anvil.agents.baseline``) is resolved via
    :func:`importlib.import_module`. A path containing a separator or
    ending in ``.py`` is loaded from disk via ``spec_from_file_location``
    — this is how FORGE loads candidate modules the optimizer just wrote
    to ``agents/`` that are not yet installed packages.
    """
    if module_path.endswith(".py") or "/" in module_path or os.sep in module_path:
        path = Path(module_path)
        if not path.is_file():
            raise FileNotFoundError(f"agent module not found: {path}")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create import spec for agent module: {path}")
        module = importlib.util.module_from_spec(spec)
        # Register before exec so @dataclass, __init_subclass__, and runtime
        # type-resolution mechanisms that look up the module in sys.modules
        # work during import. Mirrors importlib.import_module's contract.
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            # Remove the broken module so a retry doesn't find a partially-
            # initialized entry.
            sys.modules.pop(spec.name, None)
            raise
        return module
    return importlib.import_module(module_path)


def _find_memory_system_subclass(module: ModuleType) -> type[MemorySystem]:
    """Find the concrete ``MemorySystem`` subclass defined in ``module``.

    The class must be *defined* in this module (``__module__`` match) so
    that a re-exported base class or an imported helper does not get
    mistaken for the agent. Exactly one subclass is expected; zero or
    multiple are configuration errors.
    """
    candidates: list[type[MemorySystem]] = []
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, MemorySystem)
            and obj is not MemorySystem
            and getattr(obj, "__module__", None) == module.__name__
            and not inspect.isabstract(obj)
        ):
            candidates.append(obj)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            f"no concrete MemorySystem subclass found in agent module {module.__name__!r}"
        )
    raise ValueError(
        f"multiple concrete MemorySystem subclasses found in {module.__name__!r}: "
        f"{[c.__name__ for c in candidates]}"
    )


def _load_memory_system(
    module_path: str,
    *,
    llm_client: OpenAI | None = None,
    model: str = "",
) -> MemorySystem:
    """Import an agent module and instantiate its ``MemorySystem`` subclass.

    ``module_path`` is either a dotted Python module path (e.g.
    ``anvil.agents.baseline``) or a ``.py`` file path. The module must
    define exactly one concrete ``MemorySystem`` subclass, which is
    instantiated with ``llm_client`` and ``model`` as constructor kwargs.
    """
    module = _import_agent_module(module_path)
    cls = _find_memory_system_subclass(module)
    return cls(llm_client=llm_client, model=model)


# mlflow reads this env var to size the predict/score thread pools inside
# ``mlflow.genai.evaluate`` (default 10 when unset). anvil wires
# ``eval.n_workers`` into it so the configured value controls concurrency
# rather than mlflow's default.
_MLFLOW_MAX_WORKERS_ENV = "MLFLOW_GENAI_EVAL_MAX_WORKERS"

# mlflow runs ``check_model_prediction`` before the eval, invoking predict_fn once
# under ``@trace_disabled``. Two reasons anvil skips it, both observed live:
#
# 1. It converts a prediction failure into an opaque crash. The exception escapes
#    predict_fn into mlflow's own pyfunc validation wrapper -- a generator-based
#    context manager that mishandles ``throw()`` -- and emerges as
#    ``RuntimeError: generator didn't stop after throw()``. A wrong endpoint name
#    in harness/config.yaml surfaced as exactly that, with the underlying 404
#    replaced. mlflow then aborts the whole run, so the row never becomes an
#    error record and none of the Phase 2 guards engage: no error_rate, no
#    exclusion, no legible refusal. Skipping the check lets the failure happen on
#    the first real row instead, where it IS captured as evidence.
# 2. Its other job -- auto-wrapping an untraced predict_fn -- is redundant here.
#    ``evaluate_branch`` opens its own root CHAIN span per row, which is the
#    stronger guarantee because it also nests the RETRIEVER span.
#
# A signature mismatch, the check's remaining value, would fail on row 1 and be
# reported as an error rather than silently passing.
_MLFLOW_SKIP_VALIDATION_ENV = "MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"

# Name of the per-row root span ``evaluate_branch`` wraps every
# ``predict_fn`` invocation in. The span yields a real per-row trace
# carrying the ``RETRIEVER`` span that ``RetrievalGroundedness`` scores;
# ``_resilient_eval_harness`` (PR #21) is the safety net that prevents a
# crash if a row's trace is still None. See ``evaluate_branch``.
_PREDICT_SPAN_NAME = "anvil.predict"

# How hard to retry the minimal-trace fallback before giving up on a row. The
# fallback itself calls ``mlflow.get_trace``, which is the unreliable step, so a
# single attempt is not a fallback at all on the live tracing server.
_TRACE_FALLBACK_ATTEMPTS = 3
_TRACE_FALLBACK_BACKOFF_S = 0.25


def _traced_predict(inputs: dict[str, Any], body: Callable[[], str]) -> str:
    """Run ``body`` inside the per-row root span, re-raising *after* it closes.

    The span is what guarantees a per-row trace (see the ``predict_fn``s in
    :func:`evaluate_branch`). The subtlety is what happens when ``body`` fails.

    Raising inside the ``with`` throws the exception *into*
    ``mlflow.start_span``'s generator-based context manager. Live, that manager
    is sometimes a no-op -- mlflow's own pre-flight ``check_model_prediction``
    invokes ``predict_fn`` under ``@trace_disabled`` -- and throwing into it
    produces ``RuntimeError: generator didn't stop after throw()``, which
    *replaces* the real exception. Observed live: one wrong endpoint name in
    ``harness/config.yaml`` surfaced as that RuntimeError and nothing else, with
    the underlying 404 gone.

    That is a Failure-vs-Error problem, not just a confusing message. mlflow
    records ``eval_item.error_message`` from whatever escapes ``predict_fn``, and
    :func:`_resilient_eval_harness` captures it so the row can be excluded and
    the round guarded. If what escapes is a generator-plumbing RuntimeError, the
    evidence records that instead of the endpoint failure, and an operator
    debugging a degraded round is sent to the wrong place.

    So the exception is caught, noted on the span, and re-raised once the span
    has closed normally. ``predict_fn`` still raises, so nothing downstream
    changes -- except that what it raises is the real error.
    """
    error: BaseException | None = None
    text = ""
    with mlflow.start_span(name=_PREDICT_SPAN_NAME, span_type=SpanType.CHAIN) as span:
        span.set_inputs(inputs)
        try:
            text = body()
        except BaseException as exc:  # noqa: BLE001 - re-raised below, unchanged
            error = exc
            # Record the failure ON the span. Catching inside the ``with`` means
            # ``mlflow.start_span`` never sees the exception, so it cannot run its
            # own ``record_exception`` -- without this the span would close with
            # status OK and a failed row's trace would read as a success in the
            # MLflow UI, which is the opposite of the legibility this function
            # exists for. Best-effort: a no-op span (trace_disabled) may accept
            # none of these, and losing the annotation must not lose the error.
            with suppress(Exception):
                span.record_exception(exc)
            with suppress(Exception):
                span.set_status("ERROR")
            with suppress(Exception):
                span.set_attribute("anvil.predict_error", f"{type(exc).__name__}: {exc}")
        else:
            span.set_outputs({"response": text})
    if error is not None:
        raise error
    return text


@contextmanager
def _synchronous_trace_logging():
    """Temporarily force MLflow trace export to complete synchronously."""
    previous = os.environ.get(_MLFLOW_ASYNC_TRACE_LOGGING_ENV)
    os.environ[_MLFLOW_ASYNC_TRACE_LOGGING_ENV] = "false"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_MLFLOW_ASYNC_TRACE_LOGGING_ENV, None)
        else:
            os.environ[_MLFLOW_ASYNC_TRACE_LOGGING_ENV] = previous


@contextmanager
def _resilient_eval_harness(
    *,
    error_sink: dict[str, str] | None = None,
    dropped_sink: set[str] | None = None,
    kept_rows_sink: list[int] | None = None,
):
    """Scope a defensive shim around ``mlflow.genai.evaluate``'s harness.

    Also the interception point for per-row prediction errors. When
    ``error_sink`` is given it is filled with ``trace_id -> error_message`` for
    every row whose ``predict_fn`` raised. mlflow already records that message
    (``_run_predict``'s ``except`` sets ``eval_item.error_message`` and leaves
    ``outputs`` None) and then reads it nowhere: the row proceeds to scoring
    and the judges score the absence of an answer as a wrong answer, at or near
    0.0. Capturing it here is what lets ``_aggregate_report`` exclude the row
    instead — a failure is a fact about the agent, an error is a fact about the
    infrastructure, and only the first belongs in the score. This costs no new
    interception point because the shim already wraps ``_run_predict``.

    The sink is keyed by ``trace_id`` rather than mlflow's internal
    ``request_id`` because ``trace_id`` is the only row identifier that also
    appears in ``result_df``, which is what the aggregate reads. The key is
    read *after* the minimal-trace fallback below, so an errored row -- which
    creates no span and therefore has no trace of its own -- still has one.

    Workaround for a known mlflow 3.11.x bug (verified against 3.11.1, the
    newest in-range release on the internal proxy — no patch bump is
    available to fix this). When ``predict_fn`` is supplied, the harness
    retrieves each row's trace via ``mlflow.get_trace(request_id,
    silent=True)`` (``harness._run_predict``, ~line 782). On the Databricks
    Tracing Server that trace is sometimes not retrievable at scoring time
    — even with synchronous export forced from process start (see
    ``anvil/__init__.py``) and PR #16's root span — leaving
    ``eval_item.trace`` None for some rows. The harness then dereferences it
    without a None check and aborts the whole run:

    * ``_get_new_expectations`` (``harness.py``:934-942) reads
      ``eval_item.trace.info.assessments`` and raises
      ``AttributeError: 'NoneType' object has no attribute 'info'`` — the
      live ``make_baseline`` crash, typically ~row 2-3 of 8.

    This context manager monkeypatches two harness symbols, scoped to the
    ``mlflow.genai.evaluate`` call (restored on exit — NOT a global
    import-time patch), so a missing per-row trace never crashes the run:

    1. ``_get_new_expectations`` → a None-safe wrapper that yields ``[]``
       (no trace-derived expectations) for a None-trace row instead of
       raising, and delegates to the original implementation otherwise.
       This directly neutralizes the confirmed crash site. Rows WITH a
       trace are scored normally — ``RetrievalGroundedness`` and the other
       scorers are NOT globally disabled; a None-trace row simply
       contributes no expectations and its scorers run as-is (scorer
       exceptions are already caught by the harness at ``run_scorer``:874
       and recorded as error feedbacks, never aborting the run).

    2. ``_run_predict`` → a wrapper that, after the original runs, falls
       back to ``create_minimal_trace(eval_item)`` when
       ``mlflow.get_trace(request_id)`` returned None. This is the SAME
       fallback the static-dataset path uses (``harness.py``:795) but the
       ``predict_fn`` path omits. ``create_minimal_trace`` fetches the
       trace by its own just-created ``trace_id`` under
       ``is_evaluate=True`` (synchronous export) — the reliable retrieval
       mechanism, not the failing request_id lookup. This ensures every
       row carries a trace so the eval COMPLETES with a real result
       DataFrame, instead of merely moving the crash one step downstream
       into ``batch_link_traces_to_run`` (``trace_utils.py``:1014, an
       unguarded ``eval_item.trace.info.trace_id`` list-comprehension) or
       ``construct_eval_result_df`` (``trace_utils.py``:925, caught but
       yields a None DataFrame that breaks ``_aggregate_report``).

    The shim (1) is the direct guard against the confirmed crash; the
    fallback (2) is the root-cause fix that prevents the crash from
    relocating. Together they bring the ``predict_fn`` path to the same
    per-row-trace reliability the production static-dataset path already
    relies on.
    """
    import mlflow.genai.evaluation.harness as _harness
    from mlflow.genai.utils.trace_utils import create_minimal_trace

    _orig_get_new_expectations = _harness._get_new_expectations
    _orig_run_predict = _harness._run_predict
    _orig_batch_link = _harness.batch_link_traces_to_run
    _orig_construct_df = _harness.construct_eval_result_df

    def _get_new_expectations_none_safe(eval_item):
        # mlflow 3.11.x harness.py:936 derefs ``eval_item.trace.info.assessments``
        # without a None check. A row whose trace the Databricks backend did not
        # return leaves ``eval_item.trace`` None and crashes here. Yield no
        # expectations for that row instead of raising; rows with a trace are
        # scored normally via the original implementation.
        if eval_item.trace is None:
            return []
        return _orig_get_new_expectations(eval_item)

    def _run_predict_with_minimal_trace_fallback(
        eval_item, predict_fn, run_id, rate_limiter, max_retries=0, experiment_id=None
    ):
        _orig_run_predict(
            eval_item, predict_fn, run_id, rate_limiter, max_retries, experiment_id
        )
        # harness.py:782 sets ``eval_item.trace = mlflow.get_trace(request_id)``.
        # On the Databricks backend that returns None for some rows. The
        # static-dataset path (harness.py:795) falls back to a minimal trace;
        # the predict_fn path does not, so apply the same fallback here. This
        # fetches by the just-created trace_id (reliable, sync), not request_id.
        if predict_fn is not None and eval_item.trace is None:
            # ``create_minimal_trace`` ends in ``mlflow.get_trace(...)``, i.e. it
            # depends on the very retrieval this fallback exists to work around.
            # Against a local file store it always succeeds, so every offline
            # test passes; against the Databricks Tracing Server it can return
            # None, and then nothing has been fixed. Retry a couple of times --
            # export is synchronous here, but the server can still lag the write.
            for attempt in range(_TRACE_FALLBACK_ATTEMPTS):
                # create_minimal_trace opens a span and calls get_trace; either can
                # raise on a transport error. An escape here propagates through
                # mlflow's ``future.result()`` and aborts the whole run after every
                # prediction and judge call is paid for -- the exact failure this
                # fallback exists to prevent. Retrying made that three chances
                # instead of one, so a raise is treated exactly like a None.
                try:
                    eval_item.trace = create_minimal_trace(eval_item)
                except Exception as exc:  # noqa: BLE001 - a raise is just a miss
                    logger.warning(
                        "minimal-trace fallback attempt %s for row %s raised %s: %s",
                        attempt + 1,
                        eval_item.request_id,
                        type(exc).__name__,
                        exc,
                    )
                    eval_item.trace = None
                if eval_item.trace is not None:
                    break
                if attempt + 1 < _TRACE_FALLBACK_ATTEMPTS:
                    time.sleep(_TRACE_FALLBACK_BACKOFF_S * (attempt + 1))
            if eval_item.trace is None:
                # Counted, not just logged. A dropped row shrinks the sample the
                # aggregate rests on without being an *error*, so nothing else
                # would notice -- the same shape of silent sample loss that
                # excluding errored rows introduced. See EvalReport.n_dropped_rows.
                if dropped_sink is not None:
                    dropped_sink.add(str(eval_item.request_id))
                logger.warning(
                    "row %s has no retrievable trace after %s fallback attempts; "
                    "it will be dropped from the result frame rather than crash "
                    "the run",
                    eval_item.request_id,
                    _TRACE_FALLBACK_ATTEMPTS,
                )
        # The row never produced an answer. Record it so the aggregate can
        # exclude it rather than scoring the absence as a wrong answer.
        if error_sink is not None and eval_item.error_message:
            trace = eval_item.trace
            key = str(trace.info.trace_id) if trace is not None else str(eval_item.request_id)
            # One assignment per row from mlflow's predict thread pool; dict
            # __setitem__ is atomic, so no lock is needed.
            error_sink[key] = str(eval_item.error_message)

    def _batch_link_traces_to_run_none_safe(run_id, eval_results):
        # trace_utils.py:1014 is an unguarded
        # ``[r.eval_item.trace.info.trace_id for r in eval_results]``. A row whose
        # trace is still None after the fallback above kills the whole run HERE,
        # after every prediction and every judge call has already been paid for.
        # Observed live on 2026-08-22: 8/8 rows predicted and scored, then the
        # run died at this line. Link the rows that have a trace and drop the
        # rest -- a run that loses one row from the frame is worth far more than
        # a run that loses all of them.
        linkable = [r for r in eval_results if getattr(r.eval_item, "trace", None) is not None]
        dropped = len(eval_results) - len(linkable)
        if dropped:
            logger.warning(
                "%s row(s) had no trace to link and were dropped from the result "
                "frame; the aggregate is computed over the rest",
                dropped,
            )
        if not linkable:
            return None
        return _orig_batch_link(run_id=run_id, eval_results=linkable)

    def _construct_eval_result_df_none_safe(run_id, traces, eval_results):
        # trace_utils.py:925 derefs ``eval_result.eval_item.trace.info.trace_id``
        # inside a try/except that swallows the AttributeError and returns None.
        # A None DataFrame then reaches ``_aggregate_report`` as
        # ``len(None)`` -> TypeError, so guarding the two sites above merely
        # moved the crash here. Observed live on 2026-08-22, after the
        # batch-link guard: "Evaluation completed", run logged, then TypeError.
        kept = [
            i
            for i, r in enumerate(eval_results)
            if getattr(r.eval_item, "trace", None) is not None
        ]
        # Which rows survived, not just how many. ``_aggregate_report`` joins the
        # frame to ``examples`` BY POSITION, so filtering rows out silently shifts
        # every example_id and query by the number dropped before it -- a failure
        # would be reported against the previous golden case, and the optimizer
        # reads those failures to choose its next mutation. Recording the indices
        # lets the caller filter ``examples`` the same way.
        if kept_rows_sink is not None:
            kept_rows_sink.clear()
            kept_rows_sink.extend(kept)
        return _orig_construct_df(run_id, traces, [eval_results[i] for i in kept])

    _harness._get_new_expectations = _get_new_expectations_none_safe
    _harness._run_predict = _run_predict_with_minimal_trace_fallback
    _harness.batch_link_traces_to_run = _batch_link_traces_to_run_none_safe
    _harness.construct_eval_result_df = _construct_eval_result_df_none_safe
    try:
        yield
    finally:
        _harness._get_new_expectations = _orig_get_new_expectations
        _harness._run_predict = _orig_run_predict
        _harness.batch_link_traces_to_run = _orig_batch_link
        _harness.construct_eval_result_df = _orig_construct_df


def _predict_one(
    predict_fn: Callable[[str], str],
    query: str,
    *,
    case_id: str,
    row: int,
    max_retries: int,
) -> CaseRecord:
    """Predict one case, retrying **errors only**, and record the outcome.

    Retrying an error is buying another sample of the infrastructure.
    Retrying a *failure* would be buying another sample of the agent, which
    is a different thing entirely: it turns the score into a function of how
    many attempts were paid for, and a self-optimizing loop that can spend
    its way to a better number will. So a returned answer is final however
    bad it is -- including an empty one -- and only a raised exception is
    tried again.
    """
    attempts: list[Attempt] = []
    started = time.monotonic()
    for attempt_no in range(max_retries + 1):
        attempt_started = time.monotonic()
        try:
            output = predict_fn(query)
        except Exception as exc:  # noqa: BLE001 — isolate per-row failures
            attempts.append(
                Attempt(
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    duration_ms=_elapsed_ms(attempt_started),
                )
            )
            if attempt_no < max_retries:
                logger.warning(
                    "prediction failed for row %s (attempt %s/%s), retrying: %s",
                    row,
                    attempt_no + 1,
                    max_retries + 1,
                    exc,
                )
                continue
            logger.warning("prediction failed for row %s: %s", row, exc)
            return CaseRecord.errored(
                case_id,
                exc,
                attempts=tuple(attempts),
                duration_ms=_elapsed_ms(started),
            )
        return CaseRecord(
            case_id=case_id,
            outcome=CaseOutcome.OK,
            output=output,
            attempts=tuple(attempts),
            duration_ms=_elapsed_ms(started),
        )
    raise AssertionError("unreachable: the loop returns on success and on exhaustion")


def _elapsed_ms(since: float) -> int:
    return int((time.monotonic() - since) * 1000)


def _drain_after_interrupt(
    executor: ThreadPoolExecutor,
    future_to_idx: dict[Future[CaseRecord], int],
    slots: list[CaseRecord | None],
    ids: list[str],
) -> list[CaseRecord]:
    """After a Ctrl-C: stop new work, keep what finished, account for the rest.

    ``cancel()`` succeeds only for rows that never started, which is exactly the
    set that should be marked interrupted. The rows already running are waited
    for, because throwing away a completed answer to exit a few seconds sooner
    is the wrong trade.

    But that wait is unbounded -- there is no per-case timeout (see
    :func:`_run_predictions_parallel`) and the HTTP client's default deadline is
    600s -- so an operator could face ten minutes of apparent hang with no
    output. Hence two things: the wait is announced, and a *second* Ctrl-C
    abandons it and still returns the records. Otherwise the natural response to
    the hang would raise out of ``shutdown`` and discard everything collected,
    which is precisely the guarantee :class:`RunInterrupted` makes.

    The abandoned threads keep running until their own sockets time out; Python
    cannot cancel them. They no longer hold up the *records*, which is what this
    function is responsible for.
    """
    n_cancelled = sum(1 for future in future_to_idx if future.cancel())
    in_flight = sum(1 for future in future_to_idx if not future.done())
    print(
        f"\nInterrupted. {n_cancelled} case(s) not started; waiting for {in_flight} "
        "in flight (Ctrl-C again to abandon them and keep what finished).",
        file=sys.stderr,
    )
    try:
        executor.shutdown(wait=True)
    except KeyboardInterrupt:
        print(
            f"Abandoning {in_flight} in-flight case(s); their threads run on until "
            "the HTTP client times out.",
            file=sys.stderr,
        )

    for future, idx in future_to_idx.items():
        if slots[idx] is not None:
            continue
        if not future.done() or future.cancelled():
            slots[idx] = CaseRecord(case_id=ids[idx], outcome=CaseOutcome.INTERRUPTED)
            continue
        try:
            slots[idx] = future.result()
        except KeyboardInterrupt:
            # The row's own predict_fn was the thing that raised.
            slots[idx] = CaseRecord(case_id=ids[idx], outcome=CaseOutcome.INTERRUPTED)
    return [
        r or CaseRecord(case_id=ids[i], outcome=CaseOutcome.INTERRUPTED)
        for i, r in enumerate(slots)
    ]


def _run_predictions_parallel(
    predict_fn: Callable[[str], str],
    queries: list[str],
    n_workers: int = 1,
    *,
    case_ids: list[str] | None = None,
    max_retries: int = 0,
) -> list[CaseRecord]:
    """Run ``predict_fn`` across ``queries`` and return one record per row.

    Uses :class:`concurrent.futures.ThreadPoolExecutor` — the runtime
    agent's work is I/O-bound (LLM / tool HTTP calls), so threads are
    sufficient and avoid the serialization overhead of processes. When
    ``n_workers <= 1`` the function runs sequentially (backward compatible
    with the pre-parallel eval path).

    Records preserve input order regardless of completion order: each
    future is keyed by its input index, so the slot it writes is fixed. A
    prediction that raises is recorded as a
    :attr:`~anvil.eval.outcome.CaseOutcome.ERROR` record and logged, so one
    bad row does not abort the whole eval — mirroring mlflow's own per-row
    error isolation in ``_run_predict``.

    This used to record a raised prediction as ``""``. That was the defect
    :mod:`anvil.eval.outcome` exists to correct: an empty string is not a
    neutral value but a very bad answer, so the judges scored it near 0.0
    and a throttled gateway moved the promotion gate exactly as a bad
    mutation would. An error record is excluded from the aggregate instead.

    ``Ctrl-C`` raises :class:`~anvil.eval.outcome.RunInterrupted` carrying a
    record for every row — the rows already done keep their results, the
    rest are :attr:`~anvil.eval.outcome.CaseOutcome.INTERRUPTED` — so a
    killed run is readable rather than lost.

    Thread-safety: ``predict_fn`` must be safe to invoke from multiple
    threads concurrently. For prompt mode ``AnvilAgent.predict`` issues
    stateless HTTP calls against the runtime endpoint (thread-safe); for
    code mode a ``MemorySystem.predict`` subclass is thread-safe as long
    as it does not mutate shared state inside ``predict``.

    Note:
        There is deliberately **no per-case timeout here**. A thread cannot
        be cancelled in Python, so a deadline in this pool would bound only
        how long we *wait*: the hung request would keep running, and the
        pool's shutdown would join it anyway. A per-request deadline belongs
        on the HTTP client (``openai.OpenAI(timeout=...)`` in
        ``runtime/client.py``, currently the SDK default of 600s), which is
        the layer that can actually abandon the socket. Adding a fake one
        here would be worse than none, because it would read as a guarantee.

    Note:
        The live ``evaluate_branch`` flow delegates predict parallelism to
        mlflow's own harness (sized via ``MLFLOW_GENAI_EVAL_MAX_WORKERS``)
        so that mlflow builds a per-row trace carrying the ``RETRIEVER``
        span that ``RetrievalGroundedness`` scores against. Pre-computing
        outputs here and passing them as a static dataset would yield a
        root-span-only trace and make ``RetrievalGroundedness`` raise, so
        this primitive is exercised directly by the unit tests and is
        available for offline/pre-compute paths that do not need traces.
    """
    if case_ids is None:
        ids = [str(i) for i in range(len(queries))]
    elif len(case_ids) != len(queries):
        raise ValueError(
            f"case_ids has {len(case_ids)} entries for {len(queries)} queries — "
            "a mismatch would attribute records to the wrong cases"
        )
    else:
        ids = list(case_ids)

    def _interrupted(idx: int) -> CaseRecord:
        return CaseRecord(case_id=ids[idx], outcome=CaseOutcome.INTERRUPTED)

    if n_workers <= 1:
        # Sequential path — same per-row error isolation as the parallel
        # path, so the contract holds uniformly for both.
        records: list[CaseRecord] = []
        for i, q in enumerate(queries):
            try:
                records.append(
                    _predict_one(
                        predict_fn, q, case_id=ids[i], row=i, max_retries=max_retries
                    )
                )
            except KeyboardInterrupt:
                # This row and every row after it were never assessed. They
                # are interrupted, not errored: nothing is wrong with the
                # infrastructure, so they must not feed the error-rate guard.
                records.extend(_interrupted(j) for j in range(i, len(queries)))
                raise RunInterrupted(records) from None
        return records

    slots: list[CaseRecord | None] = [None] * len(queries)
    future_to_idx: dict[Future[CaseRecord], int] = {}
    executor = ThreadPoolExecutor(max_workers=n_workers)
    try:
        try:
            # Submission is INSIDE the try: a Ctrl-C landing here would
            # otherwise escape as a bare KeyboardInterrupt, discarding the rows
            # already submitted and breaking the promise that a killed run is
            # readable.
            for i, q in enumerate(queries):
                future_to_idx[
                    executor.submit(
                        _predict_one,
                        predict_fn,
                        q,
                        case_id=ids[i],
                        row=i,
                        max_retries=max_retries,
                    )
                ] = i
            for future in as_completed(future_to_idx):
                slots[future_to_idx[future]] = future.result()
        except KeyboardInterrupt:
            raise RunInterrupted(
                _drain_after_interrupt(executor, future_to_idx, slots, ids)
            ) from None
    finally:
        # wait=False: the in-flight rows were already waited for inside
        # _drain_after_interrupt on the interrupt path, and on the normal path
        # every future has completed. Waiting again here would be the second
        # place a hung row could block, with no records to show for it.
        executor.shutdown(wait=False)

    # _predict_one never raises for a row (it converts an exception into an
    # error record), and as_completed yields every submitted future, so every
    # slot is filled by the time we get here. Reconstructed rather than filtered
    # so the returned list is the same length as ``queries`` even with asserts
    # stripped under ``python -O`` -- a short list would silently shift every
    # index-based join downstream (case_ids, examples[i]) by one.
    assert all(r is not None for r in slots)
    return [r or _interrupted(i) for i, r in enumerate(slots)]


def evaluate_branch(
    *,
    scaffold_root: Path | str,
    runtime_config_path: Path | str | None = None,
    kb_dir: Path | str = "data/kb",
    golden_set_path: Path | str = "data/golden_set.jsonl",
    evaluator_path: str | Path | None = None,
    profile: str | None = None,
    mode: str | None = None,
    allow_test: bool = False,
    include_safety: bool = False,
    runtime_client: OpenAI | None = None,
    judge_client: OpenAI | None = None,
) -> EvalReport:
    """Run the active scorers against a sub-set of the golden set."""
    scaffold_path = Path(scaffold_root)
    runtime_path = (
        Path(runtime_config_path)
        if runtime_config_path is not None
        else default_runtime_config_path(scaffold_path)
    )

    snapshot = load_harness(scaffold_path, runtime_path)
    cfg: EvalConfig = snapshot.config.eval
    selected_mode = mode or cfg.default_mode
    if selected_mode == "test" and not allow_test:
        raise ValueError("test mode is held out and may only be run by explicit finalization")
    if selected_mode not in cfg.modes:
        raise ValueError(
            f"mode {selected_mode!r} not in harness/config.yaml > eval.modes ({list(cfg.modes)})"
        )
    if profile:
        mlflow.set_tracking_uri(f"databricks://{profile}")
        os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
    mlflow.set_experiment(snapshot.config.experiments.eval)

    enable_runtime_tracing()

    # Both the runtime agent and the judge route through the AI Gateway
    # client (the sole LLM route). The gateway resolves host + token from
    # the environment; when ``profile`` is set above it is already in
    # ``DATABRICKS_CONFIG_PROFILE``, which the SDK honors at token-refresh
    # time. ``profile`` is therefore not passed to the factory.
    runtime_client = runtime_client or build_gateway_client()
    judge_client = judge_client or build_gateway_client()

    examples = load_golden_set(golden_set_path)
    selected = _select_mode_examples(examples, cfg=cfg, selected_mode=selected_mode)

    if snapshot.config.mode == "code":
        # Code mode: import the active MemorySystem subclass and call
        # predict() per row instead of the LLM agent tool-calling loop.
        # The same scorers (programmatic + LLM judges) score the output.
        memory_system = _load_memory_system(
            snapshot.config.agent_module,
            llm_client=runtime_client,
            model=snapshot.config.runtime_endpoint,
        )

        def predict_fn(query: str, **_kwargs: Any) -> str:
            # Wrap in an explicit root span so every row yields a trace. See the
            # prompt-mode predict_fn below for the full rationale, and
            # ``_traced_predict`` for why the body's exception is re-raised
            # *outside* the span rather than thrown into it.
            def _body() -> str:
                answer, _metadata = memory_system.predict(query)
                return answer

            return _traced_predict({"query": query}, _body)
    else:
        # Prompt mode: compose the system prompt from scaffold/ and run
        # the AnvilAgent tool-calling loop against the runtime endpoint.
        tool_executor = make_kb_executor(kb_dir)
        agent = AnvilAgent(
            scaffold_root=scaffold_path,
            runtime_config_path=runtime_path,
            source=SOURCE_EVAL,
            client=runtime_client,
            tool_executor=tool_executor,
        )

        def predict_fn(query: str, **_kwargs: Any) -> str:
            # Wrap each row in an explicit root CHAIN span so the row
            # yields a real per-row trace carrying the ``RETRIEVER`` span
            # that ``RetrievalGroundedness`` scores. The harness retrieves
            # each row's trace via ``mlflow.get_trace(request_id)``; without
            # this span a row can leave ``eval_item.trace`` None.
            #
            # ``_resilient_eval_harness`` (PR #21) is the safety net: it
            # makes ``_get_new_expectations`` None-safe and falls back to a
            # minimal trace, so a None-trace row no longer crashes the run.
            # But the minimal trace lacks the ``RETRIEVER`` span, so
            # ``RetrievalGroundedness`` is degraded for those rows — this
            # span remains the primary guarantee.
            #
            # It also supersedes the fragile ``mlflow.openai.autolog`` path
            # (``enable_runtime_tracing``): ``AnvilAgent.predict`` calls
            # ``tag_current_trace`` (``mlflow.update_current_trace``) before
            # any chat call, when no span is active — the "No active trace
            # found" warning — and on the live backend the autolog trace was
            # not retrievable by the row's request id. This root span gives
            # ``tag_current_trace`` an active trace to tag (the warning
            # disappears) and nests autolog's CHAT_MODEL spans and the
            # ``search_knowledge_base`` RETRIEVER span under one coherent
            # per-row trace. The span ends (and the trace exports) when the
            # ``with`` block exits — async logging is disabled during eval
            # (``is_evaluate=True``), so the trace is available immediately.
            def _body() -> str:
                request = ResponsesAgentRequest(
                    input=[{"type": "message", "role": "user", "content": query}]
                )
                response = agent.predict(request)
                return _extract_final_text(response)

            return _traced_predict({"query": query}, _body)

    aggregate_scorer_configs = list(cfg.scorers)
    aggregate_scorer_names = [c.name for c in aggregate_scorer_configs]
    weights = {c.name: c.weight for c in aggregate_scorer_configs}
    scorer_fingerprint = compute_scorer_fingerprint(aggregate_scorer_configs)
    active_scorer_configs = list(aggregate_scorer_configs)
    active_scorer_names = list(aggregate_scorer_names)
    if include_safety and "safety" not in active_scorer_names:
        active_scorer_configs.append(ScorerConfig(name="safety"))
        active_scorer_names.append("safety")

    scorers = build_scorers(
        judge_client=judge_client,
        judge_model=snapshot.config.judge_endpoint,
        scorer_configs=active_scorer_configs,
        evaluator_path=evaluator_path,
    )
    dataset = _build_dataset(selected)

    # Wire anvil's ``eval.n_workers`` into mlflow's parallel predict/score
    # pool. mlflow's harness already runs ``predict_fn`` per row in a
    # ``ThreadPoolExecutor`` sized by ``MLFLOW_GENAI_EVAL_MAX_WORKERS``
    # (default 10); setting it from the config makes the configured value
    # actually control concurrency. We keep passing ``predict_fn`` (not
    # pre-computed ``outputs``) so mlflow builds a per-row trace carrying
    # the ``RETRIEVER`` span that ``RetrievalGroundedness`` requires — a
    # static-dataset trace is root-span-only and makes that scorer raise.
    # The env var is saved/restored so the override is scoped to this call.
    # NOTE: the env var is process-global, so this override is not safe for
    # concurrent ``evaluate_branch`` calls in one process; the optimizer
    # runs rounds/evals synchronously, so this is not a live issue today.
    n_workers = max(1, cfg.n_workers)
    # Filled by the harness shim with trace_id -> error_message for any row
    # whose prediction raised. Those rows are excluded from the scores.
    errored: dict[str, str] = {}
    # Rows whose trace never became retrievable, so mlflow cannot represent them
    # in the result frame. Counted so the sample-size floor sees them.
    dropped: set[str] = set()
    kept_rows: list[int] = []
    _prev_workers = os.environ.get(_MLFLOW_MAX_WORKERS_ENV)
    os.environ[_MLFLOW_MAX_WORKERS_ENV] = str(n_workers)
    _prev_skip = os.environ.get(_MLFLOW_SKIP_VALIDATION_ENV)
    os.environ[_MLFLOW_SKIP_VALIDATION_ENV] = "True"
    try:
        # On the Databricks Tracing Server, async export can race the eval
        # harness's immediate per-row get_trace(request_id). A missing trace
        # then reaches scoring as None and crashes _get_new_expectations.
        # Keep export synchronous until evaluate has finished reading traces
        # (PR #17; the env var is also forced from process start in
        # anvil/__init__.py because the exporter caches the flag at construction).
        # ``_resilient_eval_harness`` is the guarantee: even if a row's trace
        # is still None despite the above, the harness shim yields no
        # expectations for that row (no crash) and the _run_predict fallback
        # synthesizes a minimal trace so the run completes with a real
        # result DataFrame. See its docstring for the exact harness.py
        # symbols and lines patched.
        with (
            _resilient_eval_harness(
                error_sink=errored, dropped_sink=dropped, kept_rows_sink=kept_rows
            ),
            _synchronous_trace_logging(),
        ):
            result = mlflow.genai.evaluate(
                data=dataset,
                scorers=scorers,
                predict_fn=predict_fn,
            )
    finally:
        if _prev_workers is None:
            os.environ.pop(_MLFLOW_MAX_WORKERS_ENV, None)
        else:
            os.environ[_MLFLOW_MAX_WORKERS_ENV] = _prev_workers
        if _prev_skip is None:
            os.environ.pop(_MLFLOW_SKIP_VALIDATION_ENV, None)
        else:
            os.environ[_MLFLOW_SKIP_VALIDATION_ENV] = _prev_skip

    if result.result_df is None:
        # mlflow returns None here for more than one reason -- every row filtered,
        # an empty ``search_traces``, or an exception swallowed inside its own
        # try -- so report what is actually known instead of asserting a cause and
        # sending the operator hunting for the wrong thing. Raising beats letting
        # ``len(None)`` surface as a TypeError three frames deeper, which is how
        # this presented live. The run id matters: the eval was paid for and the
        # traces are still inspectable.
        raise RuntimeError(
            f"mlflow produced no result frame for run {result.run_id!r}: "
            f"{len(dataset)} row(s) submitted, {len(dropped)} known to have lost "
            f"their trace during predict, {len(errored)} errored. The eval ran and "
            "was paid for, but nothing can be scored from it."
        )

    # Rows dropped anywhere between submission and the frame. Derived, not
    # accumulated: mlflow nulls traces AFTER our shims run -- its
    # ``_refresh_eval_result_traces`` reassigns ``eval_item.trace`` from
    # ``mlflow.get_trace``, which returns None on a miss -- so a counter fed only
    # at predict time undercounts, and the sample-size floor built on it would be
    # defeated by the exact flakiness it targets. The frame is the ground truth.
    n_dropped = max(0, len(dataset) - len(result.result_df))

    # Realign ``examples`` with the surviving frame. The join is positional, so a
    # filtered frame otherwise attributes every failure to the wrong golden case.
    # ``kept_rows`` indexes the eval_results list, whose order the frame follows.
    aligned = [selected[i] for i in kept_rows if i < len(selected)] if kept_rows else selected
    if len(aligned) != len(result.result_df):
        # Checked unconditionally, not only when n_dropped is nonzero. An empty
        # ``kept_rows`` means either "the shim never ran" (fall back to selected,
        # correct) or "the shim filtered every row" (selected is longer than the
        # frame, and the positional join would mislabel every case). Those two are
        # indistinguishable from the sink alone, and relying on mlflow returning a
        # None frame to rule the second one out makes this correctness depend on
        # someone else's early return. Comparing lengths keeps the invariant here.
        logger.warning(
            "could not realign examples with the result frame (%s example(s), %s "
            "row(s)); per-case attribution in failures/errors is unreliable for "
            "this run",
            len(aligned),
            len(result.result_df),
        )

    experiment = mlflow.get_experiment_by_name(snapshot.config.experiments.eval)
    return _aggregate_report(
        result_df=result.result_df,
        metrics=result.metrics,
        scorer_names=active_scorer_names,
        aggregate_scorer_names=aggregate_scorer_names,
        weights=weights,
        examples=aligned,
        run_id=result.run_id,
        experiment_id=experiment.experiment_id if experiment else "",
        mode=selected_mode,
        scorer_fingerprint=scorer_fingerprint,
        errored=errored,
        n_dropped_rows=n_dropped,
        attempted_examples=selected,
    )
