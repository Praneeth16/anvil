"""End-to-end evaluation runner — driver of ``mlflow.genai.evaluate``.

Wraps :func:`mlflow.genai.evaluate` with the active scorers (3 by
default; Safety opt-in), the golden set sub-set per mode
(``quick``/``standard``/``full``), and an :class:`AnvilAgent`
constructed with ``source=SOURCE_EVAL``.

Sequential by design today: ``mlflow.genai.evaluate`` in 3.10.x does
not accept ``n_workers``. Parallelism with a thread pool is the
documented Phase-2c follow-up; for now, sequential keeps the rate-
limit footprint predictable.

Public surface:

* :func:`evaluate_branch` — driver function callable from
  ``scripts/evaluate.py`` or another module.
* :class:`EvalReport` — aggregate / per-judge / per-bucket / failures.
"""

from __future__ import annotations

import contextlib
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.types.responses import ResponsesAgentRequest
from openai import OpenAI

from anvil.data import load_golden_set, select_subset
from anvil.eval.cache import compute_scorer_fingerprint
from anvil.eval.scorers import build_scorers
from anvil.observability import SOURCE_EVAL, enable_runtime_tracing
from anvil.runtime.agent import AnvilAgent
from anvil.runtime.client import build_databricks_client
from anvil.runtime.loader import default_runtime_config_path, load_harness
from anvil.runtime.models import EvalConfig, ScorerConfig
from anvil.tools.search_knowledge_base import make_kb_executor


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
            if key not in expectations and (
                key.startswith("json_") or key.startswith("expected_")
            ):
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
) -> EvalReport:
    n_rows = len(result_df)

    per_judge_rows: dict[str, list[float | None]] = {
        name: [_row_score(result_df.iloc[i], name) for i in range(n_rows)] for name in scorer_names
    }

    def _mean(values: list[float | None]) -> float:
        nums = [v for v in values if v is not None]
        return sum(nums) / len(nums) if nums else 0.0

    per_judge: dict[str, float] = {}
    for name in scorer_names:
        metric_key = f"{name}/mean"
        if metric_key in metrics:
            per_judge[name] = float(metrics[metric_key])
        else:
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
        category = _category_for_row(result_df.iloc[i], examples, i)
        if category:
            bucket_rows[category].append(i)
    per_bucket: dict[str, dict[str, float]] = {}
    for bucket, idxs in bucket_rows.items():
        per_bucket[bucket] = {
            name: _mean([per_judge_rows[name][i] for i in idxs]) for name in scorer_names
        }

    failures: list[dict[str, Any]] = []
    trace_ids: list[str] = []
    for i in range(n_rows):
        row = result_df.iloc[i]
        trace_id = row.get("trace_id") if hasattr(row, "get") else None
        if trace_id:
            trace_ids.append(str(trace_id))
        judge_failures = [
            name for name in scorer_names if (s := per_judge_rows[name][i]) is not None and s < 1.0
        ]
        if not judge_failures:
            continue
        category = _category_for_row(row, examples, i)
        example_id = examples[i]["example_id"] if i < len(examples) else ""
        query = examples[i]["query"] if i < len(examples) else ""
        failures.append(
            {
                "example_id": example_id,
                "query": query,
                "category": category,
                "judge_failures": judge_failures,
                "trace_id": trace_id,
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
        cost_metrics={
            "total_context_chars": float(
                sum(len(str(ex.get("query", ""))) for ex in examples[:n_rows])
            ),
            "n_rows": float(n_rows),
        },
        scorer_fingerprint=scorer_fingerprint,
    )


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
    mode_config = cfg.modes[selected_mode]

    if profile:
        mlflow.set_tracking_uri(f"databricks://{profile}")
        os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
    mlflow.set_experiment(snapshot.config.experiments.eval)

    enable_runtime_tracing()

    runtime_client = runtime_client or build_databricks_client(profile=profile)
    judge_client = judge_client or build_databricks_client(profile=profile)

    examples = load_golden_set(golden_set_path)
    selected = select_subset(examples, buckets=mode_config.buckets)

    tool_executor = make_kb_executor(kb_dir)
    agent = AnvilAgent(
        scaffold_root=scaffold_path,
        runtime_config_path=runtime_path,
        source=SOURCE_EVAL,
        client=runtime_client,
        tool_executor=tool_executor,
    )

    def predict_fn(query: str, **_kwargs: Any) -> str:
        request = ResponsesAgentRequest(
            input=[{"type": "message", "role": "user", "content": query}]
        )
        response = agent.predict(request)
        # Drain async export queue so eval_item.trace is not None
        # downstream. Documented in the legacy lessons (rounds 3-5).
        mlflow.flush_trace_async_logging()
        with contextlib.suppress(AttributeError, TypeError):
            mlflow.end_trace()
        return _extract_final_text(response)

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

    result = mlflow.genai.evaluate(
        data=dataset,
        scorers=scorers,
        predict_fn=predict_fn,
    )

    experiment = mlflow.get_experiment_by_name(snapshot.config.experiments.eval)
    return _aggregate_report(
        result_df=result.result_df,
        metrics=result.metrics,
        scorer_names=active_scorer_names,
        aggregate_scorer_names=aggregate_scorer_names,
        weights=weights,
        examples=selected,
        run_id=result.run_id,
        experiment_id=experiment.experiment_id if experiment else "",
        mode=selected_mode,
        scorer_fingerprint=scorer_fingerprint,
    )
