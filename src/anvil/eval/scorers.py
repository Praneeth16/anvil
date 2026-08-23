"""Scorers for the ANVIL evaluation runner.

Three scorers are active by default and contribute to the
aggregate:

* :class:`mlflow.genai.scorers.Correctness` — reads
  ``expected_facts`` from the row's ``expectations`` dict.
* :func:`retrieval_groundedness` — wraps
  :class:`mlflow.genai.scorers.RetrievalGroundedness`, which extracts
  retrieved chunks from the trace's ``RETRIEVER`` span (the
  ``_KbToolExecutor`` emits one) and judges grounding. The wrapper
  supplies the *applicability* rule the bare scorer has no way to
  express — see :func:`_build_groundedness_scorer`.
* :func:`refusal_appropriateness` — custom binary verdict via a
  Databricks-served LLM judge. Asks the judge whether the agent
  refused iff ``should_refuse=True`` and whether the refusal was
  operationally correct.

:class:`mlflow.genai.scorers.Safety` is available behind
``--include-safety`` but is **not** in the aggregate by default. In
all 11 measured rounds it has been a flat 1.000 — it provides no
gradient. Treat it as a guard-rail that aborts a round if it ever
drops below ``safety_guard_threshold``.

**Tracing suppression around the judge call is scoped, not global**, and the
difference is the whole point. The custom judge's
``chat.completions.create`` is autologged by ``mlflow.openai.autolog``, which
without suppression spawns an orphan ``CHAT_MODEL`` trace per row. That much was
always worth preventing. It used to be prevented with
``mlflow.tracing.disable()``, which installs a **process-global**
``NoOpTracerProvider`` — and ``mlflow.genai.evaluate`` runs scoring
**concurrently with predictions**: two pools, ``MlflowGenAIEvalPredict`` and
``MlflowGenAIEvalScore``, with a score task submitted the moment one row's
prediction returns (``genai/evaluation/harness.py:569-603``). So a judge call on
a scorer thread blinded the tracer for every prediction thread running at that
moment, and the damage landed two ways:

* A prediction wholly inside the window registers no trace under the
  ``eval_request_id`` that ``_run_predict`` resolves by, so
  ``mlflow.get_trace`` returns ``None`` and the row loses its trace — the
  failure ``_resilient_eval_harness`` was built to survive.
* A prediction only *partly* inside it keeps its trace but loses whichever
  spans were emitted while the provider was a no-op. Lose the
  ``search_knowledge_base`` ``RETRIEVER`` span and
  ``RetrievalGroundedness`` raises "No retrieval context found in the trace"
  while ``Correctness`` and the refusal judge — which read inputs and outputs,
  not the trace — score the same row perfectly happily.

``mlflow.tracing.context(enabled=False)`` is the right instrument and it exists
in 3.11.1: it sets a **ContextVar**, so suppression is confined to the calling
thread, and its own docstring notes it "does not affect the global tracing state
set by ``mlflow.tracing.disable``". Verified rather than assumed — inside the
block a span comes back as ``MLFLOW_NO_OP_SPAN_TRACE_ID`` while
``is_tracing_enabled()`` stays ``True``, and a concurrently running thread still
gets a real trace id.

Worth recording how the earlier attempt went wrong, because the reasoning looks
sound: ``trace_disabled``, ``disable_autologging`` and
``disable_discrete_autologging`` were each checked, each found to be
process-global, and "no scoped form exists" was concluded from three misses
rather than from the API. The fix then reached for
``MLFLOW_GENAI_EVAL_ENABLE_SCORER_TRACING``, which does remove the global switch
but pays for it: 24 extra retained root traces per quick run, plus an unguarded
``set_trace_tag`` server call per scorer whose failure can abort scoring — a new
dependency on the very tracing server this module already works around.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import mlflow
from mlflow.entities import AssessmentSource, Feedback
from mlflow.genai import judges
from mlflow.genai.scorers import Correctness, Safety, scorer
from mlflow.genai.utils.trace_utils import (
    extract_request_from_trace,
    extract_response_from_trace,
    extract_retrieval_context_from_trace,
)
from openai import OpenAI

from anvil.runtime.models import ScorerConfig

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "databricks-claude-sonnet-4-6"

REFUSAL_SCORER_NAME = "refusal_appropriateness"

GROUNDEDNESS_SCORER_NAME = "retrieval_groundedness"

# Set by ``anvil.eval.runner``'s harness shim on a trace it synthesized because
# the tracing server would not return the real one. Such a trace is root-span-only
# -- no ``RETRIEVER`` span -- so a trace-reading scorer must not treat its absence
# as evidence about the agent. Defined here, next to the only reader, and imported
# by the writer.
SYNTHESIZED_TRACE_TAG = "anvil.synthesized_trace"

# Bumped when a scorer's *semantics* change without its config changing.
# ``compute_scorer_fingerprint`` folds this in, so a cached baseline measured
# under the old meaning is correctly reported as incomparable instead of
# silently becoming the bar a 50-round run chases. Config-level changes
# (weight, check_function) are already covered by the config itself; this
# covers the case the config cannot see.
#
# retrieval_groundedness:
#   v1 — scored whichever rows happened to reach the judge, and whichever
#        retrieval span happened to be flattened last.
#   v2 — applicability rule added (see ``_build_groundedness_scorer``).
#   v3 — one verdict over the union of all retrieval spans instead of one
#        verdict per span collapsed last-wins.
#   v4 — a row whose trace was lost and replaced by a synthesized stand-in is
#        not applicable rather than ungrounded.
SCORER_SEMANTICS_VERSIONS: dict[str, int] = {
    GROUNDEDNESS_SCORER_NAME: 4,
}

# Default location of the programmatic check-function module, relative
# to the harness working directory (matches the ``data/golden_set.jsonl``
# convention). Overridable per-call via ``evaluator_path``.
DEFAULT_EVALUATOR_PATH = Path("data/evaluator.py")

# Maps the YAML scorer-name strings to the actual scorer factories.
# ``retrieval_groundedness`` is deliberately absent: it is built by
# :func:`_build_groundedness_scorer`, which wraps the mlflow scorer rather than
# handing it over bare.
_BUILTIN_SCORERS = {
    "correctness": Correctness,
    "safety": Safety,
}

# The refusal judge's description of the domain it is grading. This is the only
# domain-specific text left in the eval plane, and it used to be a literal in
# the template below -- which meant pointing ANVIL at a new domain required
# editing library source. It is config now (``judge_domain_name`` and
# ``judge_domain_context`` in harness/config.yaml), and deliberately NOT part of
# ``scaffold/``: the optimizer is graded by this judge, and PR #1 removed its
# ability to edit its own grader. See docs/decisions.md D8.
#
# ``None`` at every call site means "use these defaults", and the defaults
# reproduce the shipped NeoVolt prompt byte-for-byte -- so every baseline cached
# before this field existed stays valid and no semantics bump is needed.
DEFAULT_JUDGE_DOMAIN_NAME = "NeoVolt"
DEFAULT_JUDGE_DOMAIN_CONTEXT = """\
Domain: NeoVolt, a fictional electricity & gas utility. The agent
has a knowledge base of NeoVolt policies and a `search_knowledge_base`
tool. The agent should refuse out-of-scope or unanswerable questions
and answer in-scope ones using the knowledge base."""

_JUDGE_PROMPT_TEMPLATE = """\
You are an evaluator grading a customer-support agent's response.

{domain_context}

You must judge a single binary axis: REFUSAL APPROPRIATENESS.

Definitions:
- "Refusal": the agent declines to answer the user's question, e.g.
  "I can only help with {domain_name}-related questions", "I don't have
  that information in my knowledge base", "I cannot help with that".
- An *appropriate refusal* happens when `should_refuse=true` and the
  agent declines without leaking partial answers, without inventing
  facts, and without redirecting the user away from a legitimate
  channel.
- An *appropriate non-refusal* happens when `should_refuse=false` and
  the agent answers the question (correctly or otherwise — that
  axis is judged by Correctness, not by you).
- A *failure* is over-refusing a valid in-scope query, half-answering
  while refusing (leaking information), or answering an out-of-scope
  query that should have been refused.

Inputs:
USER QUERY: {query}

AGENT RESPONSE:
{response}

GROUND TRUTH:
- should_refuse: {should_refuse}
- judge notes: {notes}

Output JSON ONLY (no prose, no code fences) with these keys:
{{
  "verdict": "pass" | "fail",
  "rationale": "<one short sentence explaining the verdict>"
}}
"""


@dataclass(frozen=True)
class _JudgeContext:
    client: OpenAI
    model: str
    domain_name: str = DEFAULT_JUDGE_DOMAIN_NAME
    domain_context: str = DEFAULT_JUDGE_DOMAIN_CONTEXT


def _judge_prompt(
    query: str,
    response: str,
    should_refuse: bool,
    notes: str,
    *,
    domain_name: str = DEFAULT_JUDGE_DOMAIN_NAME,
    domain_context: str = DEFAULT_JUDGE_DOMAIN_CONTEXT,
) -> str:
    return _JUDGE_PROMPT_TEMPLATE.format(
        domain_name=domain_name,
        domain_context=domain_context,
        query=query,
        response=response,
        should_refuse=str(should_refuse).lower(),
        notes=notes or "(none)",
    )


def _parse_judge_json(raw: str) -> dict:
    if not raw:
        raise ValueError("judge returned empty content")
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object in judge output: {text[:200]!r}")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError(f"judge output is not a JSON object: {text[:200]!r}")
    if obj.get("verdict") not in ("pass", "fail"):
        raise ValueError(f"judge verdict missing or invalid: {obj!r}")
    return obj


def _build_refusal_scorer(ctx: _JudgeContext):
    """Return a ``@scorer`` that judges refusal appropriateness."""
    source = AssessmentSource(source_type="LLM_JUDGE", source_id=ctx.model)

    @scorer(name=REFUSAL_SCORER_NAME)
    def refusal_appropriateness(inputs: dict, outputs: str, expectations: dict) -> Feedback:
        query = inputs.get("query", "")
        should_refuse = bool(expectations.get("should_refuse", False))
        notes = expectations.get("notes_for_judge", "")
        prompt = _judge_prompt(
            query,
            str(outputs),
            should_refuse,
            notes,
            domain_name=ctx.domain_name,
            domain_context=ctx.domain_context,
        )
        try:
            # ContextVar-scoped, so this silences the judge's own autolog on THIS
            # thread only. Never mlflow.tracing.disable() here: that swaps the
            # global tracer provider, and mlflow scores this row while still
            # predicting others. See the module docstring.
            with mlflow.tracing.context(enabled=False):
                response = ctx.client.chat.completions.create(
                    model=ctx.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    temperature=0,
                )
            raw = response.choices[0].message.content or ""
            parsed = _parse_judge_json(raw)
        except Exception as exc:
            return Feedback(
                value=False,
                rationale=f"judge JSON malformed: {exc}",
                source=source,
            )
        return Feedback(
            value=parsed["verdict"] == "pass",
            rationale=parsed.get("rationale", ""),
            source=source,
        )

    return refusal_appropriateness


# ---------------------------------------------------------------------------
# Retrieval groundedness — applicability, not just grounding.
# ---------------------------------------------------------------------------


def _expects_retrieval(expectations: Any) -> bool:
    """Whether this row's ground truth says the agent had to retrieve.

    Read off ``expected_doc_ids`` rather than ``should_refuse`` because
    ``expected_doc_ids`` is what groundedness is *about*. In the shipped golden
    set the two agree exactly (empty iff ``should_refuse``, 4 of 20 rows, all
    ``out_of_scope``), but they are separate columns and a future row could
    legitimately expect a refusal to cite policy.
    """
    if not isinstance(expectations, dict):
        return False
    return bool(expectations.get("expected_doc_ids"))


def _retrieved_chunks(trace: Any) -> list[Any]:
    """Every chunk the agent retrieved, across all retrieval spans, flattened.

    Uses mlflow's own extractor — the same one
    :meth:`RetrievalGroundedness.__call__` uses — so "the judge will find
    context" and "we predicted it would" cannot drift apart. Hand-rolling a
    ``span_type == RETRIEVER`` scan would be a second, silently diverging
    definition.

    Flattened because grounding is a question about the answer and *everything
    the agent found*, not about each search in isolation. mlflow's scorer instead
    returns one feedback per retrieval span, and
    ``construct_eval_result_df`` flattens those into a single ``{name}/value``
    column where the last one wins. Two things follow, and both are wrong:

    * **It is a lever.** Which span lands last is the agent's choice, so a final
      narrow search whose chunks trivially support a closing sentence carries the
      row. Same shape as the applicability hole — a score whose denominator, or
      here whose *subject*, the agent picks.
    * **It mismeasures multi-hop rows.** 6 of the 20 golden rows expect 2-4
      documents, so the agent searches several times and no single search
      supports the whole answer. Judging the complete answer against only the
      last search's chunks understates those rows systematically. The legacy
      baseline's ``multi_hop: {retrieval_groundedness: 0.5}`` is consistent with
      exactly that.

    Returns ``[]`` when nothing is extractable — extraction failing is not
    evidence the agent retrieved.
    """
    try:
        by_span = extract_retrieval_context_from_trace(trace)
    except Exception:
        return []
    if not by_span:
        return []
    return [chunk for chunks in by_span.values() for chunk in chunks]


def _is_synthesized_trace(trace: Any) -> bool:
    """Whether this trace is the harness's stand-in for one the server lost.

    See :data:`SYNTHESIZED_TRACE_TAG`. Defensive about the shape because a
    ``Trace`` reaches scorers from several paths and a missing ``info`` or
    ``tags`` must read as "not synthesized" rather than raise inside a scorer.
    """
    try:
        tags = trace.info.tags
    except AttributeError:
        return False
    return bool(tags) and str(tags.get(SYNTHESIZED_TRACE_TAG, "")).lower() == "true"


def _build_groundedness_scorer(*, name: str = GROUNDEDNESS_SCORER_NAME):
    """Return a ``@scorer`` wrapping ``RetrievalGroundedness`` with an
    applicability rule.

    :class:`mlflow.genai.scorers.RetrievalGroundedness` has exactly one
    behaviour when a trace carries no ``RETRIEVER`` span: it raises. mlflow
    turns that into a ``SCORER_ERROR`` feedback with no value, and anvil's
    aggregation drops valueless rows from the mean. Two very different
    situations therefore reached the gate as the same silent exclusion:

    * The row was **never meant to retrieve.** Every ``out_of_scope`` row in
      the golden set has ``expected_doc_ids == []``; the agent correctly
      refuses, never calls ``search_knowledge_base``, and there is no
      ``RETRIEVER`` span to ground against. Grounding is *not applicable*, and
      quick mode contains exactly 2 such rows out of 8 — so a healthy run
      always logged 2 scorer errors and nobody could tell them from real ones.
    * The row **was** meant to retrieve and the agent did not. That is a
      finding about the agent, and excluding it inverts the incentive: the
      optimizer is graded on its own output, so any route from "change
      behaviour" to "score goes up" gets taken. Groundedness is binary, so
      withholding retrieval on rows it was losing moved this judge from
      ``1/8 = 0.125`` to ``1/1 = 1.0`` — an **0.875 swing available purely by
      searching less**, and the largest single lever in the scoring system.
      ``pareto.enabled`` is currently ``false``, so only ``aggregate`` gates a
      round, but the aggregate is the weighted mean of the per-judge values, so
      the lever reaches it either way.

    Hence: ``None`` (mlflow reads that as "no assessment", see
    ``standardize_scorer_value``) when nothing was expected, ``"no"`` when
    documents were expected and no retrieval happened, and the real judge
    otherwise. Abstaining now costs what a wrong answer costs, which is the
    only arrangement under which "the agent stopped retrieving" cannot look
    like an improvement.

    And it emits **one verdict per retrieval span**, which
    ``construct_eval_result_df`` then collapses last-wins into a single column.
    So the row's score was decided by whichever search happened to be flattened
    last — the agent's choice, and a lever in its own right — and a multi-hop
    answer drawn from several searches was judged against only the final one. So
    the wrapper asks the grounding question once, against the union of everything
    retrieved. See :func:`_retrieved_chunks`.

    ``prompts/anvil-round.md`` has told the optimizer since round one that this
    scorer is "only computed for in-scope rows with ``expected_doc_ids``". This
    makes that true.
    """
    source = AssessmentSource(source_type="CODE", source_id=f"applicability:{name}")

    @scorer(name=name)
    def retrieval_groundedness(expectations: dict, trace: Any) -> Feedback | None:
        if not _expects_retrieval(expectations):
            # Not applicable. Returning None yields zero feedbacks, so the row
            # contributes nothing to this judge's mean and — unlike raising —
            # is not counted as a scorer error either.
            return None
        if _is_synthesized_trace(trace):
            # The prediction ran; the tracing server would not give back its
            # trace, so the harness substituted a root-span-only stand-in. There
            # is no RETRIEVER span to find and no evidence either way. Returning
            # "no" here would score infrastructure damage as an agent failure --
            # and reliably, since a lost trace is exactly what this repo's
            # resilience work keeps hitting live.
            return None
        chunks = _retrieved_chunks(trace)
        if not chunks:
            return Feedback(
                # "no" and not False: the value type of the scorer being wrapped
                # is Literal["yes", "no"], and mixing a bool into the same
                # column mlflow averages is asking for trouble.
                value="no",
                rationale=(
                    "expected_doc_ids were specified but the trace has no retrieval "
                    "context, so the agent answered without consulting the knowledge "
                    "base. Scored as ungrounded rather than skipped: skipping would "
                    "reward not retrieving."
                ),
                source=source,
            )
        # Same judge and the same request/response extraction the built-in scorer
        # uses; only the context differs, and only by being complete.
        return judges.is_grounded(
            request=extract_request_from_trace(trace),
            response=extract_response_from_trace(trace),
            context=chunks,
            name=name,
        )

    return retrieval_groundedness


# ---------------------------------------------------------------------------
# Programmatic scorers — deterministic check functions, no LLM call.
# ---------------------------------------------------------------------------


def load_evaluator_module(evaluator_path: str | Path | None = None) -> ModuleType:
    """Dynamically import the programmatic check-function module.

    Resolves ``evaluator_path`` (default :data:`DEFAULT_EVALUATOR_PATH`,
    CWD-relative like ``data/golden_set.jsonl``) to an absolute path and
    imports it via :mod:`importlib` under a stable module name. The
    module is re-executed on every call — the eval runner builds scorers
    once per ``evaluate_branch`` call, so there is no per-row cost, and
    always-fresh execution avoids stale-cache bugs when the file is
    edited between runs.
    """
    path = Path(evaluator_path) if evaluator_path is not None else DEFAULT_EVALUATOR_PATH
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"evaluator module not found: {resolved}")
    spec = importlib.util.spec_from_file_location("anvil_evaluator", resolved)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_check_function(
    name: str | None,
    evaluator_path: str | Path | None = None,
):
    """Look up ``name`` on the evaluator module and return the callable.

    Raises ``ValueError`` if ``name`` is missing or not callable, so a
    typo in ``check_function`` fails at scorer-build time (before any
    row is scored) rather than mid-eval.
    """
    if not name:
        raise ValueError("check_function name is required for a programmatic scorer")
    module = load_evaluator_module(evaluator_path)
    fn = getattr(module, name, None)
    if not callable(fn):
        raise ValueError(f"check function {name!r} not found in {module.__file__}")
    return fn


def _clamp_score(score: float) -> float:
    # NaN comparisons are always False in Python, so a NaN passes both
    # the ``< 0.0`` and ``> 1.0`` guards and leaks through as NaN — which
    # would poison the aggregate. Reject any non-finite value (NaN or
    # inf) by mapping it to 0.0 so a misbehaving custom check cannot
    # corrupt the weighted average.
    if not math.isfinite(score):
        return 0.0
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return float(score)


def _run_programmatic_check(check_fn, inputs, outputs, expectations) -> float:
    """Invoke a check function with the ``(prediction, ground_truth)`` shape.

    Pure and mlflow-free so it is unit-testable in isolation. ``outputs``
    becomes the prediction string; ``expectations`` (the eval row's
    golden-set projection) becomes the ``ground_truth`` dict. The score
    is clamped to ``[0.0, 1.0]`` so a misbehaving custom check cannot
    poison the aggregate.
    """
    prediction = "" if outputs is None else str(outputs)
    ground_truth = dict(expectations) if isinstance(expectations, dict) else {}
    return _clamp_score(float(check_fn(prediction, ground_truth)))


def build_programmatic_scorer(*, name: str, check_fn):
    """Return a ``@scorer`` that wraps a deterministic check function.

    The returned scorer runs inside ``mlflow.genai.evaluate`` like the
    LLM judges, but its body is pure Python — it calls ``check_fn`` with
    the prediction and ground-truth dict and records the result as a
    ``Feedback`` with a ``CODE`` assessment source. No LLM call is made.
    """
    source = AssessmentSource(source_type="CODE", source_id=f"programmatic:{name}")

    @scorer(name=name)
    def _programmatic(inputs: dict, outputs: str, expectations: dict) -> Feedback:
        score = _run_programmatic_check(check_fn, inputs, outputs, expectations)
        return Feedback(value=score, rationale=f"programmatic:{name}", source=source)

    return _programmatic


def build_scorers(
    *,
    judge_client: OpenAI,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    scorer_configs: list[ScorerConfig] | None = None,
    evaluator_path: str | Path | None = None,
    judge_domain_name: str | None = None,
    judge_domain_context: str | None = None,
) -> list:
    """Return the active scorers ready for ``mlflow.genai.evaluate``.

    Args:
        judge_client: OpenAI-compatible client for the custom
            ``refusal_appropriateness`` judge. Not invoked for
            programmatic scorers.
        judge_model: Endpoint name for the custom judge.
        scorer_configs: The configured scorers (LLM + programmatic).
            Defaults to the three built-in LLM judges. Each
            ``type: llm`` scorer maps to its MLflow factory (or the
            custom refusal judge); each ``type: programmatic`` scorer
            loads its ``check_function`` from ``data/evaluator.py``.
        evaluator_path: Override path to the programmatic check-function
            module. Defaults to ``data/evaluator.py``.
        judge_domain_name: Short name of the domain, interpolated into the
            refusal judge's example phrasing. ``None`` uses
            :data:`DEFAULT_JUDGE_DOMAIN_NAME`.
        judge_domain_context: The refusal judge's description of the domain.
            ``None`` uses :data:`DEFAULT_JUDGE_DOMAIN_CONTEXT`. Set it to point
            the judge at a domain other than the shipped one; see
            ``examples/`` for a worked case.
    """
    if scorer_configs is None:
        scorer_configs = [
            ScorerConfig(name="correctness"),
            ScorerConfig(name="retrieval_groundedness"),
            ScorerConfig(name="refusal_appropriateness"),
        ]

    ctx = _JudgeContext(
        client=judge_client,
        model=judge_model,
        domain_name=judge_domain_name or DEFAULT_JUDGE_DOMAIN_NAME,
        domain_context=judge_domain_context or DEFAULT_JUDGE_DOMAIN_CONTEXT,
    )
    out: list = []
    for cfg in scorer_configs:
        if cfg.type == "programmatic":
            check_fn = load_check_function(cfg.check_function, evaluator_path)
            out.append(build_programmatic_scorer(name=cfg.name, check_fn=check_fn))
        else:  # llm
            if cfg.name == REFUSAL_SCORER_NAME:
                out.append(_build_refusal_scorer(ctx))
            elif cfg.name == GROUNDEDNESS_SCORER_NAME:
                out.append(_build_groundedness_scorer())
            elif cfg.name in _BUILTIN_SCORERS:
                out.append(_BUILTIN_SCORERS[cfg.name]())
            else:
                raise ValueError(f"unknown llm scorer name: {cfg.name!r}")
    return out
