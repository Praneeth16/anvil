"""Cached baseline for fast score-delta calculation.

Running ``mlflow.genai.evaluate`` against the parent branch every
round to compute ``score_delta_vs_parent`` doubles the wall-clock and
the rate-limit hit. Instead, we cache the parent's aggregate and only
recompute when explicitly requested (``--refresh-baseline``) or when
external dependencies change (model endpoint, scorer set,
golden-set rows).

Cache file: ``eval/runs/baseline.json``.

Schema::

    {
      "scaffold_commit_sha": "<40-char SHA>",
      "evaluated_at": "<UTC ISO8601>",
      "mode": "standard",
      "scorers": ["correctness", "retrieval_groundedness", ...],
      "runtime_endpoint": "databricks-claude-sonnet-4-6",
      "judge_endpoint": "databricks-claude-sonnet-4-6",
      "aggregate": 0.861,
      "per_judge": {...},
      "per_bucket": {...},
      "n_examples": 12,
      "mlflow_run_id": "..."
    }

If any field of the requesting context (mode / scorers / endpoints)
differs from the cache header, the cache is stale and must be
refreshed before producing a delta.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anvil.eval.scorers import (
    DEFAULT_JUDGE_DOMAIN_CONTEXT,
    DEFAULT_JUDGE_DOMAIN_NAME,
    REFUSAL_SCORER_NAME,
    SCORER_SEMANTICS_VERSIONS,
)
from anvil.runtime.models import ScorerConfig

if TYPE_CHECKING:
    # EvalReport lives in the sibling runner module. Referenced under
    # TYPE_CHECKING only so this module stays import-light (no mlflow /
    # openai pulled in just for a type hint). ``report_to_baseline``
    # reads attributes off the report, so it is duck-typed at runtime.
    from anvil.eval.runner import EvalReport


def compute_scorer_fingerprint(
    scorer_configs: list[ScorerConfig],
    *,
    judge_domain_name: str | None = None,
    judge_domain_context: str | None = None,
) -> str:
    """Compute a stable JSON fingerprint of the active scorer configs.

    Captures the full scorer specification (name, type, weight,
    check_function) so a weight change or check_function swap invalidates
    a cached baseline even when the scorer names are unchanged. The list
    is sorted by name for deterministic output.

    Also folds in :data:`anvil.eval.scorers.SCORER_SEMANTICS_VERSIONS`, because
    the config cannot see a change in what a scorer *means*. When
    ``retrieval_groundedness`` gained its applicability rule, every field above
    stayed byte-identical while the number it produced stopped being the same
    measurement — so a baseline from before the change would have remained
    "compatible" and gone on being the bar the loop chased. Only scorers whose
    semantics have been versioned carry the key, so bumping one does not
    invalidate baselines for configs that do not use it.

    ``judge_domain_name`` / ``judge_domain_context`` are folded into the refusal
    scorer's entry for the same reason. They are the judge's own description of
    the domain, and they are config rather than code, so a customized domain
    context changes the verdicts that scorer returns while every field above
    stays byte-identical. They are recorded only when set: ``None`` means the
    shipped default is in use, so baselines cached before these keys existed
    remain compatible.

    Storing the fingerprint in :class:`CachedBaseline` closes the
    comparability hole where a cached uniform-weight baseline stayed
    "compatible" after weights changed — the loop would then compare a
    new weighted aggregate against an old uniform-weight aggregate and
    make an invalid frontier decision.
    """
    specs: list[dict[str, Any]] = []
    for c in scorer_configs:
        spec: dict[str, Any] = {
            "name": c.name,
            "type": c.type,
            "weight": c.weight,
            "check_function": c.check_function,
        }
        semantics = SCORER_SEMANTICS_VERSIONS.get(c.name)
        if semantics is not None:
            spec["semantics"] = semantics
        if c.name == REFUSAL_SCORER_NAME:
            # Resolve through the same ``or`` fallback ``build_scorers`` uses, then
            # record only what actually differs from the shipped default. Testing
            # ``is not None`` instead would make writing the default text into
            # harness/config.yaml -- a no-op edit that renders a byte-identical
            # prompt -- invalidate the baseline and abort the next round after it
            # had already paid for an optimizer session and a full eval. It would
            # also disagree with ``build_scorers`` on the empty string.
            effective_name = judge_domain_name or DEFAULT_JUDGE_DOMAIN_NAME
            effective_context = judge_domain_context or DEFAULT_JUDGE_DOMAIN_CONTEXT
            if effective_name != DEFAULT_JUDGE_DOMAIN_NAME:
                spec["judge_domain_name"] = effective_name
            if effective_context != DEFAULT_JUDGE_DOMAIN_CONTEXT:
                spec["judge_domain_context"] = effective_context
        specs.append(spec)
    specs.sort(key=lambda s: str(s["name"]))
    return json.dumps(specs, sort_keys=True)


def compute_dataset_fingerprint(
    kb_dir: Path | str,
    golden_set_path: Path | str,
) -> str:
    """Content fingerprint of the domain an eval measured.

    The scorer fingerprint captures how the agent is graded. This captures *what
    it is graded on*, which nothing recorded until the domain became a
    parameter. Without it a baseline cached from one domain compares as
    perfectly compatible with a round evaluated against another: the mode,
    scorer names, endpoints and scorer fingerprint are all identical, and only
    the questions changed. Fifty rounds would then be kept or reverted against a
    bar measured on different data, and the highest-stakes artifact --
    ``eval/runs/finalized.json``, single-use and write-once -- would record that
    with nothing on disk to show it.

    Content-based rather than path-based, and so machine-independent: two
    checkouts agree, while editing a single golden row or knowledge-base
    document correctly invalidates the baseline, because it changed what is
    being measured.

    Returns ``""`` when either path is missing, which keeps the field's
    "absent means unchecked" contract rather than inventing a fingerprint for a
    domain that could not be read.
    """
    kb_path, golden_path = Path(kb_dir), Path(golden_set_path)
    if not kb_path.is_dir() or not golden_path.is_file():
        return ""
    digest = hashlib.sha256()
    digest.update(golden_path.read_bytes())
    # Sorted by name so filesystem ordering cannot change the fingerprint.
    for doc in sorted(kb_path.glob("*.md")):
        digest.update(b"\0")
        digest.update(doc.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(doc.read_bytes())
    return f"sha256:{digest.hexdigest()[:32]}"


def dataset_incomparability_reason(
    cached: CachedBaseline,
    *,
    dataset_fingerprint: str = "",
) -> str:
    """Why ``cached`` was measured on a different domain, or ``""``.

    Absent on either side means unchecked, exactly as
    :func:`scorer_incomparability_reason` treats an absent scorer fingerprint:
    every baseline written before this field existed stays usable, and no
    live re-run is forced by adding it.
    """
    if not cached.dataset_fingerprint or not dataset_fingerprint:
        return ""
    if cached.dataset_fingerprint != dataset_fingerprint:
        return (
            "the cached baseline was measured on a different dataset "
            f"({cached.dataset_fingerprint} vs {dataset_fingerprint}) — its aggregate "
            "answers different questions, so comparing them decides keep/revert on "
            "evidence about another domain"
        )
    return ""


@dataclass(frozen=True)
class CachedBaseline:
    scaffold_commit_sha: str
    evaluated_at: str
    mode: str
    scorers: list[str]
    runtime_endpoint: str
    judge_endpoint: str
    aggregate: float
    per_judge: dict[str, float] = field(default_factory=dict)
    per_bucket: dict[str, dict[str, float]] = field(default_factory=dict)
    n_examples: int = 0
    mlflow_run_id: str | None = None
    # JSON fingerprint of the scorer configs that produced this baseline
    # (see :func:`compute_scorer_fingerprint`). Empty on baselines written
    # before this field existed — :func:`is_compatible` treats an empty
    # fingerprint on either side as "not checked" for backward compat.
    scorer_fingerprint: str = ""
    # Content fingerprint of the kb + golden set this baseline was measured on
    # (see :func:`compute_dataset_fingerprint`). Empty on baselines written
    # before the domain was a parameter; empty on either side means unchecked.
    dataset_fingerprint: str = ""
    cost_metrics: dict[str, float] = field(default_factory=dict)
    # How many of ``n_examples`` were never assessed. The gate compares every
    # round against this aggregate, so a baseline measured on two of eight rows
    # is a bar the loop will chase for its whole run -- and post-exclusion such
    # a baseline looks *better*, not broken. Recording the count is what lets a
    # reader tell the difference later.
    n_errors: int = 0
    # Rows that never reached the result frame. ``n_examples`` counts SURVIVORS,
    # so without this a baseline measured on two of eight rows reads
    # ``{"n_examples": 2, "n_errors": 0}`` -- perfectly clean -- and the gate
    # chases that two-row bar for the whole 50+-round run with nothing on disk
    # recording why. Same argument that put ``n_errors`` here.
    n_dropped_rows: int = 0
    # Per-row scores, ``example_id`` -> {scorer: value}. What makes the paired
    # comparison in :mod:`anvil.eval.significance` possible: the aggregate alone
    # cannot say whether a delta is a real gain or the ~0.15 of judge noise two
    # identical runs produced. Empty on baselines written before this field
    # existed, and the gate treats empty as "cannot run the paired test" -- a
    # stated reason, not a silent fall-through to promoting on any delta.
    per_row: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "scaffold_commit_sha": self.scaffold_commit_sha,
            "evaluated_at": self.evaluated_at,
            "mode": self.mode,
            "scorers": list(self.scorers),
            "runtime_endpoint": self.runtime_endpoint,
            "judge_endpoint": self.judge_endpoint,
            "aggregate": self.aggregate,
            "per_judge": dict(self.per_judge),
            "per_bucket": {k: dict(v) for k, v in self.per_bucket.items()},
            "n_examples": self.n_examples,
            "mlflow_run_id": self.mlflow_run_id,
            "scorer_fingerprint": self.scorer_fingerprint,
            "dataset_fingerprint": self.dataset_fingerprint,
        }
        # Additive: keep the historical on-disk schema byte-for-byte identical
        # for a clean baseline, which is every baseline written before the
        # failure/error split existed.
        if self.n_errors:
            payload["n_errors"] = self.n_errors
        if self.n_dropped_rows:
            payload["n_dropped_rows"] = self.n_dropped_rows
        # Keep the historical on-disk schema byte-for-byte compatible for
        # baselines created before cost tracking, while retaining metrics
        # whenever the eval report supplies them.
        if self.cost_metrics:
            payload["cost_metrics"] = dict(self.cost_metrics)
        # Also additive, and for the same reason: a baseline with no per-row
        # scores must serialize exactly as it did before the field existed, so
        # regenerating is a choice rather than a forced migration.
        if self.per_row:
            payload["per_row"] = {k: dict(v) for k, v in self.per_row.items()}
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CachedBaseline:
        return cls(
            scaffold_commit_sha=raw["scaffold_commit_sha"],
            evaluated_at=raw["evaluated_at"],
            mode=raw["mode"],
            scorers=list(raw["scorers"]),
            runtime_endpoint=raw["runtime_endpoint"],
            judge_endpoint=raw["judge_endpoint"],
            aggregate=float(raw["aggregate"]),
            per_judge=dict(raw.get("per_judge", {})),
            per_bucket={k: dict(v) for k, v in raw.get("per_bucket", {}).items()},
            n_examples=int(raw.get("n_examples", 0)),
            mlflow_run_id=raw.get("mlflow_run_id"),
            scorer_fingerprint=raw.get("scorer_fingerprint", ""),
            dataset_fingerprint=raw.get("dataset_fingerprint", ""),
            cost_metrics={k: float(v) for k, v in raw.get("cost_metrics", {}).items()},
            n_errors=int(raw.get("n_errors", 0)),
            n_dropped_rows=int(raw.get("n_dropped_rows", 0)),
            per_row={
                str(k): {str(n): float(v) for n, v in scores.items()}
                for k, scores in raw.get("per_row", {}).items()
            },
        )


def baseline_path(repo_root: Path | str) -> Path:
    return Path(repo_root) / "eval" / "runs" / "baseline.json"


def load_baseline(repo_root: Path | str) -> CachedBaseline | None:
    path = baseline_path(repo_root)
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return CachedBaseline.from_dict(raw)


def save_baseline(repo_root: Path | str, baseline: CachedBaseline) -> Path:
    path = baseline_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def parent_path(repo_root: Path | str) -> Path:
    return Path(repo_root) / "eval" / "runs" / "parent.json"


def load_parent(repo_root: Path | str) -> CachedBaseline | None:
    """Load the current parent scaffold's most recent eval draw, or ``None``.

    ``parent.json`` is the paired test's comparator once the loop has KEPT at
    least one round: the kept candidate becomes the parent of everything that
    follows, so its per-row scores — not the frozen baseline's — are what the
    next candidate must beat row by row. Absent means no KEEP has happened
    (or the baseline was just regenerated), and the caller falls back to the
    frozen baseline, which is the correct comparator for exactly that case.

    The schema is :class:`CachedBaseline` unchanged, so
    :func:`report_to_baseline` writes it and the ``*_incomparability_reason``
    helpers check it with no special cases.
    """
    path = parent_path(repo_root)
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return CachedBaseline.from_dict(raw)


def save_parent(repo_root: Path | str, parent: CachedBaseline) -> Path:
    """Persist ``parent`` to ``eval/runs/parent.json``. Returns the path.

    Called by the round loop on KEEP and nowhere else: a KEEP replaces the
    parent, so the file is replaced wholesale — a superseded parent's draw is
    never consulted again. A REVERT writes nothing, because the parent did
    not change; reusing the existing draw across a revert streak is correct
    by construction, not the frozen-control bug this file exists to fix.

    One honest limitation, recorded in docs/decisions.md: the parent's draw
    comes from an earlier judge session, so the paired test cancels row
    difficulty but not cross-session judge drift. The contemporaneous
    alternative (re-evaluate the parent every round) was rejected — it
    doubles eval spend to control a drift the frontier gate already tolerates
    by comparing best-so-far scores across sessions.
    """
    path = parent_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(parent.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def scorer_incomparability_reason(
    cached: CachedBaseline,
    *,
    scorers: list[str],
    scorer_fingerprint: str = "",
) -> str:
    """Why ``cached``'s scorer configuration cannot be compared, or ``""``.

    The single definition of scorer comparability, because there used to be two:
    :func:`is_compatible` and an inline copy in ``loop/round.py``. They agreed
    until one was fixed, at which point the gate — the copy that actually decides
    keep/revert — kept the old behaviour. Same failure mode
    :mod:`anvil.eval.judgeability` exists to prevent one level up.

    When both sides carry a fingerprint, they must match: a weight or
    ``check_function`` change gives the cached aggregate a different meaning even
    though the scorer names are identical.

    A missing fingerprint on either side is normally waved through, for baselines
    written before the field existed. The exception is a scorer whose *semantics*
    have been versioned (:data:`anvil.eval.scorers.SCORER_SEMANTICS_VERSIONS`):
    there the exemption and the version bump contradict each other. The bump says
    this scorer's meaning demonstrably changed; an absent fingerprint says we
    cannot tell which meaning the baseline was measured under. That is the one
    case where waving it through is knowably wrong.

    Not hypothetical. The shipped ``eval/runs/baseline.json`` predates
    fingerprinting entirely, and it is exactly the baseline that needed
    invalidating when ``retrieval_groundedness`` gained its applicability rule —
    its ``per_bucket`` still records ``out_of_scope: {retrieval_groundedness:
    0.0}``, a bucket that now has no groundedness value at all. Without this the
    version bump protected every baseline except the only one on disk.
    """
    if cached.scorer_fingerprint and scorer_fingerprint:
        if cached.scorer_fingerprint != scorer_fingerprint:
            return (
                "scorer configuration has changed since the baseline was cached "
                "(fingerprint mismatch)"
            )
        return ""
    if not cached.scorer_fingerprint:
        versioned = sorted(name for name in scorers if name in SCORER_SEMANTICS_VERSIONS)
        if versioned:
            return (
                f"the cached baseline carries no scorer fingerprint, so it cannot be "
                f"shown to have been measured under the current meaning of "
                f"{', '.join(versioned)} — whose semantics have changed since"
            )
    return ""


def is_compatible(
    cached: CachedBaseline,
    *,
    mode: str,
    scorers: list[str],
    runtime_endpoint: str,
    judge_endpoint: str,
    scorer_fingerprint: str = "",
    dataset_fingerprint: str = "",
) -> bool:
    """Return True if ``cached`` is comparable with the requesting context.

    Mode, scorer names and both endpoints must match exactly; the scorer
    configuration is delegated to :func:`scorer_incomparability_reason` and the
    domain to :func:`dataset_incomparability_reason`.
    """
    if (
        cached.mode != mode
        or list(cached.scorers) != list(scorers)
        or cached.runtime_endpoint != runtime_endpoint
        or cached.judge_endpoint != judge_endpoint
    ):
        return False
    if dataset_incomparability_reason(cached, dataset_fingerprint=dataset_fingerprint):
        return False
    return not scorer_incomparability_reason(
        cached, scorers=scorers, scorer_fingerprint=scorer_fingerprint
    )


def report_to_baseline(
    report: EvalReport,
    *,
    scaffold_commit_sha: str,
    runtime_endpoint: str,
    judge_endpoint: str,
) -> CachedBaseline:
    """Convert an :class:`EvalReport` into a storable :class:`CachedBaseline`.

    ``evaluate_branch`` returns an ``EvalReport`` — the eval runner's
    own schema (``n_rows`` / ``run_id`` / ``failures`` / ``trace_ids``).
    The loop's keep/revert gate, by contrast, reads a ``CachedBaseline``
    (``n_examples`` / ``mlflow_run_id``) from
    ``eval/runs/baseline.json``. This function bridges the two schemas
    so a fresh scaffold can produce the baseline the gate needs without
    re-running the eval every round.

    The two schemas intentionally diverge on two field names:
    ``EvalReport.n_rows`` → ``CachedBaseline.n_examples`` and
    ``EvalReport.run_id`` → ``CachedBaseline.mlflow_run_id``. The
    eval-only fields (``failures`` / ``experiment_id`` / ``trace_ids``)
    are dropped — the cache header only carries what
    :func:`is_compatible` and :func:`load_baseline` consume.

    The three fields the eval does not know — ``scaffold_commit_sha``
    (git), ``runtime_endpoint`` and ``judge_endpoint``
    (``harness/config.yaml``) — are passed in by the caller, keeping
    this plane git-agnostic and config-source-agnostic (see the module
    docstring: cross-plane knowledge is forbidden here).
    """
    return CachedBaseline(
        scaffold_commit_sha=scaffold_commit_sha,
        evaluated_at=report.evaluated_at,
        mode=report.mode,
        scorers=list(report.scorers),
        runtime_endpoint=runtime_endpoint,
        judge_endpoint=judge_endpoint,
        aggregate=report.aggregate,
        per_judge=dict(report.per_judge),
        per_bucket={k: dict(v) for k, v in report.per_bucket.items()},
        n_examples=report.n_rows,
        mlflow_run_id=report.run_id,
        scorer_fingerprint=getattr(report, "scorer_fingerprint", ""),
        dataset_fingerprint=getattr(report, "dataset_fingerprint", ""),
        cost_metrics=dict(getattr(report, "cost_metrics", {})),
        n_errors=getattr(report, "n_errors", 0),
        n_dropped_rows=getattr(report, "n_dropped_rows", 0),
        per_row={k: dict(v) for k, v in getattr(report, "per_row", {}).items()},
    )
