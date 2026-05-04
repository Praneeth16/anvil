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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
        return {
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
        }

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


def is_compatible(
    cached: CachedBaseline,
    *,
    mode: str,
    scorers: list[str],
    runtime_endpoint: str,
    judge_endpoint: str,
) -> bool:
    """Return True if ``cached`` is comparable with the requesting context."""
    return (
        cached.mode == mode
        and list(cached.scorers) == list(scorers)
        and cached.runtime_endpoint == runtime_endpoint
        and cached.judge_endpoint == judge_endpoint
    )
