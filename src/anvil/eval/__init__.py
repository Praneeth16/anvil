"""ANVIL eval plane.

Knows how to run ``mlflow.genai.evaluate`` against a scaffold over
the golden set with the configured scorers (3 by default; Safety
opt-in via ``--include-safety``). Cached baselines live in
``eval/runs/baseline.json`` and are refreshed only on explicit
``--refresh-baseline``.

Cross-plane knowledge is forbidden: this plane does not know about
the optimizer or git.
"""

from anvil.eval.cache import (
    CachedBaseline,
    compute_scorer_fingerprint,
    is_compatible,
    load_baseline,
    report_to_baseline,
    save_baseline,
)
from anvil.eval.runner import EvalReport, evaluate_branch
from anvil.eval.scorers import build_scorers

__all__ = [
    "CachedBaseline",
    "EvalReport",
    "build_scorers",
    "compute_scorer_fingerprint",
    "evaluate_branch",
    "is_compatible",
    "load_baseline",
    "report_to_baseline",
    "save_baseline",
]
