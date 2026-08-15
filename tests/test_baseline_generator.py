"""Tests for the EvalReport → CachedBaseline baseline generator.

Covers the three contracts the accept/reject gate depends on:

* :func:`report_to_baseline` maps every ``EvalReport`` field onto the
  ``CachedBaseline`` schema (including the two deliberate renames:
  ``n_rows`` → ``n_examples`` and ``run_id`` → ``mlflow_run_id``).
* The generated file is loadable by :func:`anvil.eval.load_baseline`
  — the exact reader ``round.py`` calls before every round.
* Every required ``CachedBaseline`` field is populated.

The CLI integration test exercises ``scripts/make_baseline.py``
end-to-end with ``evaluate_branch`` monkeypatched, so **no LLM and no
git** are invoked.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from anvil.eval.cache import (
    CachedBaseline,
    load_baseline,
    report_to_baseline,
    save_baseline,
)
from anvil.eval.runner import EvalReport

REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_FIELDS = [
    "scaffold_commit_sha",
    "evaluated_at",
    "mode",
    "scorers",
    "runtime_endpoint",
    "judge_endpoint",
    "aggregate",
    "per_judge",
    "per_bucket",
    "n_examples",
    "mlflow_run_id",
]

_SHA = "a" * 40
_RUNTIME = "databricks-claude-sonnet-4-6"
_JUDGE = "databricks-claude-sonnet-4-6"


def _fake_report() -> EvalReport:
    """A mock eval result — no LLM, no mlflow, just the dataclass."""
    return EvalReport(
        aggregate=0.75,
        per_judge={
            "correctness": 0.5,
            "retrieval_groundedness": 0.875,
            "refusal_appropriateness": 1.0,
        },
        per_bucket={
            "direct": {
                "correctness": 0.5,
                "retrieval_groundedness": 1.0,
                "refusal_appropriateness": 1.0,
            },
            "out_of_scope": {
                "correctness": 0.0,
                "retrieval_groundedness": 0.0,
                "refusal_appropriateness": 1.0,
            },
        },
        failures=[],
        run_id="abc123def456",
        experiment_id="exp_1",
        n_rows=8,
        mode="quick",
        scorers=["correctness", "retrieval_groundedness", "refusal_appropriateness"],
        evaluated_at="2026-08-16T12:00:00+00:00",
        trace_ids=["t0", "t1"],
    )


def _baseline_from_fake() -> CachedBaseline:
    return report_to_baseline(
        _fake_report(),
        scaffold_commit_sha=_SHA,
        runtime_endpoint=_RUNTIME,
        judge_endpoint=_JUDGE,
    )


# ---------------------------------------------------------------------------
# 1. Field mapping (EvalReport → CachedBaseline) using a mock eval result.
# ---------------------------------------------------------------------------


def test_report_to_baseline_maps_all_fields() -> None:
    report = _fake_report()
    baseline = report_to_baseline(
        report,
        scaffold_commit_sha=_SHA,
        runtime_endpoint=_RUNTIME,
        judge_endpoint=_JUDGE,
    )

    # Direct-copy fields.
    assert baseline.evaluated_at == report.evaluated_at
    assert baseline.mode == report.mode
    assert baseline.scorers == list(report.scorers)
    assert baseline.aggregate == report.aggregate
    assert baseline.per_judge == report.per_judge
    assert baseline.per_bucket == report.per_bucket

    # Fields sourced from the caller (git + config), not the report.
    assert baseline.scaffold_commit_sha == _SHA
    assert baseline.runtime_endpoint == _RUNTIME
    assert baseline.judge_endpoint == _JUDGE

    # The conversion MUST drop the eval-only fields — the cache header
    # only carries what is_compatible() / load_baseline() consume.
    dumped = baseline.to_dict()
    assert set(dumped.keys()) == set(REQUIRED_FIELDS)
    for dropped in ("failures", "experiment_id", "trace_ids", "n_rows", "run_id"):
        assert dropped not in dumped


def test_report_to_baseline_renames_n_rows_and_run_id() -> None:
    """The two schemas intentionally diverge on these two field names."""
    report = _fake_report()
    baseline = report_to_baseline(
        report,
        scaffold_commit_sha=_SHA,
        runtime_endpoint=_RUNTIME,
        judge_endpoint=_JUDGE,
    )

    assert baseline.n_examples == report.n_rows
    assert baseline.n_examples != 0  # not the dataclass default
    assert baseline.mlflow_run_id == report.run_id
    assert baseline.mlflow_run_id is not None


def test_report_to_baseline_copies_not_aliases() -> None:
    """Mutating the report's containers must not leak into the baseline."""
    report = _fake_report()
    baseline = report_to_baseline(
        report,
        scaffold_commit_sha=_SHA,
        runtime_endpoint=_RUNTIME,
        judge_endpoint=_JUDGE,
    )

    report.scorers.append("safety")
    report.per_judge["correctness"] = 0.0
    report.per_bucket["direct"]["correctness"] = 0.0

    assert "safety" not in baseline.scorers
    assert baseline.per_judge["correctness"] == 0.5
    assert baseline.per_bucket["direct"]["correctness"] == 0.5


# ---------------------------------------------------------------------------
# 2. load_baseline() can load the generated file (the round.py reader).
# ---------------------------------------------------------------------------


def test_generated_baseline_loads_via_load_baseline(tmp_path: Path) -> None:
    baseline = _baseline_from_fake()
    save_baseline(tmp_path, baseline)

    loaded = load_baseline(tmp_path)
    assert loaded is not None
    assert loaded.scaffold_commit_sha == baseline.scaffold_commit_sha
    assert loaded.evaluated_at == baseline.evaluated_at
    assert loaded.mode == baseline.mode
    assert loaded.scorers == baseline.scorers
    assert loaded.runtime_endpoint == baseline.runtime_endpoint
    assert loaded.judge_endpoint == baseline.judge_endpoint
    assert loaded.aggregate == baseline.aggregate
    assert loaded.per_judge == baseline.per_judge
    assert loaded.per_bucket == baseline.per_bucket
    assert loaded.n_examples == baseline.n_examples
    assert loaded.mlflow_run_id == baseline.mlflow_run_id


def test_load_baseline_returns_none_when_file_missing(tmp_path: Path) -> None:
    assert load_baseline(tmp_path) is None


# ---------------------------------------------------------------------------
# 3. All required CachedBaseline fields are populated.
# ---------------------------------------------------------------------------


def test_generated_baseline_has_all_required_fields() -> None:
    baseline = _baseline_from_fake()
    dumped = baseline.to_dict()

    for field in REQUIRED_FIELDS:
        assert field in dumped, f"missing required field: {field}"

    assert dumped["scaffold_commit_sha"]
    assert dumped["evaluated_at"]
    assert dumped["mode"]
    assert dumped["scorers"]
    assert dumped["runtime_endpoint"]
    assert dumped["judge_endpoint"]
    assert dumped["aggregate"] == pytest.approx(0.75)
    assert dumped["per_judge"]
    assert dumped["per_bucket"]
    assert dumped["n_examples"] == 8
    assert dumped["mlflow_run_id"]


def test_generated_baseline_json_round_trips(tmp_path: Path) -> None:
    """The on-disk JSON (as make_baseline.py writes it) re-parses cleanly."""
    baseline = _baseline_from_fake()
    path = tmp_path / "eval" / "runs" / "baseline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline.to_dict(), indent=2) + "\n", encoding="utf-8")

    raw = json.loads(path.read_text(encoding="utf-8"))
    reborn = CachedBaseline.from_dict(raw)
    assert reborn == baseline


# ---------------------------------------------------------------------------
# 4. CLI integration: scripts/make_baseline.py writes a loadable file.
#    evaluate_branch + git are monkeypatched — no LLM, no git.
# ---------------------------------------------------------------------------


_MIN_CONFIG_YAML = """\
runtime_endpoint: databricks-claude-sonnet-4-6
optimizer_endpoint: databricks-claude-opus-4-7
judge_endpoint: databricks-claude-sonnet-4-6
experiments:
  runtime: "/Shared/anvil-runtime"
  eval: "/Shared/anvil-eval"
  optimizer: "/Shared/anvil-optimizer"
"""


def _import_make_baseline():
    """Load ``scripts/make_baseline.py`` as a module without touching sys.path.

    ``scripts/`` is not a package, so the script is loaded directly from
    its file via :mod:`importlib`. Each call returns a fresh module
    object, which keeps per-test ``monkeypatch`` of ``evaluate_branch``
    / ``_git_head_sha`` fully isolated.
    """
    path = REPO_ROOT / "scripts" / "make_baseline.py"
    spec = importlib.util.spec_from_file_location("make_baseline", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # runs the script's top-level imports
    return mod


def test_make_baseline_cli_writes_loadable_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_baseline = _import_make_baseline()

    # Minimal harness/config.yaml so _load_endpoints validates via RuntimeYAML.
    (tmp_path / "harness").mkdir(parents=True)
    (tmp_path / "harness" / "config.yaml").write_text(_MIN_CONFIG_YAML, encoding="utf-8")
    (tmp_path / "scaffold").mkdir()

    # Mock the eval (no LLM) and the git sha lookup (no git repo).
    monkeypatch.setattr(make_baseline, "evaluate_branch", lambda **_kw: _fake_report())
    monkeypatch.setattr(make_baseline, "_git_head_sha", lambda _root: _SHA)

    out_path = tmp_path / "eval" / "runs" / "baseline.json"
    rc = make_baseline.main(["--scaffold", str(tmp_path / "scaffold"), "--out", str(out_path)])
    assert rc == 0
    assert out_path.is_file()

    # load_baseline (the round.py reader) must consume the script's output.
    loaded = load_baseline(tmp_path)
    assert loaded is not None
    assert loaded.scaffold_commit_sha == _SHA
    assert loaded.runtime_endpoint == "databricks-claude-sonnet-4-6"
    assert loaded.judge_endpoint == "databricks-claude-sonnet-4-6"
    assert loaded.n_examples == 8
    assert loaded.mlflow_run_id == "abc123def456"
    assert loaded.mode == "quick"
    assert loaded.aggregate == pytest.approx(0.75)


def test_make_baseline_help_lists_options(
    capsys: pytest.CaptureFixture,
) -> None:
    make_baseline = _import_make_baseline()

    with pytest.raises(SystemExit) as exc:
        make_baseline._arg_parser().parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--mode" in out
    assert "--out" in out
    assert "--scaffold" in out
    assert "--profile" in out
    assert "--include-safety" in out
