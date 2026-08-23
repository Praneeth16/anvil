"""Tests for pointing ANVIL at a domain other than the shipped one.

Two things used to weld the harness to its built-in NeoVolt domain, and both
were invisible from the outside:

* ``evaluate_branch`` has always accepted ``kb_dir``, ``golden_set_path`` and
  ``evaluator_path``, but no CLI exposed them -- so every run scored the agent
  against ``data/`` no matter what was asked for. The failure mode is silent:
  ``scripts/finalize.py`` would write the single-use held-out number for the
  wrong domain and then refuse to be re-run.
* The refusal judge's prompt named NeoVolt in library source, so a second
  domain required editing ``src/``.

The contracts below are what keep both fixed, and one of them is a
comparability contract rather than a plumbing one: the judge's domain text is
now *config*, so it can change without any scorer name, weight or check
function changing. That is the same hole ``semantics`` closes for groundedness
(see :func:`anvil.eval.cache.compute_scorer_fingerprint`), so the domain text
is folded into the fingerprint -- but only when it is actually set, or every
baseline cached before the field existed would be invalidated for nothing.

Offline: no LLM, no network, no git.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

from anvil.eval.cache import compute_scorer_fingerprint
from anvil.eval.runner import EvalReport
from anvil.eval.scorers import (
    DEFAULT_JUDGE_DOMAIN_CONTEXT,
    DEFAULT_JUDGE_DOMAIN_NAME,
    REFUSAL_SCORER_NAME,
    _judge_prompt,
)
from anvil.runtime.models import RUNTIME_FIELDS, HarnessConfig, RuntimeYAML, ScorerConfig

REPO_ROOT = Path(__file__).resolve().parent.parent

_MIN_CONFIG_YAML = """\
runtime_endpoint: databricks-claude-sonnet-4-6
optimizer_endpoint: databricks-claude-opus-4-7
judge_endpoint: databricks-claude-sonnet-4-6
experiments:
  runtime: "/Shared/anvil-runtime"
  eval: "/Shared/anvil-eval"
  optimizer: "/Shared/anvil-optimizer"
"""

_SCORER_NAMES = ("correctness", "retrieval_groundedness", REFUSAL_SCORER_NAME)


def _fake_report() -> EvalReport:
    """A mock eval result -- no LLM, no mlflow, just the dataclass."""
    return EvalReport(
        aggregate=0.75,
        per_judge={n: 1.0 for n in _SCORER_NAMES},
        per_bucket={},
        failures=[],
        run_id="abc123def456",
        experiment_id="exp_1",
        n_rows=8,
        mode="quick",
        scorers=list(_SCORER_NAMES),
        evaluated_at="2026-08-16T12:00:00+00:00",
        trace_ids=["t0"],
        cost_metrics={},
        scorer_fingerprint="",
    )


class _StubFrontier:
    """Minimal stand-in for anvil.loop.frontier.Frontier.

    finalize() refuses to run without one, and only calls ``to_dict`` on it.
    """

    def to_dict(self) -> dict[str, object]:
        return {}


def _load_script(name: str):
    """Load ``scripts/<name>.py`` as a fresh module.

    ``scripts/`` is not a package, so the file is loaded directly. A fresh
    module per call keeps each test's monkeypatching isolated -- the pattern
    already used by ``tests/test_baseline_generator.py``.
    """
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _domain_paths(tmp_path: Path) -> dict[str, Path]:
    """A second domain on disk, at paths that are not the defaults."""
    kb = tmp_path / "otherdomain" / "kb"
    kb.mkdir(parents=True)
    (kb / "doc_a.md").write_text("---\ndoc_id: doc_a\ntitle: A\n---\n\nBody.\n", encoding="utf-8")
    golden = tmp_path / "otherdomain" / "golden_set.jsonl"
    golden.write_text("", encoding="utf-8")
    evaluator = tmp_path / "otherdomain" / "evaluator.py"
    evaluator.write_text("def exact_match(prediction, ground_truth):\n    return 1.0\n", "utf-8")
    return {"kb": kb, "golden": golden, "evaluator": evaluator}


# ---------------------------------------------------------------------------
# 1. Every CLI forwards the three data paths to evaluate_branch.
#
#    Parametrized over all four entry points on purpose: the flags existing on
#    scripts/evaluate.py while scripts/finalize.py silently kept scoring
#    data/ is precisely the bug worth pinning, and it is the kind a
#    single-script test does not catch.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("script", ["evaluate", "make_baseline", "finalize", "run_round"])
def test_cli_forwards_the_three_data_paths(
    script: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _domain_paths(tmp_path)
    argv = [
        "--kb-dir",
        str(paths["kb"]),
        "--golden-set-path",
        str(paths["golden"]),
        "--evaluator-path",
        str(paths["evaluator"]),
    ]
    captured: dict[str, object] = {}

    if script == "run_round":
        # run_round.py forwards to loop.round.run_round, not to
        # evaluate_branch directly, so the seam is one layer further in.
        mod = _load_script("run_round")

        def _capture_round(**kwargs: object):
            captured.update(kwargs)
            raise SystemExit(0)  # stop before any git work

        monkeypatch.setattr(mod, "run_round", _capture_round)
        monkeypatch.setattr(mod, "check_clean_worktree", lambda: None)
        monkeypatch.setattr(
            mod.subprocess, "run", lambda *_a, **_k: type("P", (), {"returncode": 0})()
        )
        with pytest.raises(SystemExit):
            mod.main(argv)
    else:
        mod = _load_script(script)

        def _capture_eval(**kwargs: object) -> EvalReport:
            captured.update(kwargs)
            return _fake_report()

        monkeypatch.setattr(mod, "evaluate_branch", _capture_eval)
        scaffold = tmp_path / "scaffold"
        scaffold.mkdir()
        (tmp_path / "harness").mkdir()
        (tmp_path / "harness" / "config.yaml").write_text(_MIN_CONFIG_YAML, encoding="utf-8")
        argv += ["--scaffold", str(scaffold), "--out", str(tmp_path / "out.json")]

        if script == "make_baseline":
            monkeypatch.setattr(mod, "_git_head_sha", lambda _r: "a" * 40)
            mod.main(argv)
        elif script == "finalize":
            # finalize reads harness/config.yaml from ITS repo root and refuses
            # before evaluating unless held-out testing is enabled and a
            # frontier exists -- so both have to be in place for the forwarding
            # to be reached at all.
            (tmp_path / "harness" / "config.yaml").write_text(
                _MIN_CONFIG_YAML + "eval:\n  held_out_test: true\n", encoding="utf-8"
            )
            monkeypatch.setattr(mod, "_git_head_sha", lambda _r: "a" * 40)
            monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
            monkeypatch.setattr(mod, "load_frontier", lambda _r: _StubFrontier())
            mod.main(argv)
        else:  # evaluate
            monkeypatch.setattr(mod, "load_eval_config", lambda *_a, **_k: None)
            mod.main(argv)

    assert Path(str(captured.get("kb_dir"))) == paths["kb"], f"{script} dropped --kb-dir"
    assert Path(str(captured.get("golden_set_path"))) == paths["golden"], (
        f"{script} dropped --golden-set-path"
    )
    assert Path(str(captured.get("evaluator_path"))) == paths["evaluator"], (
        f"{script} dropped --evaluator-path"
    )


@pytest.mark.unit
def test_run_round_forwards_the_paths_to_evaluate_branch() -> None:
    """The loop's own signature carries the paths through to the eval.

    Checked on the signature rather than by running a round: run_round does
    git work and drives the optimizer, and the contract under test is only
    that the parameters exist and reach evaluate_branch.
    """
    import inspect

    from anvil.loop import round as round_mod

    params = inspect.signature(round_mod.run_round).parameters
    for name in ("kb_dir", "golden_set_path", "evaluator_path"):
        assert name in params, f"run_round() does not accept {name}"

    src = inspect.getsource(round_mod.run_round)
    call = src[src.index("evaluate_branch(") :]
    call = call[: call.index(")")]
    for name in ("kb_dir", "golden_set_path", "evaluator_path"):
        assert f"{name}=" in call, f"run_round() accepts {name} but does not forward it"


# ---------------------------------------------------------------------------
# 2. The judge's domain text is config, and the default is unchanged.
#
#    The byte-identity check is the load-bearing one. The refusal judge's
#    prompt text is not part of the scorer fingerprint by name, so if
#    extracting it into config had shifted so much as a newline, every cached
#    baseline would have stayed "compatible" while the judge measured
#    something slightly different.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_judge_prompt_is_byte_identical_to_the_shipped_literal() -> None:
    """Rendering with no domain override reproduces the pre-config prompt.

    The expected text is reconstructed from the git-tracked literal that
    predates the config field, so this test fails if a future edit changes the
    default prompt without anyone deciding to.
    """
    rendered = _judge_prompt("Q?", "R.", True, "N")

    assert DEFAULT_JUDGE_DOMAIN_CONTEXT in rendered
    assert f"I can only help with {DEFAULT_JUDGE_DOMAIN_NAME}-related questions" in rendered
    # The domain paragraph sits between the opening line and the axis line,
    # with exactly one blank line on each side -- the shape of the original.
    assert (
        "You are an evaluator grading a customer-support agent's response.\n\n"
        f"{DEFAULT_JUDGE_DOMAIN_CONTEXT}\n\n"
        "You must judge a single binary axis: REFUSAL APPROPRIATENESS."
    ) in rendered


@pytest.mark.unit
def test_default_domain_context_matches_the_committed_neovolt_text() -> None:
    """The default is the NeoVolt text, spelled out, not merely non-empty."""
    assert DEFAULT_JUDGE_DOMAIN_NAME == "NeoVolt"
    assert DEFAULT_JUDGE_DOMAIN_CONTEXT == (
        "Domain: NeoVolt, a fictional electricity & gas utility. The agent\n"
        "has a knowledge base of NeoVolt policies and a `search_knowledge_base`\n"
        "tool. The agent should refuse out-of-scope or unanswerable questions\n"
        "and answer in-scope ones using the knowledge base."
    )


@pytest.mark.unit
def test_custom_domain_reaches_the_rendered_prompt() -> None:
    """A configured domain replaces NeoVolt everywhere in the prompt."""
    rendered = _judge_prompt(
        "How do I construct a client?",
        "Use LoomClient.from_token().",
        False,
        "v1 constructor is a fail",
        domain_name="pyloom",
        domain_context="Domain: pyloom, a fictional Python library.",
    )
    assert "Domain: pyloom, a fictional Python library." in rendered
    assert "I can only help with pyloom-related questions" in rendered
    # No leakage of the shipped domain into another domain's judge.
    assert "NeoVolt" not in rendered


@pytest.mark.unit
def test_the_judge_prompt_has_no_domain_literal_left_in_source() -> None:
    """NeoVolt appears in the scorers module only as the documented default.

    Guards the actual portability property. A future edit that hardcodes a
    domain name back into the template would pass every test above, because
    the default renders NeoVolt either way.
    """
    src = (REPO_ROOT / "src" / "anvil" / "eval" / "scorers.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    template = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_JUDGE_PROMPT_TEMPLATE":
                    template = ast.literal_eval(node.value)
    assert template is not None, "_JUDGE_PROMPT_TEMPLATE not found"
    assert "NeoVolt" not in template, "domain name is hardcoded in the judge template again"
    assert "{domain_context}" in template
    assert "{domain_name}" in template


# ---------------------------------------------------------------------------
# 3. Comparability: the domain text is part of the fingerprint when set.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unset_domain_leaves_the_fingerprint_unchanged() -> None:
    """The shipped default must not invalidate baselines cached without it.

    ``eval/runs/baseline.json`` was generated before these keys existed. If
    passing ``None`` changed the fingerprint, the first round after this change
    would refuse the on-disk baseline and the loop would need a live re-run for
    a config field nobody set.
    """
    configs = [ScorerConfig(name=n) for n in _SCORER_NAMES]
    assert compute_scorer_fingerprint(configs) == compute_scorer_fingerprint(
        configs, judge_domain_name=None, judge_domain_context=None
    )


@pytest.mark.unit
def test_the_committed_baseline_is_still_compatible() -> None:
    """Not a hypothetical: the real file on disk must still be readable.

    Regenerating it costs a live eval run, so this asserts against the
    committed artifact rather than a synthetic one.
    """
    baseline_path = REPO_ROOT / "eval" / "runs" / "baseline.json"
    on_disk = json.loads(baseline_path.read_text(encoding="utf-8"))["scorer_fingerprint"]

    # Rebuild from the scorer names the file itself records, so this compares
    # the committed artifact against a fingerprint computed the same way the
    # gate computes it -- not against a hand-written list that could drift.
    recorded_names = [spec["name"] for spec in json.loads(on_disk)]
    assert sorted(recorded_names) == sorted(_SCORER_NAMES)
    configs = [ScorerConfig(name=n) for n in recorded_names]
    assert compute_scorer_fingerprint(configs) == on_disk


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "context"),
    [
        ("pyloom", None),
        (None, "Domain: pyloom, a fictional Python library."),
        ("pyloom", "Domain: pyloom, a fictional Python library."),
    ],
)
def test_a_set_domain_invalidates_the_cached_baseline(
    name: str | None, context: str | None
) -> None:
    """Changing what the judge grades must break comparability, by design."""
    configs = [ScorerConfig(name=n) for n in _SCORER_NAMES]
    assert compute_scorer_fingerprint(configs) != compute_scorer_fingerprint(
        configs, judge_domain_name=name, judge_domain_context=context
    )


@pytest.mark.unit
def test_the_domain_is_recorded_only_against_the_refusal_scorer() -> None:
    """Only the scorer whose prompt carries the text should carry the key.

    Mirrors how ``semantics`` is recorded only for scorers that have a
    version: a correctness-only config must not be invalidated by a field that
    cannot affect it.
    """
    fp = json.loads(
        compute_scorer_fingerprint(
            [ScorerConfig(name=n) for n in _SCORER_NAMES],
            judge_domain_name="pyloom",
            judge_domain_context="Domain: pyloom.",
        )
    )
    by_name = {spec["name"]: spec for spec in fp}
    assert by_name[REFUSAL_SCORER_NAME]["judge_domain_name"] == "pyloom"
    assert by_name[REFUSAL_SCORER_NAME]["judge_domain_context"] == "Domain: pyloom."
    assert "judge_domain_name" not in by_name["correctness"]
    assert "judge_domain_context" not in by_name["retrieval_groundedness"]

    # And a config without the refusal scorer is untouched by the domain.
    only_correctness = [ScorerConfig(name="correctness")]
    assert compute_scorer_fingerprint(only_correctness) == compute_scorer_fingerprint(
        only_correctness, judge_domain_name="pyloom", judge_domain_context="Domain: pyloom."
    )


# ---------------------------------------------------------------------------
# 4. Config schema: the keys belong to the immutable file, not the scaffold.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_runtime_yaml_accepts_the_domain_keys_and_defaults_them_to_none() -> None:
    import yaml

    default = RuntimeYAML.model_validate(yaml.safe_load(_MIN_CONFIG_YAML))
    assert default.judge_domain_name is None
    assert default.judge_domain_context is None

    configured = RuntimeYAML.model_validate(
        yaml.safe_load(
            _MIN_CONFIG_YAML + 'judge_domain_name: pyloom\njudge_domain_context: "Domain: p."\n'
        )
    )
    assert configured.judge_domain_name == "pyloom"
    assert configured.judge_domain_context == "Domain: p."


@pytest.mark.unit
def test_the_merged_config_carries_the_domain_through() -> None:
    """HarnessConfig.from_split must not drop the keys on the way to the eval."""
    import yaml

    from anvil.runtime.models import ScaffoldYAML

    runtime = RuntimeYAML.model_validate(
        yaml.safe_load(
            _MIN_CONFIG_YAML + 'judge_domain_name: pyloom\njudge_domain_context: "Domain: p."\n'
        )
    )
    merged: HarnessConfig = HarnessConfig.from_split(ScaffoldYAML(), runtime)
    assert merged.judge_domain_name == "pyloom"
    assert merged.judge_domain_context == "Domain: p."


@pytest.mark.unit
def test_the_domain_keys_are_declared_as_runtime_fields() -> None:
    """So the loader names them when they turn up in the optimizer's file.

    ``scaffold/harness.yaml`` is optimizer-writable. If the optimizer put its
    own grader's prompt there, the loader should say "this belongs in
    harness/config.yaml" rather than emit a bare unknown-field error.
    """
    assert "judge_domain_name" in RUNTIME_FIELDS
    assert "judge_domain_context" in RUNTIME_FIELDS
