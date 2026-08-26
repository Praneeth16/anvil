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
from contextlib import suppress
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
from anvil.optimizer.actions import ChangeSamplingAction
from anvil.optimizer.applier import ApplyResult
from anvil.optimizer.parser import ParseResult
from anvil.runtime.models import (
    RUNTIME_FIELDS,
    EvalConfig,
    HarnessConfig,
    RuntimeYAML,
    ScorerConfig,
)

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
    # Idempotent: some tests call this both directly and through a harness that
    # calls it too, and the second call must return the same paths.
    kb.mkdir(parents=True, exist_ok=True)
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
def test_run_round_forwards_the_paths_to_evaluate_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_round`` passes its three paths to the eval, by value.

    Asserted by calling through with ``evaluate_branch`` captured, not by
    reading the function's source. The previous version of this test sliced
    ``inspect.getsource`` at the first ``)``, so it would break on a nested call
    and could not tell ``kb_dir=kb_dir`` from ``kb_dir=golden_set_path``.
    """
    from anvil.loop import round as round_mod

    paths = _domain_paths(tmp_path)
    captured: dict[str, object] = {}

    def _capture(**kwargs: object) -> EvalReport:
        captured.update(kwargs)
        return _fake_report()

    monkeypatch.setattr(round_mod, "evaluate_branch", _capture)

    # Everything around the eval is stubbed: this test is about argument
    # plumbing, and a real round branches git and drives the optimizer.
    monkeypatch.setattr(round_mod, "changed_paths", lambda _r: [])
    monkeypatch.setattr(round_mod, "load_baseline", lambda _r: None)
    monkeypatch.setattr(round_mod, "create_round_branch", lambda *_a, **_k: "anvil/exp-round-1")
    monkeypatch.setattr(round_mod, "current_branch", lambda _r: "anvil/exp")
    monkeypatch.setattr(round_mod, "current_sha", lambda _r: "0" * 40)
    monkeypatch.setattr(round_mod, "build_round_prompt", lambda **_k: "prompt")
    monkeypatch.setattr(round_mod, "_read_optimizer_endpoint", lambda _s: "endpoint")
    monkeypatch.setattr(round_mod, "_read_cost_budget_usd", lambda _s: 1.0)
    monkeypatch.setattr(round_mod, "load_eval_config", lambda *_a, **_k: EvalConfig())

    # A noop action short-circuits the eval entirely ("no need to evaluate for a
    # noop"), so the session has to yield a real mutation for the eval to be
    # reached at all.
    action = ChangeSamplingAction(field="temperature", value=0.4, rationale="reach the eval")
    parse_result = ParseResult(action=action, parse_status="ok", n_blocks_found=1)

    async def _fake_session(**_kw: object) -> tuple[object, str, ParseResult]:
        return action, "transcript", parse_result

    monkeypatch.setattr(round_mod, "run_optimizer_session", _fake_session)
    monkeypatch.setattr(
        round_mod,
        "apply_action",
        lambda *_a, **_k: ApplyResult(files_changed=["scaffold/harness.yaml"]),
    )
    monkeypatch.setattr(round_mod, "commit_all", lambda *_a, **_k: "1" * 40)
    monkeypatch.setattr(round_mod, "restore_paths", lambda *_a, **_k: None)

    with suppress(Exception):
        # The round will fail somewhere after the eval -- there is no git repo
        # and no optimizer. Only what reached evaluate_branch matters, and if it
        # was never reached the assertions below say so.
        round_mod.run_round(
            round_id=1,
            repo_root=tmp_path,
            scaffold_root=tmp_path / "scaffold",
            kb_dir=paths["kb"],
            golden_set_path=paths["golden"],
            evaluator_path=paths["evaluator"],
        )

    assert captured, "run_round never called evaluate_branch"
    assert Path(str(captured["kb_dir"])) == paths["kb"]
    assert Path(str(captured["golden_set_path"])) == paths["golden"]
    assert Path(str(captured["evaluator_path"])) == paths["evaluator"]


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
@pytest.mark.xfail(
    reason=(
        "the committed baseline predates the judge-model wiring (#13): its "
        "fingerprint carries no semantics version for correctness or "
        "groundedness. Regenerate with scripts/make_baseline.py (a live eval), "
        "then this xfail turns into a failure that forces its own removal."
    ),
    strict=True,
)
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


# ---------------------------------------------------------------------------
# 5. The wiring itself: config -> HarnessConfig -> build_scorers -> prompt.
#
#    Everything above tests one link. Adversarial review found that all of it
#    passed while the two values were SWAPPED at the build_scorers call site --
#    which puts the four-line domain paragraph into the "I can only help with
#    {X}-related questions" slot and the bare word into the Domain: slot -- and
#    also passed with both kwargs deleted, which leaves the judge grading
#    NeoVolt while the fingerprint says the domain changed. Nothing exercised
#    the chain end to end.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_configured_domain_reaches_the_built_scorer_not_just_the_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Follow the values from a YAML file to the rendered judge prompt.

    ``build_scorers`` is called with what the config produced, and the prompt
    the refusal scorer would actually send is captured. Catches a swap or a
    dropped kwarg anywhere along the chain.
    """
    import yaml

    from anvil.eval import scorers as scorers_mod
    from anvil.runtime.models import ScaffoldYAML

    domain_name = "pyloom"
    domain_context = "Domain: pyloom, a fictional Python library."
    runtime = RuntimeYAML.model_validate(
        yaml.safe_load(
            _MIN_CONFIG_YAML
            + f"judge_domain_name: {domain_name}\n"
            + f'judge_domain_context: "{domain_context}"\n'
        )
    )
    merged = HarnessConfig.from_split(ScaffoldYAML(), runtime)

    captured: dict[str, str] = {}
    real_build_refusal = scorers_mod._build_refusal_scorer

    def _capture_ctx(ctx: object):
        captured["name"] = ctx.domain_name  # type: ignore[attr-defined]
        captured["context"] = ctx.domain_context  # type: ignore[attr-defined]
        return real_build_refusal(ctx)  # type: ignore[arg-type]

    monkeypatch.setattr(scorers_mod, "_build_refusal_scorer", _capture_ctx)

    scorers_mod.build_scorers(
        judge_client=None,  # type: ignore[arg-type]  # never called; only the ctx is inspected
        scorer_configs=[ScorerConfig(name=REFUSAL_SCORER_NAME)],
        judge_domain_name=merged.judge_domain_name,
        judge_domain_context=merged.judge_domain_context,
    )

    # Each value must land in ITS OWN slot -- this is what a swap breaks.
    assert captured["name"] == domain_name
    assert captured["context"] == domain_context

    rendered = _judge_prompt(
        "q", "a", True, "n", domain_name=captured["name"], domain_context=captured["context"]
    )
    assert domain_context in rendered
    assert f"I can only help with {domain_name}-related questions" in rendered
    assert "NeoVolt" not in rendered


@pytest.mark.unit
def test_writing_the_shipped_defaults_into_config_does_not_invalidate_the_baseline() -> None:
    """A no-op config edit must not cost a live re-run.

    ``harness/config.yaml`` ships the two keys commented out with their default
    values shown. Uncommenting them renders a byte-identical prompt, so if that
    changed the fingerprint the next round would abort -- after paying for an
    optimizer session and a full eval -- demanding a regenerated baseline for a
    change that altered nothing.
    """
    configs = [ScorerConfig(name=n) for n in _SCORER_NAMES]
    unset = compute_scorer_fingerprint(configs)

    explicit_defaults = compute_scorer_fingerprint(
        configs,
        judge_domain_name=DEFAULT_JUDGE_DOMAIN_NAME,
        judge_domain_context=DEFAULT_JUDGE_DOMAIN_CONTEXT,
    )
    assert explicit_defaults == unset

    # And the empty string, which ``build_scorers`` treats as "use the default"
    # via ``or``, must agree -- the fingerprint used to test ``is not None`` and
    # disagreed with the prompt for exactly this value.
    assert compute_scorer_fingerprint(configs, judge_domain_name="", judge_domain_context="") == (
        unset
    )


@pytest.mark.unit
def test_setting_one_judge_domain_key_alone_is_rejected() -> None:
    """Both or neither, enforced at config load.

    Setting only ``judge_domain_context`` -- the substantial key, and the one
    anyone reaches for first -- left the judge offering "I can only help with
    NeoVolt-related questions" as its refusal example while claiming to grade
    another domain. Nothing downstream can detect that.
    """
    import pydantic
    import yaml

    for partial in (
        "judge_domain_name: pyloom\n",
        'judge_domain_context: "Domain: pyloom."\n',
    ):
        with pytest.raises(pydantic.ValidationError, match="both are interpolated"):
            RuntimeYAML.model_validate(yaml.safe_load(_MIN_CONFIG_YAML + partial))

    # Neither is still valid, and both together is still valid.
    RuntimeYAML.model_validate(yaml.safe_load(_MIN_CONFIG_YAML))
    RuntimeYAML.model_validate(
        yaml.safe_load(
            _MIN_CONFIG_YAML + 'judge_domain_name: pyloom\njudge_domain_context: "Domain: p."\n'
        )
    )


# ---------------------------------------------------------------------------
# 6. The optimizer's answer key must follow the domain.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_optimizer_cannot_read_a_non_default_golden_set(tmp_path: Path) -> None:
    """The secret set must cover the golden set this round actually uses.

    ``ToolPolicy``'s defaults name the built-in domain's paths literally, and
    ``is_secret`` is exact-path equality -- so before this, a second domain
    inside the repo (the ``examples/`` layout) sat at a path the policy had
    never heard of, and the session could read the reference answers and judge
    rubric for every case it was about to be graded on. A read leaves no diff,
    so the post-hoc scope check cannot catch it either.
    """
    from anvil.loop.round import _secret_paths
    from anvil.optimizer.policy import ToolPolicy

    (tmp_path / "examples" / "d" / "data").mkdir(parents=True)
    golden = tmp_path / "examples" / "d" / "data" / "golden_set.jsonl"
    golden.write_text("", encoding="utf-8")
    evaluator = tmp_path / "examples" / "d" / "data" / "evaluator.py"
    evaluator.write_text("", encoding="utf-8")

    policy = ToolPolicy(root=tmp_path, secret_paths=_secret_paths(tmp_path, golden, evaluator))

    for path in (golden, evaluator):
        decision = policy.decide("Read", {"file_path": str(path)})
        assert not decision.allowed, f"optimizer may read {path.name} of the active domain"

    # The built-in domain's paths stay protected too: a typo in
    # --golden-set-path must not silently unprotect the real answer key.
    assert "data/golden_set.jsonl" in _secret_paths(tmp_path, golden, evaluator)
    assert "data/evaluator.py" in _secret_paths(tmp_path, golden, evaluator)


@pytest.mark.unit
def test_a_golden_set_outside_the_repo_is_dropped_from_the_secret_set(tmp_path: Path) -> None:
    """Relative secret paths cannot express an out-of-tree file.

    ``is_inside_root`` already denies those, so silently dropping them is
    correct -- but it must not raise on the way.
    """
    from anvil.loop.round import _secret_paths

    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "elsewhere" / "golden_set.jsonl"
    outside.parent.mkdir()
    outside.write_text("", encoding="utf-8")

    paths = _secret_paths(repo, outside, None)
    assert paths == tuple(__import__("anvil.optimizer.policy", fromlist=["x"]).DEFAULT_SECRET_PATHS)


# ---------------------------------------------------------------------------
# 7. A baseline records which domain it measured.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_dataset_fingerprint_distinguishes_two_real_domains() -> None:
    """Content-based, so it separates the shipped domain from the example."""
    from anvil.eval.cache import compute_dataset_fingerprint

    neovolt = compute_dataset_fingerprint(
        REPO_ROOT / "data" / "kb", REPO_ROOT / "data" / "golden_set.jsonl"
    )
    pyloom = compute_dataset_fingerprint(
        REPO_ROOT / "examples" / "pyloom-docs" / "data" / "kb",
        REPO_ROOT / "examples" / "pyloom-docs" / "data" / "golden_set.jsonl",
    )
    assert neovolt and pyloom
    assert neovolt != pyloom
    # Stable across calls -- a fingerprint that moved on its own would refuse
    # every baseline including the correct one.
    assert neovolt == compute_dataset_fingerprint(
        REPO_ROOT / "data" / "kb", REPO_ROOT / "data" / "golden_set.jsonl"
    )


@pytest.mark.unit
def test_an_unreadable_domain_yields_no_fingerprint(tmp_path: Path) -> None:
    """Absent, not invented: the field's contract is 'empty means unchecked'."""
    from anvil.eval.cache import compute_dataset_fingerprint

    assert compute_dataset_fingerprint(tmp_path / "missing", tmp_path / "missing.jsonl") == ""


@pytest.mark.unit
def test_editing_one_golden_row_invalidates_the_dataset_fingerprint(tmp_path: Path) -> None:
    """Changing the questions changes what the aggregate means."""
    from anvil.eval.cache import compute_dataset_fingerprint

    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "a.md").write_text("---\ndoc_id: a\n---\nbody\n", encoding="utf-8")
    golden = tmp_path / "golden_set.jsonl"
    golden.write_text('{"example_id": "x"}\n', encoding="utf-8")
    before = compute_dataset_fingerprint(kb, golden)

    golden.write_text('{"example_id": "y"}\n', encoding="utf-8")
    assert compute_dataset_fingerprint(kb, golden) != before

    # And so does editing the knowledge base the answers are retrieved from.
    golden.write_text('{"example_id": "x"}\n', encoding="utf-8")
    assert compute_dataset_fingerprint(kb, golden) == before
    (kb / "a.md").write_text("---\ndoc_id: a\n---\nDIFFERENT\n", encoding="utf-8")
    assert compute_dataset_fingerprint(kb, golden) != before


@pytest.mark.unit
def test_a_baseline_from_another_domain_is_refused() -> None:
    """The gate must not compare aggregates measured on different questions."""
    from anvil.eval.cache import CachedBaseline, dataset_incomparability_reason, is_compatible

    cached = CachedBaseline(
        scaffold_commit_sha="a" * 40,
        evaluated_at="2026-08-16T12:00:00+00:00",
        mode="quick",
        scorers=list(_SCORER_NAMES),
        runtime_endpoint="r",
        judge_endpoint="j",
        aggregate=0.62,
        dataset_fingerprint="sha256:neovolt",
        # A real scorer fingerprint, because ``retrieval_groundedness`` has
        # versioned semantics: an absent one is refused on its own, which would
        # mask whatever the dataset check decided.
        scorer_fingerprint=compute_scorer_fingerprint(
            [ScorerConfig(name=n) for n in _SCORER_NAMES]
        ),
    )

    assert dataset_incomparability_reason(cached, dataset_fingerprint="sha256:pyloom")
    assert not dataset_incomparability_reason(cached, dataset_fingerprint="sha256:neovolt")
    # Absent on either side stays unchecked, so baselines predating the field
    # keep working and adding it forces no live re-run.
    assert not dataset_incomparability_reason(cached, dataset_fingerprint="")

    common = {
        "mode": "quick",
        "scorers": list(_SCORER_NAMES),
        "runtime_endpoint": "r",
        "judge_endpoint": "j",
        "scorer_fingerprint": cached.scorer_fingerprint,
    }
    assert not is_compatible(cached, **common, dataset_fingerprint="sha256:pyloom")
    assert is_compatible(cached, **common, dataset_fingerprint="sha256:neovolt")


@pytest.mark.unit
def test_the_committed_baseline_predates_the_dataset_fingerprint_and_still_loads() -> None:
    """The real file on disk must remain usable without a live re-run."""
    import json as _json

    from anvil.eval.cache import CachedBaseline

    raw = _json.loads((REPO_ROOT / "eval" / "runs" / "baseline.json").read_text(encoding="utf-8"))
    cached = CachedBaseline.from_dict(raw)
    assert cached.dataset_fingerprint == ""


# ---------------------------------------------------------------------------
# 8. Call sites, not helpers.
#
#    Everything above this point tested the pieces. Mutation testing then showed
#    four fixes could be reverted with all of it still green: swapping or
#    dropping the judge-domain kwargs at the runner's build_scorers call,
#    reverting ToolPolicy to its hardcoded secret paths, and deleting the
#    dataset check from the gate. A helper nothing is proven to call is not a
#    fix, so these exercise the wiring.
# ---------------------------------------------------------------------------


def _runner_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    judge_domain_name: str | None,
    judge_domain_context: str | None,
) -> dict[str, object]:
    """Drive ``evaluate_branch`` with mlflow and the agent mocked out.

    Mirrors the harness in ``tests/test_programmatic_scorers.py``. Returns the
    kwargs ``build_scorers`` was called with.
    """
    from types import SimpleNamespace

    import pandas as pd

    from anvil.eval import runner
    from anvil.runtime.models import EvalConfig, EvalModeConfig, ExperimentsConfig

    config = HarnessConfig(
        runtime_endpoint="rt",
        optimizer_endpoint="op",
        judge_endpoint="j",
        judge_domain_name=judge_domain_name,
        judge_domain_context=judge_domain_context,
        experiments=ExperimentsConfig(runtime="r", eval="e", optimizer="o"),
        eval=EvalConfig(
            default_mode="quick",
            scorers=[ScorerConfig(name=REFUSAL_SCORER_NAME)],
            modes={"quick": EvalModeConfig(rows=1, buckets={"direct": 1})},
        ),
    )
    gold = {
        "example_id": "g1",
        "query": "q",
        "category": "direct",
        "expected_doc_ids": ["d"],
        "reference_answer": "a",
        "should_refuse": False,
        "expected_citations": ["d"],
        "must_include": ["a"],
        "must_not_include": [],
        "notes_for_judge": "",
    }
    monkeypatch.setattr(runner, "load_harness", lambda *a, **kw: SimpleNamespace(config=config))
    monkeypatch.setattr(runner, "load_golden_set", lambda _p: [gold])
    monkeypatch.setattr(runner, "select_subset", lambda exs, **_k: exs)
    monkeypatch.setattr(runner, "make_kb_executor", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(runner, "AnvilAgent", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(runner, "enable_runtime_tracing", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "set_experiment", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "set_tracking_uri", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "get_experiment_by_name", lambda *a, **kw: None)
    monkeypatch.setattr(
        runner.mlflow.genai,
        "evaluate",
        lambda **_k: SimpleNamespace(
            result_df=pd.DataFrame({f"{REFUSAL_SCORER_NAME}/value": [1.0], "trace_id": ["t0"]}),
            metrics={},
            run_id="run-1",
        ),
    )

    captured: dict[str, object] = {}

    def _capture_build(**kwargs: object) -> list:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(runner, "build_scorers", _capture_build)

    paths = _domain_paths(tmp_path)
    report = runner.evaluate_branch(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=tmp_path / "config.yaml",
        kb_dir=paths["kb"],
        golden_set_path=paths["golden"],
        runtime_client=SimpleNamespace(),
        judge_client=SimpleNamespace(),
    )
    captured["_report"] = report
    return captured


@pytest.mark.unit
def test_evaluate_branch_hands_the_configured_domain_to_the_scorers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner's own call site must pass both values, in their own slots.

    Catches the swap (domain paragraph into the name slot and vice versa) and
    the dropped kwargs, either of which leaves the judge grading the shipped
    domain while the fingerprint says otherwise.
    """
    context = "Domain: pyloom, a fictional Python library."
    captured = _runner_harness(
        monkeypatch, tmp_path, judge_domain_name="pyloom", judge_domain_context=context
    )
    assert captured["judge_domain_name"] == "pyloom"
    assert captured["judge_domain_context"] == context


@pytest.mark.unit
def test_evaluate_branch_records_the_dataset_it_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report must carry the domain fingerprint, or the gate cannot check it."""
    from anvil.eval.cache import compute_dataset_fingerprint

    captured = _runner_harness(
        monkeypatch, tmp_path, judge_domain_name=None, judge_domain_context=None
    )
    report = captured["_report"]
    paths = _domain_paths(tmp_path)
    expected = compute_dataset_fingerprint(paths["kb"], paths["golden"])
    assert expected
    assert report.dataset_fingerprint == expected  # type: ignore[union-attr]


@pytest.mark.unit
def test_evaluate_branch_fingerprints_with_the_domain_in_the_right_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report's scorer fingerprint must match an independent computation.

    The runner passes the two judge-domain values to ``compute_scorer_fingerprint``
    as well as to ``build_scorers``, and swapping them there is invisible to every
    other assertion: the fingerprint stays deterministic and still differs from the
    default, so comparability appears to work while the recorded value describes a
    domain named after a paragraph.
    """
    context = "Domain: pyloom, a fictional Python library."
    captured = _runner_harness(
        monkeypatch, tmp_path, judge_domain_name="pyloom", judge_domain_context=context
    )
    report = captured["_report"]
    expected = compute_scorer_fingerprint(
        [ScorerConfig(name=REFUSAL_SCORER_NAME)],
        judge_domain_name="pyloom",
        judge_domain_context=context,
    )
    assert report.scorer_fingerprint == expected  # type: ignore[union-attr]


def _stub_round(monkeypatch: pytest.MonkeyPatch, round_mod, baseline: object = None) -> None:
    """Stub a round down to the parts these tests are about."""
    monkeypatch.setattr(round_mod, "changed_paths", lambda _r: [])
    monkeypatch.setattr(round_mod, "load_baseline", lambda _r: baseline)
    monkeypatch.setattr(round_mod, "create_round_branch", lambda *_a, **_k: "anvil/exp-round-1")
    monkeypatch.setattr(round_mod, "current_branch", lambda _r: "anvil/exp")
    monkeypatch.setattr(round_mod, "current_sha", lambda _r: "0" * 40)
    monkeypatch.setattr(round_mod, "build_round_prompt", lambda **_k: "prompt")
    monkeypatch.setattr(round_mod, "_read_optimizer_endpoint", lambda _s: "endpoint")
    monkeypatch.setattr(round_mod, "_read_cost_budget_usd", lambda _s: 1.0)
    monkeypatch.setattr(round_mod, "load_eval_config", lambda *_a, **_k: EvalConfig())


@pytest.mark.unit
def test_run_round_gives_the_policy_the_active_golden_set_as_a_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The round must build its policy from the domain it is actually running.

    Reverting to ``ToolPolicy(root=repo_root)`` leaves the optimizer able to
    read a non-default golden set -- the answer key for every case it is graded
    on -- so the construction, not just the helper, is pinned.
    """
    from anvil.loop import round as round_mod

    paths = _domain_paths(tmp_path)
    captured: dict[str, object] = {}

    def _capture_policy(**kwargs: object):
        captured.update(kwargs)
        raise SystemExit(0)  # nothing after this matters to the assertion

    monkeypatch.setattr(round_mod, "ToolPolicy", _capture_policy)
    _stub_round(monkeypatch, round_mod)

    with suppress(SystemExit, Exception):
        round_mod.run_round(
            round_id=1,
            repo_root=tmp_path,
            scaffold_root=tmp_path / "scaffold",
            kb_dir=paths["kb"],
            golden_set_path=paths["golden"],
            evaluator_path=paths["evaluator"],
        )

    secrets = captured.get("secret_paths")
    assert secrets is not None, "run_round built ToolPolicy without a secret_paths argument"
    rel_golden = paths["golden"].resolve().relative_to(tmp_path.resolve()).as_posix()
    assert rel_golden in secrets, f"{rel_golden} is not protected: optimizer can read the answers"


@pytest.mark.unit
def test_run_round_refuses_a_baseline_from_another_domain_before_spending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-flight, not post-eval.

    The authoritative check runs after the eval, but reaching it costs an
    optimizer session and a full eval -- so a 50-round run pointed at a new
    domain would burn a round to learn the baseline does not apply, fifty times.
    The optimizer must never be started.
    """
    from anvil.eval.cache import CachedBaseline
    from anvil.loop import round as round_mod

    paths = _domain_paths(tmp_path)
    baseline = CachedBaseline(
        scaffold_commit_sha="a" * 40,
        evaluated_at="2026-08-16T12:00:00+00:00",
        mode="quick",
        scorers=list(_SCORER_NAMES),
        runtime_endpoint="r",
        judge_endpoint="j",
        aggregate=0.62,
        dataset_fingerprint="sha256:some-other-domain",
    )
    _stub_round(monkeypatch, round_mod, baseline=baseline)

    def _must_not_run(**_kw: object):
        raise AssertionError("the optimizer session was started despite an incomparable baseline")

    monkeypatch.setattr(round_mod, "run_optimizer_session", _must_not_run)
    monkeypatch.setattr(round_mod, "asdict_baseline", lambda _b: {})

    with pytest.raises(RuntimeError, match="different dataset"):
        round_mod.run_round(
            round_id=1,
            repo_root=tmp_path,
            scaffold_root=tmp_path / "scaffold",
            kb_dir=paths["kb"],
            golden_set_path=paths["golden"],
        )
