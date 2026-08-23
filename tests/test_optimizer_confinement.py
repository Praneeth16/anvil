"""Adversarial tests for the optimizer's confinement.

Every test here is a cheat the optimizer could otherwise take to raise its score
without improving the agent. They are written as attacks rather than as feature
checks because that is what they are defending against: a process that is
rewarded for the number going up, exploring the space of ways to make it go up.

Two independent layers are exercised:

* :meth:`ToolPolicy.decide` -- the permission callback, which stops the call.
* :meth:`ToolPolicy.verify_changed_paths` -- the git-diff check, which stops the
  *round* even when the callback did not run. The second layer exists because the
  first depends on the Claude Agent SDK honouring ``can_use_tool``, and a diff
  depends on nothing.
"""

from __future__ import annotations

import subprocess

import pytest

from anvil.eval.cache import CachedBaseline, save_baseline
from anvil.loop.decision import Decision
from anvil.optimizer.actions import NoopAction
from anvil.optimizer.parser import ParseResult
from anvil.optimizer.policy import ALLOWED_TOOLS, ToolPolicy

pytestmark = pytest.mark.unit


@pytest.fixture
def repo(tmp_path):
    """A miniature ANVIL tree with the files that matter."""
    (tmp_path / "scaffold" / "skills").mkdir(parents=True)
    (tmp_path / "scaffold" / "memory").mkdir(parents=True)
    (tmp_path / "agents").mkdir()
    (tmp_path / "harness").mkdir()
    (tmp_path / "data" / "kb").mkdir(parents=True)
    (tmp_path / "eval" / "runs").mkdir(parents=True)
    (tmp_path / "src" / "anvil" / "loop").mkdir(parents=True)
    (tmp_path / "tests").mkdir()

    (tmp_path / "scaffold" / "skills" / "refund.md").write_text("# refund\n")
    (tmp_path / "harness" / "config.yaml").write_text("gate:\n  epsilon: 0.0\n")
    (tmp_path / "data" / "golden_set.jsonl").write_text(
        '{"example_id":"g1","reference_answer":"$0.142 per kWh","must_include":["$0.142"]}\n'
    )
    (tmp_path / "data" / "evaluator.py").write_text("def exact_match(a, b): return a == b\n")
    (tmp_path / "data" / "kb" / "tariff.md").write_text("# tariff\n")
    (tmp_path / "eval" / "runs" / "baseline.json").write_text('{"aggregate": 0.5}\n')
    (tmp_path / "src" / "anvil" / "loop" / "frontier.py").write_text("EPSILON = 0.0\n")
    (tmp_path / "tests" / "test_frontier.py").write_text("def test_x(): pass\n")
    return tmp_path


@pytest.fixture
def policy(repo):
    return ToolPolicy(root=repo)


# -- reading the answer key -------------------------------------------------


def test_reading_the_golden_set_is_denied(policy):
    """The single highest-value cheat: read the reference answers, hardcode them."""
    decision = policy.decide("Read", {"file_path": "data/golden_set.jsonl"})
    assert not decision.allowed
    assert "memorisation" in decision.reason


def test_grepping_the_repo_root_is_denied_because_it_spans_the_golden_set(policy):
    """A repo-wide grep returns matching lines *from* the answer key."""
    decision = policy.decide("Grep", {"pattern": "0.142", "path": "."})
    assert not decision.allowed
    assert "narrower" in decision.reason


def test_grep_with_no_path_is_judged_as_a_repo_wide_read(policy):
    """Omitting `path` defaults to the working directory -- not a way around it."""
    decision = policy.decide("Grep", {"pattern": "reference_answer"})
    assert not decision.allowed


def test_globbing_the_data_dir_is_denied(policy):
    decision = policy.decide("Glob", {"pattern": "*.jsonl", "path": "data"})
    assert not decision.allowed


def test_reading_the_knowledge_base_is_allowed(policy):
    """The KB is the agent's legitimate retrieval corpus, not the answer key."""
    assert policy.decide("Read", {"file_path": "data/kb/tariff.md"}).allowed
    assert policy.decide("Grep", {"pattern": "tariff", "path": "data/kb"}).allowed


def test_reading_own_scaffold_is_allowed(policy):
    assert policy.decide("Read", {"file_path": "scaffold/skills/refund.md"}).allowed


# -- weakening the grader ---------------------------------------------------


@pytest.mark.parametrize(
    "target",
    [
        "harness/config.yaml",
        "data/evaluator.py",
        "data/golden_set.jsonl",
        "src/anvil/loop/frontier.py",
        "tests/test_frontier.py",
        "eval/runs/baseline.json",
    ],
)
def test_writing_to_the_grader_is_denied(policy, target):
    """Each of these can raise the score without improving the agent."""
    for tool in ("Write", "Edit", "MultiEdit"):
        decision = policy.decide(tool, {"file_path": target})
        assert not decision.allowed, f"{tool} on {target} was permitted"


def test_appending_an_easy_case_to_the_golden_set_is_denied(policy):
    decision = policy.decide("Edit", {"file_path": "data/golden_set.jsonl"})
    assert not decision.allowed
    assert "graded" in decision.reason


def test_writing_the_scaffold_is_allowed(policy):
    """The mutation surface itself must stay open, or the harness does nothing."""
    assert policy.decide("Write", {"file_path": "scaffold/rules/new_rule.md"}).allowed
    assert policy.decide("Edit", {"file_path": "scaffold/skills/refund.md"}).allowed


def test_writing_a_code_mode_agent_is_allowed(policy):
    assert policy.decide("Write", {"file_path": "agents/candidate.py"}).allowed


# -- escaping the tree ------------------------------------------------------


def test_parent_traversal_is_denied(policy):
    decision = policy.decide("Write", {"file_path": "scaffold/../harness/config.yaml"})
    assert not decision.allowed


def test_absolute_path_outside_the_repo_is_denied(policy):
    decision = policy.decide("Write", {"file_path": "/etc/hosts"})
    assert not decision.allowed
    assert "outside the repository" in decision.reason


def test_symlink_escape_is_denied(policy, repo, tmp_path):
    """A link inside the writable scope pointing out of it resolves out of it."""
    outside = tmp_path.parent / "outside_target"
    outside.mkdir(exist_ok=True)
    link = repo / "scaffold" / "escape"
    link.symlink_to(outside, target_is_directory=True)

    decision = policy.decide("Write", {"file_path": "scaffold/escape/payload.md"})
    assert not decision.allowed


def test_reading_outside_the_repo_is_denied(policy):
    assert not policy.decide("Read", {"file_path": "/etc/passwd"}).allowed


# -- tool allowlist ---------------------------------------------------------


@pytest.mark.parametrize("tool", ["Bash", "WebFetch", "WebSearch", "Task", "NotebookEdit"])
def test_dangerous_tools_are_denied(policy, tool):
    """Bash is a write primitive no path policy can inspect; the web tools exfiltrate."""
    decision = policy.decide(tool, {"command": "echo pwned > harness/config.yaml"})
    assert not decision.allowed
    assert "not available" in decision.reason


def test_unknown_future_tool_arrives_denied(policy):
    """An allowlist means a new SDK tool is denied until someone permits it."""
    assert not policy.decide("SomeNewTool2027", {"file_path": "scaffold/x.md"}).allowed


def test_allowlist_covers_what_the_optimizer_actually_needs(policy):
    for tool in ALLOWED_TOOLS:
        assert tool in ("Read", "Glob", "Grep", "Write", "Edit", "MultiEdit")


# -- the layer that holds when the callback does not -----------------------


def test_diff_verification_catches_a_grader_edit(policy):
    """The check that survives an SDK that stops calling `can_use_tool`."""
    violations = policy.verify_changed_paths(
        [
            "scaffold/skills/refund.md",
            "harness/config.yaml",
            "data/golden_set.jsonl",
        ]
    )
    assert violations == ("data/golden_set.jsonl", "harness/config.yaml")


def test_diff_verification_passes_a_clean_round(policy):
    violations = policy.verify_changed_paths(
        ["scaffold/skills/refund.md", "scaffold/memory/round_001_critique.md"]
    )
    assert violations == ()


def test_diff_verification_ignores_empty_lines(policy):
    """`git diff --name-only` output ends with a newline; the split yields ''."""
    assert policy.verify_changed_paths(["scaffold/x.md", ""]) == ()


def test_diff_verification_flags_writes_outside_the_repo(policy):
    assert policy.verify_changed_paths(["/etc/hosts"]) == ("/etc/hosts",)


# -- the whole round, with the callback bypassed ----------------------------


def _git(repo, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


@pytest.fixture
def anvil_repo(tmp_path):
    """A committed ANVIL repo on ``anvil/exp``, ready for ``run_round``."""
    repo = tmp_path
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@e.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")

    (repo / "scaffold" / "memory").mkdir(parents=True)
    (repo / "scaffold" / "harness.yaml").write_text("tools: []\n")
    (repo / "harness").mkdir()
    (repo / "harness" / "config.yaml").write_text("mode: prompt\n")
    (repo / "eval" / "runs").mkdir(parents=True)

    save_baseline(
        repo,
        CachedBaseline(
            scaffold_commit_sha="a" * 40,
            evaluated_at="2026-08-22T12:00:00+00:00",
            mode="test",
            scorers=["correctness"],
            runtime_endpoint="runtime",
            judge_endpoint="judge",
            aggregate=0.5,
            per_judge={"correctness": 0.5},
            per_bucket={"direct": {"correctness": 0.5}},
            n_examples=10,
        ),
    )

    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    _git(repo, "checkout", "-q", "-b", "anvil/exp")
    return repo


def test_round_fails_when_the_session_escapes_its_scope(anvil_repo, monkeypatch):
    """The case the permission callback cannot cover.

    Simulates an SDK that ignores `can_use_tool` -- the session writes the gate
    config directly. The round must not be scored: the mutation may have altered
    what was about to grade it. It must be failed, the edit reverted, and the
    branch discarded.
    """
    import anvil.loop.round as round_mod

    original_config = (anvil_repo / "harness" / "config.yaml").read_text()

    async def _escaping_session(**kwargs):
        # Bypass the policy entirely, exactly as a non-cooperating SDK would.
        (anvil_repo / "harness" / "config.yaml").write_text("mode: prompt\ngate:\n  epsilon: -1.0\n")
        action = NoopAction(rationale="pretending to do nothing")
        return action, "transcript", ParseResult(action=action, parse_status="ok", n_blocks_found=1)

    monkeypatch.setattr(round_mod, "run_optimizer_session", _escaping_session)

    def _fail_if_called(**kwargs):
        raise AssertionError("a scope-violating round must not be evaluated")

    monkeypatch.setattr(round_mod, "evaluate_branch", _fail_if_called)

    report = round_mod.run_round(round_id=1, repo_root=anvil_repo)

    assert report.decision == Decision.INFRA_FAIL
    assert "scope violation" in report.notes
    assert "harness/config.yaml" in report.notes
    # The edit is gone, not merely unstaged -- a checkout would otherwise carry
    # it onto the parent branch for the next round to inherit.
    assert (anvil_repo / "harness" / "config.yaml").read_text() == original_config
    assert "anvil/exp-round-1" not in _git(anvil_repo, "branch", "--list")


def test_preexisting_dirt_is_not_blamed_on_the_session(anvil_repo, monkeypatch):
    """The scope check attributes writes, it does not audit the tree.

    Leftover round artifacts and `--allow-dirty` runs mean out-of-scope files are
    routinely already present when a round starts. Failing the round for those
    would make the check fire at random, which is how a safety check gets turned
    off.
    """
    import anvil.loop.round as round_mod

    # Out-of-scope, untracked, and present *before* the session runs.
    (anvil_repo / "eval" / "runs" / "round_000.json").write_text("{}")

    async def _touches_nothing(**kwargs):
        action = NoopAction(rationale="no mutation this round")
        return action, "transcript", ParseResult(action=action, parse_status="ok", n_blocks_found=1)

    monkeypatch.setattr(round_mod, "run_optimizer_session", _touches_nothing)

    report = round_mod.run_round(round_id=1, repo_root=anvil_repo)

    assert report.decision != Decision.INFRA_FAIL
    assert report.notes == ""
    # And the pre-existing file was left alone, not "restored" out of existence.
    assert (anvil_repo / "eval" / "runs" / "round_000.json").exists()


def test_round_proceeds_normally_when_the_session_stays_in_scope(anvil_repo, monkeypatch):
    """The control case: an in-scope round is not tripped by the new check."""
    import anvil.loop.round as round_mod

    async def _wellbehaved_session(**kwargs):
        (anvil_repo / "scaffold" / "rules").mkdir(exist_ok=True)
        (anvil_repo / "scaffold" / "rules" / "new.md").write_text("# be concise\n")
        action = NoopAction(rationale="scaffold touched directly, in scope")
        return action, "transcript", ParseResult(action=action, parse_status="ok", n_blocks_found=1)

    monkeypatch.setattr(round_mod, "run_optimizer_session", _wellbehaved_session)

    report = round_mod.run_round(round_id=1, repo_root=anvil_repo)

    assert report.decision != Decision.INFRA_FAIL
    assert "scope violation" not in report.notes

# ---------------------------------------------------------------------------
# The PreToolUse hook: a second enforcement point for the same rule.
#
# `can_use_tool` is one SDK code path and documented as best-effort. The hook is
# another. What must NOT happen is the two disagreeing, so these tests drive the
# hook and the permission callback with the same inputs and require identical
# verdicts -- if a future change teaches one about a path and not the other, the
# parity test fails rather than the round quietly losing a layer.


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def _hook_verdict(policy, tool_name: str, tool_input: dict):
    from anvil.optimizer.session import _pre_tool_use_hook

    hook = _pre_tool_use_hook(policy)
    out = _run(hook({"tool_name": tool_name, "tool_input": tool_input}, None, None))
    specific = out.get("hookSpecificOutput") or {}
    return specific.get("permissionDecision", "allow"), specific.get(
        "permissionDecisionReason", ""
    )


@pytest.mark.unit
def test_hook_denies_reading_the_golden_set(policy):
    decision, reason = _hook_verdict(policy, "Read", {"file_path": "data/golden_set.jsonl"})
    assert decision == "deny"
    assert reason


@pytest.mark.unit
def test_hook_allows_reading_the_scaffold(policy):
    decision, _ = _hook_verdict(policy, "Read", {"file_path": "scaffold/skills/identity.md"})
    assert decision == "allow"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("tool_name", "tool_input"),
    [
        ("Read", {"file_path": "data/golden_set.jsonl"}),
        ("Read", {"file_path": "scaffold/skills/identity.md"}),
        ("Grep", {"pattern": "x"}),
        ("Glob", {"path": "data"}),
        ("Write", {"file_path": "harness/config.yaml"}),
        ("Write", {"file_path": "scaffold/skills/identity.md"}),
        ("Bash", {"command": "ls"}),
        ("SomeFutureTool", {"file_path": "scaffold/x.md"}),
        ("Read", {"file_path": "../outside.txt"}),
    ],
)
def test_hook_and_permission_callback_never_disagree(policy, tool_name, tool_input):
    """One rule, two enforcement points. Two rules would be the bug."""
    from anvil.optimizer.session import _permission_callback

    hook_decision, _ = _hook_verdict(policy, tool_name, tool_input)
    result = _run(_permission_callback(policy)(tool_name, tool_input, None))
    callback_allowed = type(result).__name__ == "PermissionResultAllow"

    assert (hook_decision == "allow") is callback_allowed


@pytest.mark.unit
def test_hook_fails_closed_when_the_policy_raises():
    """A broken hook must deny, not open the door.

    The SDK may treat a hook that raises as a hook that is not there, which
    would silently drop this layer at exactly the moment it is malfunctioning.
    ``ToolPolicy`` is frozen, so this substitutes an exploding stand-in rather
    than patching a field.
    """

    class _Exploding:
        def decide(self, *_a, **_k):
            raise RuntimeError("policy exploded")

    decision, reason = _hook_verdict(
        _Exploding(), "Read", {"file_path": "scaffold/skills/identity.md"}
    )
    assert decision == "deny"
    assert "policy exploded" in reason


@pytest.mark.unit
def test_session_registers_the_hook_alongside_the_callback():
    """Both layers must actually be wired into the options the SDK receives.

    A hook that is written and never registered is the same as no hook, and
    nothing else in this file would notice.
    """
    import inspect

    from anvil.optimizer import session

    source = inspect.getsource(session.run_optimizer_session)
    assert "can_use_tool=_permission_callback(policy)" in source
    assert "_pre_tool_use_hook(policy)" in source
    assert '"PreToolUse"' in source
