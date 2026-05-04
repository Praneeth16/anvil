"""Tests for the action applier — file writes, harness.yaml registration."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from anvil.optimizer.actions import (
    AddRuleAction,
    AddSkillAction,
    ChangeSamplingAction,
    EditRuleAction,
    EditSkillAction,
    NoopAction,
)
from anvil.optimizer.applier import ApplyError, apply_action


def _bare_scaffold(tmp_path: Path) -> Path:
    """Build a minimal valid scaffold tree under ``tmp_path/scaffold``."""
    root = tmp_path / "scaffold"
    (root / "skills").mkdir(parents=True)
    (root / "rules").mkdir(parents=True)
    (root / "memory").mkdir(parents=True)
    (root / "skills" / "identity.md").write_text(
        dedent(
            """\
            ---
            skill_id: identity
            kind: identity
            applies_to: runtime
            ---

            # role
            test agent
            """
        ),
        encoding="utf-8",
    )
    (root / "rules" / "existing.md").write_text(
        "---\nrule_id: existing\napplies_to: runtime\n---\n\n# existing\nbody\n",
        encoding="utf-8",
    )
    (root / "harness.yaml").write_text(
        dedent(
            """\
            sampling:
              temperature: 0.7
              max_tool_calls: 3
              tool_choice: auto
              max_tokens: 2048
            skills:
              - file: identity.md
            rules:
              - file: existing.md
            tools: []
            """
        ),
        encoding="utf-8",
    )
    return root


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# ---------------------------------------------------------------------------
# noop
# ---------------------------------------------------------------------------


def test_apply_noop_touches_nothing(tmp_path: Path) -> None:
    root = _bare_scaffold(tmp_path)
    before = _read_yaml(root / "harness.yaml")
    result = apply_action(NoopAction(rationale="nothing"), root)
    assert result.files_added == []
    assert result.files_changed == []
    assert _read_yaml(root / "harness.yaml") == before
    assert "noop" in result.action_summary


# ---------------------------------------------------------------------------
# add_rule + add_skill
# ---------------------------------------------------------------------------


def test_apply_add_rule_writes_file_and_registers_in_harness(tmp_path: Path) -> None:
    root = _bare_scaffold(tmp_path)
    action = AddRuleAction(
        target_file="rules/scope.md",
        content="---\nrule_id: scope\napplies_to: runtime\n---\n\n# scope\nbody\n",
        rationale="seed scope rule",
    )
    result = apply_action(action, root)

    assert (root / "rules" / "scope.md").is_file()
    assert "scaffold/rules/scope.md" in result.files_added

    harness = _read_yaml(root / "harness.yaml")
    rule_files = [r["file"] for r in harness["rules"]]
    assert "scope.md" in rule_files
    assert "existing.md" in rule_files  # not displaced


def test_apply_add_rule_idempotent_on_already_registered(tmp_path: Path) -> None:
    """If harness.yaml already lists the file, registration is a no-op.

    Edge case the optimizer can hit if it edits a file then re-emits
    the action with the same target. The applier does NOT touch
    harness.yaml twice (avoids spurious diffs).
    """
    root = _bare_scaffold(tmp_path)
    # Pre-register the entry while leaving the file absent.
    harness = _read_yaml(root / "harness.yaml")
    harness["rules"].append({"file": "scope.md"})
    (root / "harness.yaml").write_text(yaml.safe_dump(harness, sort_keys=False), encoding="utf-8")

    action = AddRuleAction(
        target_file="rules/scope.md",
        content="---\nrule_id: scope\napplies_to: runtime\n---\n\n# scope\nbody\n",
        rationale="seed",
    )
    result = apply_action(action, root)
    assert "scaffold/rules/scope.md" in result.files_added
    # harness.yaml not changed because entry already present.
    assert result.files_changed == []


def test_apply_add_rule_rejects_existing(tmp_path: Path) -> None:
    root = _bare_scaffold(tmp_path)
    action = AddRuleAction(
        target_file="rules/existing.md",
        content="---\nrule_id: existing\napplies_to: runtime\n---\n\n# x\n",
        rationale="oops",
    )
    with pytest.raises(ApplyError, match="already exists"):
        apply_action(action, root)


def test_apply_add_skill(tmp_path: Path) -> None:
    root = _bare_scaffold(tmp_path)
    action = AddSkillAction(
        target_file="skills/refund.md",
        content="---\nskill_id: refund\napplies_to: billing_requests\n---\n\n# refund\n",
        rationale="add billing flow",
    )
    result = apply_action(action, root)
    assert (root / "skills" / "refund.md").is_file()
    harness = _read_yaml(root / "harness.yaml")
    assert "refund.md" in [s["file"] for s in harness["skills"]]
    assert "scaffold/skills/refund.md" in result.files_added


# ---------------------------------------------------------------------------
# edit_*
# ---------------------------------------------------------------------------


def test_apply_edit_rule_overwrites(tmp_path: Path) -> None:
    root = _bare_scaffold(tmp_path)
    new_content = "---\nrule_id: existing\napplies_to: runtime\n---\n\n# updated\n"
    action = EditRuleAction(
        target_file="rules/existing.md",
        content=new_content,
        rationale="tighten",
    )
    result = apply_action(action, root)
    assert (root / "rules" / "existing.md").read_text(encoding="utf-8") == new_content
    assert "scaffold/rules/existing.md" in result.files_changed


def test_apply_edit_rule_rejects_missing(tmp_path: Path) -> None:
    root = _bare_scaffold(tmp_path)
    action = EditRuleAction(
        target_file="rules/missing.md",
        content="---\nrule_id: missing\napplies_to: runtime\n---\n\n# x\n",
        rationale="oops",
    )
    with pytest.raises(ApplyError, match="does not exist"):
        apply_action(action, root)


def test_apply_edit_skill_identity(tmp_path: Path) -> None:
    root = _bare_scaffold(tmp_path)
    action = EditSkillAction(
        target_file="skills/identity.md",
        content="---\nskill_id: identity\nkind: identity\napplies_to: runtime\n---\n\n# new role\n",
        rationale="update role",
    )
    result = apply_action(action, root)
    text = (root / "skills" / "identity.md").read_text(encoding="utf-8")
    assert "new role" in text
    assert "scaffold/skills/identity.md" in result.files_changed


# ---------------------------------------------------------------------------
# change_sampling
# ---------------------------------------------------------------------------


def test_apply_change_sampling_updates_field(tmp_path: Path) -> None:
    root = _bare_scaffold(tmp_path)
    action = ChangeSamplingAction(
        field="temperature",
        value=0.2,
        rationale="explore-less",
    )
    result = apply_action(action, root)
    harness = _read_yaml(root / "harness.yaml")
    assert harness["sampling"]["temperature"] == 0.2
    assert "0.7" in result.action_summary  # old value captured
    assert "0.2" in result.action_summary  # new value captured


def test_apply_change_sampling_creates_field_if_missing(tmp_path: Path) -> None:
    root = _bare_scaffold(tmp_path)
    # Strip top_p from sampling to simulate it not being present.
    harness = _read_yaml(root / "harness.yaml")
    harness["sampling"].pop("top_p", None)
    (root / "harness.yaml").write_text(yaml.safe_dump(harness, sort_keys=False), encoding="utf-8")

    action = ChangeSamplingAction(field="top_p", value=0.95, rationale="nucleus")
    apply_action(action, root)
    harness = _read_yaml(root / "harness.yaml")
    assert harness["sampling"]["top_p"] == 0.95
