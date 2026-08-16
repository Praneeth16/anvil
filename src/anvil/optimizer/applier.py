"""Apply an :class:`OptimizerAction` to the on-disk scaffold.

The optimizer plane never writes files itself — it returns a structured
action and the loop's applier writes it. This separation is what makes
the optimizer mockable (loop tests inject a fake action) and the loop
auditable (every write goes through one place that can lint, validate,
and git-add).

For ``add_*`` and ``edit_*``, this module writes the markdown file under
``scaffold/`` and registers it in ``scaffold/harness.yaml`` if it isn't
already (``add_*`` only). For ``change_sampling``, this module updates
``scaffold/harness.yaml > sampling.<field>``. For ``noop``, no-op.

Returns an :class:`ApplyResult` listing the files touched + an
``action_summary`` that goes into the mutations Delta row's
``diff_summary``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from anvil.optimizer.actions import (
    AddRuleAction,
    AddSkillAction,
    ChangeSamplingAction,
    DeleteRuleAction,
    DeleteSkillAction,
    EditRuleAction,
    EditSkillAction,
    NoopAction,
    OptimizerAction,
)


@dataclass
class ApplyResult:
    files_added: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    files_removed: list[str] = field(default_factory=list)
    action_summary: str = ""


class ApplyError(Exception):
    """Raised when the action's pre-conditions are violated.

    Examples: edit_* targeting a non-existent file; add_* targeting a
    file that already exists; change_sampling on a value Pydantic
    didn't already constrain.
    """


def apply_action(action: OptimizerAction, scaffold_root: Path | str) -> ApplyResult:
    """Apply ``action`` to ``scaffold_root``. Returns paths touched."""
    root = Path(scaffold_root)

    if isinstance(action, NoopAction):
        return ApplyResult(action_summary=f"noop: {action.rationale}")

    if isinstance(action, AddSkillAction):
        return _apply_add_file(
            root, role="skill", target=action.target_file, content=action.content,
            rationale=action.rationale,
        )
    if isinstance(action, EditSkillAction):
        return _apply_edit_file(
            root, role="skill", target=action.target_file, content=action.content,
            rationale=action.rationale,
        )
    if isinstance(action, DeleteSkillAction):
        return _apply_delete_file(
            root, role="skill", target=action.target, rationale=action.rationale,
        )
    if isinstance(action, AddRuleAction):
        return _apply_add_file(
            root, role="rule", target=action.target_file, content=action.content,
            rationale=action.rationale,
        )
    if isinstance(action, EditRuleAction):
        return _apply_edit_file(
            root, role="rule", target=action.target_file, content=action.content,
            rationale=action.rationale,
        )
    if isinstance(action, DeleteRuleAction):
        return _apply_delete_file(
            root, role="rule", target=action.target, rationale=action.rationale,
        )
    if isinstance(action, ChangeSamplingAction):
        return _apply_change_sampling(
            root, field_name=action.field, value=action.value, rationale=action.rationale,
        )

    raise ApplyError(f"unknown action type: {type(action).__name__}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_add_file(
    root: Path, *, role: str, target: str, content: str, rationale: str,
) -> ApplyResult:
    path = root / target
    if path.exists():
        raise ApplyError(
            f"{role} '{target}' already exists; use edit_{role} instead"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_ensure_trailing_newline(content), encoding="utf-8")

    # Register in harness.yaml if not already present.
    harness_path = root / "harness.yaml"
    harness = _load_yaml(harness_path)
    list_key = "skills" if role == "skill" else "rules"
    entries: list[dict] = list(harness.get(list_key) or [])
    filename = target.split("/", 1)[1]  # strip "skills/" or "rules/"
    if not any((e.get("file") == filename) for e in entries if isinstance(e, dict)):
        entries.append({"file": filename})
        harness[list_key] = entries
        _dump_yaml(harness_path, harness)
        files_changed = [str(harness_path.relative_to(root.parent))]
    else:
        files_changed = []

    return ApplyResult(
        files_added=[f"scaffold/{target}"],
        files_changed=files_changed,
        action_summary=f"add_{role} {target}: {rationale[:120]}",
    )


def _apply_edit_file(
    root: Path, *, role: str, target: str, content: str, rationale: str,
) -> ApplyResult:
    path = root / target
    if not path.is_file():
        raise ApplyError(
            f"{role} '{target}' does not exist; use add_{role} instead"
        )
    path.write_text(_ensure_trailing_newline(content), encoding="utf-8")
    return ApplyResult(
        files_changed=[f"scaffold/{target}"],
        action_summary=f"edit_{role} {target}: {rationale[:120]}",
    )


def _apply_delete_file(
    root: Path, *, role: str, target: str, rationale: str,
) -> ApplyResult:
    path = root / target
    if not path.is_file():
        raise ApplyError(f"{role} '{target}' does not exist and cannot be deleted")

    if role == "skill" and _frontmatter(path).get("kind") == "identity":
        raise ApplyError(f"cannot delete identity skill '{target}'")

    harness_path = root / "harness.yaml"
    harness = _load_yaml(harness_path)
    list_key = "skills" if role == "skill" else "rules"
    entries = list(harness.get(list_key) or [])
    filename = target.split("/", 1)[1]
    remaining = [
        entry for entry in entries
        if not (isinstance(entry, dict) and entry.get("file") == filename)
    ]
    harness[list_key] = remaining
    _dump_yaml(harness_path, harness)
    path.unlink()

    return ApplyResult(
        files_changed=[str(harness_path.relative_to(root.parent))],
        files_removed=[f"scaffold/{target}"],
        action_summary=f"delete_{role} {target}: {rationale[:120]}",
    )


def _apply_change_sampling(
    root: Path, *, field_name: str, value: float | int | str | None, rationale: str,
) -> ApplyResult:
    harness_path = root / "harness.yaml"
    harness = _load_yaml(harness_path)
    sampling = dict(harness.get("sampling") or {})
    old = sampling.get(field_name)
    sampling[field_name] = value
    harness["sampling"] = sampling
    _dump_yaml(harness_path, harness)
    return ApplyResult(
        files_changed=[str(harness_path.relative_to(root.parent))],
        action_summary=f"change_sampling {field_name}: {old!r} → {value!r}: {rationale[:120]}",
    )


def _load_yaml(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ApplyError(f"{path} did not parse as a YAML mapping")
    return raw


def _dump_yaml(path: Path, data: dict) -> None:
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)
    path.write_text(text, encoding="utf-8")


def _frontmatter(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        return {}
    end = raw.find("\n---", 4)
    if end == -1:
        return {}
    metadata = yaml.safe_load(raw[4:end]) or {}
    return metadata if isinstance(metadata, dict) else {}


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"
