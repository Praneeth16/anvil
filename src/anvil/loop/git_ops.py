"""Thin wrappers over ``git`` for round orchestration.

The loop is the only plane that knows about git. Everything else
(runtime, eval, optimizer) is git-agnostic. Keeping these wrappers
small + checked makes the loop's intent legible and the failure
modes explicit.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class GitError(RuntimeError):
    """Raised when a git operation returns non-zero."""


@dataclass(frozen=True)
class GitResult:
    stdout: str
    stderr: str


def _run(repo_root: Path | str, args: list[str], *, check: bool = True) -> GitResult:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if check and proc.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed (exit {proc.returncode}):\n"
            f"  stdout: {proc.stdout.strip()}\n"
            f"  stderr: {proc.stderr.strip()}"
        )
    return GitResult(stdout=proc.stdout.strip(), stderr=proc.stderr.strip())


def current_branch(repo_root: Path | str) -> str:
    return _run(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"]).stdout


def current_sha(repo_root: Path | str) -> str:
    return _run(repo_root, ["rev-parse", "HEAD"]).stdout


def has_changes(repo_root: Path | str) -> bool:
    """True if there are tracked or untracked changes in the working tree."""
    res = _run(repo_root, ["status", "--porcelain"], check=False)
    return bool(res.stdout)


def check_clean_worktree() -> None:
    """Raise if the Git working tree containing the current directory is dirty."""
    status = _run(Path.cwd(), ["status", "--porcelain"])
    if status.stdout:
        raise RuntimeError(
            "Working tree is dirty. Commit or stash your changes before running an "
            "optimization round. Uncommitted files:\n"
            f"{status.stdout}"
        )


def create_round_branch(
    repo_root: Path | str, round_id: int, *, parent_branch: str = "anvil/exp"
) -> str:
    """Create ``anvil/exp-round-<N>`` from ``parent_branch`` and check it out.

    Idempotent on a clean tree: if the branch already exists, raises
    ``GitError`` (the loop should not silently reuse a stale branch).
    """
    branch = f"anvil/exp-round-{round_id}"
    _run(repo_root, ["checkout", parent_branch])
    _run(repo_root, ["checkout", "-b", branch])
    return branch


def commit_all(repo_root: Path | str, *, message: str) -> str:
    """Stage ``scaffold/`` (and ``agents/`` when present) and commit.

    Returns the new SHA, or the current SHA when there is nothing to
    commit (e.g. a ``noop`` action or content identical to HEAD).

    In code mode the optimizer writes agent modules under ``agents/``
    (a repo-root sibling of ``scaffold/``); staging only ``scaffold/``
    would leave those mutations uncommitted and break the round's
    keep/revert branch operations. ``git add`` on a missing pathspec is
    fatal, so ``agents/`` is staged only when the directory exists.
    """
    _run(repo_root, ["add", "scaffold/"])
    if (Path(repo_root) / "agents").is_dir():
        _run(repo_root, ["add", "agents/"])
    if not has_changes(repo_root):
        # Nothing to commit (e.g. noop action or applier wrote identical content).
        return current_sha(repo_root)
    _run(repo_root, ["commit", "-m", message])
    return current_sha(repo_root)


def ff_merge(repo_root: Path | str, *, branch: str, target: str = "anvil/exp") -> None:
    _run(repo_root, ["checkout", target])
    _run(repo_root, ["merge", "--ff-only", branch])


def delete_branch(repo_root: Path | str, *, branch: str, target: str = "anvil/exp") -> None:
    _run(repo_root, ["checkout", target])
    _run(repo_root, ["branch", "-D", branch])
