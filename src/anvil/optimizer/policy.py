"""What the optimizer is allowed to touch, and why.

The optimizer is a Claude Agent SDK session whose output is *graded*. That
makes it an adversary in one specific, non-hypothetical sense: any path from
"edit a file" to "score goes up" is a path it may take, and editing the grader
is the shortest one. It is not malicious. It is an optimizer, which is worse,
because it does not need intent to find the cheat.

The three cheats this module exists to prevent:

1. **Read the answers.** ``data/golden_set.jsonl`` holds ``reference_answer``,
   ``must_include``, and ``notes_for_judge`` for every case -- the answer key
   *and* the judge's rubric. An optimizer that reads it can write a skill that
   hardcodes ``$0.142`` and score perfectly while learning nothing.
   ``code_validation.py`` blocks generated code from *importing* the golden set;
   nothing stopped the session from simply reading it.
2. **Weaken the grader.** ``data/evaluator.py``, ``harness/config.yaml``, and
   ``src/anvil/loop/frontier.py`` decide what "better" means. A mutation that
   lowers the bar is indistinguishable, in the score, from one that clears it.
3. **Edit the referee's notes.** ``eval/`` holds the baseline, the frontier, and
   the round records the next decision is made against.

The contract the loop actually wants is narrow: the optimizer *proposes* an
action as JSON and ``optimizer/applier.py`` performs the writes. Direct file
mutation was never part of that contract -- it was merely possible. This module
makes capability match contract.

Pure and I/O-free apart from path resolution, so the whole policy is unit
testable without an SDK session. :func:`ToolPolicy.verify_changed_paths` is the
same policy applied after the fact to a git diff: the SDK's permission callback
is version-dependent and best-effort, a diff is neither.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Tools the session may call at all. An allowlist, not a denylist: a new tool in
# a future SDK release arrives denied rather than arriving permitted.
#
# Bash is absent deliberately -- it is a general-purpose write primitive
# (`sh -c 'echo ... > file'`) that no path policy can inspect. WebFetch and
# WebSearch are absent because network reads make a round unreproducible and
# are the obvious exfiltration route for anything the session has read. Task is
# absent because a subagent inherits the session's reach without inheriting this
# policy.
ALLOWED_TOOLS: tuple[str, ...] = ("Read", "Glob", "Grep", "Write", "Edit", "MultiEdit")

# Tools that mutate. Their target must fall inside the writable scope.
WRITE_TOOLS: frozenset[str] = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})

# Tools that read. Their target must not be, or contain, a secret.
READ_TOOLS: frozenset[str] = frozenset({"Read", "Glob", "Grep"})

# Where a tool call's target path lives in its input, by tool.
_PATH_KEYS: tuple[str, ...] = ("file_path", "notebook_path", "path")

# Repo-relative directories the optimizer may write to. ``scaffold`` is the
# prompt-mode surface; ``agents`` is the code-mode surface.
DEFAULT_WRITABLE_DIRS: tuple[str, ...] = ("scaffold", "agents")

# Repo-relative paths the optimizer may neither read nor write. Reading these is
# reward hacking; writing them is grader tampering.
DEFAULT_SECRET_PATHS: tuple[str, ...] = (
    "data/golden_set.jsonl",
    "data/evaluator.py",
)


@dataclass(frozen=True)
class PolicyDecision:
    """Allow or deny, with a reason the optimizer can act on.

    The reason is fed back to the model verbatim. It should say what to do
    instead, not merely that the door is shut -- a session that understands the
    boundary spends its turns mutating the scaffold rather than retrying a wall.
    """

    allowed: bool
    reason: str = ""


@dataclass(frozen=True)
class ToolPolicy:
    """Decides tool calls against a writable scope and a secret set.

    ``root`` is resolved on construction so that symlinked roots (macOS
    ``/tmp`` -> ``/private/tmp``, most notably) compare correctly against
    resolved targets.
    """

    root: Path
    writable_dirs: tuple[str, ...] = DEFAULT_WRITABLE_DIRS
    secret_paths: tuple[str, ...] = DEFAULT_SECRET_PATHS
    allowed_tools: tuple[str, ...] = ALLOWED_TOOLS
    _resolved_root: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_resolved_root", Path(self.root).resolve())

    # -- path predicates ---------------------------------------------------

    def resolve(self, raw: str) -> Path:
        """Resolve ``raw`` against the root, following symlinks.

        ``strict=False`` because a Write targets a path that does not exist yet.
        Resolution is what defeats ``../`` traversal and symlink escapes: a link
        inside ``scaffold/`` pointing at ``/etc`` resolves outside the root and
        is denied by :meth:`is_inside_root` like any other outside path.
        """
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self._resolved_root / candidate
        return candidate.resolve()

    def is_inside_root(self, resolved: Path) -> bool:
        return resolved == self._resolved_root or self._resolved_root in resolved.parents

    def is_writable(self, resolved: Path) -> bool:
        """True if ``resolved`` sits under a writable directory."""
        if not self.is_inside_root(resolved):
            return False
        if self.is_secret(resolved):
            return False
        for rel in self.writable_dirs:
            scope = (self._resolved_root / rel).resolve()
            if resolved == scope or scope in resolved.parents:
                return True
        return False

    def is_secret(self, resolved: Path) -> bool:
        return any((self._resolved_root / rel).resolve() == resolved for rel in self.secret_paths)

    def contains_secret(self, resolved: Path) -> bool:
        """True if ``resolved`` is a directory that a secret lives under.

        A directory read is a secret read once the directory is an ancestor of a
        secret: ``Grep`` over the repo root would return matching lines *from*
        the golden set, which is the leak the file-level check would miss.
        Denying the broad read forces the session to name a narrower path, which
        is what it should be doing anyway.
        """
        for rel in self.secret_paths:
            secret = (self._resolved_root / rel).resolve()
            if resolved in secret.parents:
                return True
        return False

    # -- the decision ------------------------------------------------------

    def decide(self, tool_name: str, tool_input: dict[str, Any]) -> PolicyDecision:
        """Allow or deny one tool call."""
        if tool_name not in self.allowed_tools:
            return PolicyDecision(
                False,
                f"{tool_name} is not available to the optimizer. Propose the change as a "
                f"json-action block and let the applier perform it. Available tools: "
                f"{', '.join(self.allowed_tools)}.",
            )

        raw = self._target_path(tool_input)
        if raw is None:
            # No path in the input (e.g. a Grep with no `path`, which defaults to
            # the working directory). Treat the root as the target so a repo-wide
            # read is judged as a repo-wide read rather than waved through.
            raw = str(self._resolved_root)

        resolved = self.resolve(raw)

        if not self.is_inside_root(resolved):
            return PolicyDecision(
                False,
                f"{resolved} is outside the repository. The optimizer works only inside "
                f"{self._resolved_root}.",
            )

        if tool_name in WRITE_TOOLS:
            if self.is_secret(resolved):
                return PolicyDecision(
                    False,
                    f"{self._rel(resolved)} defines how candidates are graded. Editing it "
                    f"would change the meaning of every score in the run, including the "
                    f"comparison this round is judged by.",
                )
            if not self.is_writable(resolved):
                return PolicyDecision(
                    False,
                    f"{self._rel(resolved)} is outside the optimizer's writable scope "
                    f"({', '.join(f'{d}/' for d in self.writable_dirs)}). Mutate the "
                    f"scaffold instead.",
                )
            return PolicyDecision(True)

        if tool_name in READ_TOOLS:
            if self.is_secret(resolved):
                return PolicyDecision(
                    False,
                    f"{self._rel(resolved)} holds the reference answers and judge rubric "
                    f"for the evaluation set. Reading it would let this round score by "
                    f"memorisation instead of by capability, which is the one result the "
                    f"harness cannot use.",
                )
            if self.contains_secret(resolved):
                return PolicyDecision(
                    False,
                    f"{self._rel(resolved)} contains the evaluation answer key, so a read "
                    f"spanning it is denied. Name a narrower path.",
                )
            return PolicyDecision(True)

        # Allowlisted but neither read nor write -- no path semantics to check.
        return PolicyDecision(True)

    # -- post-hoc verification --------------------------------------------

    def verify_changed_paths(self, paths: Iterable[str]) -> tuple[str, ...]:
        """Return the repo-relative paths that the policy would not have allowed.

        Applied to ``git diff --name-only`` after a session, this catches
        anything the permission callback missed -- a tool the SDK routed around
        the callback, a write performed by a mechanism the policy cannot see, or
        simply a future SDK that stops honouring ``can_use_tool``. Empty tuple
        means the round's diff stayed inside its scope.
        """
        return tuple(sorted(p for p in paths if p and not self.is_writable(self.resolve(p))))

    # -- helpers -----------------------------------------------------------

    def _target_path(self, tool_input: dict[str, Any]) -> str | None:
        for key in _PATH_KEYS:
            value = tool_input.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    def _rel(self, resolved: Path) -> str:
        try:
            return str(resolved.relative_to(self._resolved_root))
        except ValueError:
            return str(resolved)
