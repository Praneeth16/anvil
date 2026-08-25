"""MLflow tracing helpers — shared across runtime, eval, optimizer.

Two responsibilities:

* :func:`enable_runtime_tracing` — flip on
  :func:`mlflow.openai.autolog` so the OpenAI
  ``chat.completions.create`` call inside ``AnvilAgent.predict``
  becomes a ``CHAT_MODEL`` sub-span under the ``predict`` root span.
  Idempotent.

* :func:`tag_current_trace` — attach the standard ANVIL operational
  tags (``source``, ``runtime_endpoint``, ``scaffold_branch``,
  ``scaffold_commit_sha``, optional ``round``) to the active trace.
  Every runtime trace is queried by these tags from the optimizer
  side, so they are not cosmetic.

The git-SHA tag intentionally substitutes for an MLflow Prompt
Registry link: the scaffold commit pinpoints the exact mutable state
that produced the response, and ``git checkout <sha>`` reproduces it
byte-for-byte.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Final, Literal

import mlflow
import mlflow.openai
from mlflow.exceptions import MlflowException

# The tag every trace carries, and the field observability queries filter on.
# The constants are annotated with the Literal rather than left as bare ``str``
# so a typo is a type error at the call site: an unrecognised source produces
# traces that match no query, and nothing else in the system notices.
SourceTag = Literal["production", "eval", "optimizer"]
SOURCE_PRODUCTION: Final[SourceTag] = "production"
SOURCE_EVAL: Final[SourceTag] = "eval"
SOURCE_OPTIMIZER: Final[SourceTag] = "optimizer"

_DETACHED_HEAD_LITERAL = "HEAD"

# Env-var fallbacks for runtime contexts where the scaffold lives at
# a path that is not a git repo (Databricks Apps containers,
# Volume-synced multi-pod). Populated by the deploy wrapper from
# ``git rev-parse`` at deploy time so the SHA tag still pins the
# exact scaffold the optimizer needs to query.
_ENV_SCAFFOLD_COMMIT_SHA = "ANVIL_SCAFFOLD_COMMIT_SHA"
_ENV_SCAFFOLD_BRANCH = "ANVIL_SCAFFOLD_BRANCH"


def enable_runtime_tracing() -> None:
    """Enable ``mlflow.openai.autolog`` so chat completions become CHAT_MODEL spans.

    Idempotent. Caller is responsible for setting the tracking URI and
    experiment beforehand. Must be invoked **before** any OpenAI
    client makes its first ``chat.completions.create`` call (autolog
    patches the method on import-time symbols of
    ``openai.resources.chat.completions``).
    """
    mlflow.openai.autolog()


def tag_current_trace(
    *,
    source: SourceTag,
    scaffold_root: Path | str,
    runtime_endpoint: str,
    round: int | None = None,
) -> None:
    """Attach ANVIL's standard operational tag set to the active trace."""
    tags: dict[str, str] = {
        "source": source,
        "runtime_endpoint": runtime_endpoint,
    }
    branch = _resolve_scaffold_branch(scaffold_root) or _env(_ENV_SCAFFOLD_BRANCH)
    if branch is not None:
        tags["scaffold_branch"] = branch
    sha = _resolve_scaffold_commit_sha(scaffold_root) or _env(_ENV_SCAFFOLD_COMMIT_SHA)
    if sha is not None:
        tags["scaffold_commit_sha"] = sha
    if round is not None:
        tags["round"] = str(round)

    try:
        mlflow.update_current_trace(tags=tags)
    except MlflowException:
        return


def _resolve_scaffold_commit_sha(scaffold_root: Path | str) -> str | None:
    return _git_call(scaffold_root, ["rev-parse", "HEAD"])


def _resolve_scaffold_branch(scaffold_root: Path | str) -> str | None:
    branch = _git_call(scaffold_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    if branch == _DETACHED_HEAD_LITERAL:
        return None
    return branch


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def _git_call(scaffold_root: Path | str, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(scaffold_root), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    return out or None
