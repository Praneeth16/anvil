"""Abstract base class for code-mode agent implementations.

A code-mode candidate is a full Python module that implements this ABC.
ANVIL's optimizer writes new subclasses (different retrieval algorithms,
learning strategies, memory structures) and the benchmark scores them.

The contract separates **prediction** (before ground truth) from
**learning** (after a batch completes, with ground truth). This mirrors
the train/eval split in meta-harness's ``MemorySystem`` pattern: the
agent must commit to an answer before seeing whether it was right, then
optionally learns from the outcome.

State serialization (``get_state`` / ``set_state``) lets the loop
checkpoint and reproduce a candidate's memory across eval runs.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from types import ModuleType
from typing import Any

from anvil.runtime.client import ChatClient


class MemorySystem(ABC):
    """Base class for code-mode agent implementations.

    A code-mode candidate is a full Python module that implements this
    ABC. ANVIL's optimizer writes new subclasses (different retrieval
    algorithms, learning strategies, memory structures) and the
    benchmark scores them.

    Inspired by meta-harness's MemorySystem pattern.

    Thread-safety: ``predict`` may be invoked concurrently from multiple
    threads when ``eval.n_workers > 1`` (code-mode parallel eval). A
    subclass must therefore keep ``predict`` free of unsynchronized
    shared-state mutation; stateful learning belongs in
    ``learn_from_batch``, which runs between batches (not per-row).
    """

    def __init__(self, *, llm_client: ChatClient | None = None, model: str = "") -> None:
        """Accept the kwargs the eval instantiates every candidate with.

        ``anvil.eval.runner._load_memory_system`` calls
        ``cls(llm_client=..., model=...)``. That contract used to live only in a
        docstring, so an optimizer-authored subclass with a different signature
        was accepted by validation and failed as a ``TypeError`` deep inside the
        eval -- where judgeability reads it as an infrastructure failure and
        aborts the round, rather than as the invalid candidate it is. Declaring
        it here makes the mismatch checkable before any money is spent; see
        ``anvil.optimizer.code_validation.check_constructor_contract``.

        A subclass that needs no construction arguments now inherits a working
        constructor instead of having to restate one.
        """
        self.llm_client = llm_client
        self.model = model

    @abstractmethod
    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        """Run a prediction BEFORE seeing ground truth.

        Returns ``(answer, metadata)`` where metadata tracks cost info
        like ``context_chars``, ``tokens_used``, etc.
        """
        ...

    @abstractmethod
    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        """Learn AFTER a batch completes, from results WITH ground truth.

        Can be a no-op for no-memory baselines.
        """
        ...

    def get_state(self) -> str:
        """Serialize memory state for reproducibility. Default: empty."""
        return "{}"

    def set_state(self, state: str) -> None:  # noqa: B027
        """Restore memory state. Default: no-op."""
        pass


def find_memory_system_subclass(module: ModuleType) -> type[MemorySystem]:
    """Find the concrete ``MemorySystem`` subclass defined in ``module``.

    The class must be *defined* in this module (``__module__`` match) so
    that a re-exported base class or an imported helper does not get
    mistaken for the agent. Exactly one subclass is expected; zero or
    multiple are configuration errors.
    """
    candidates: list[type[MemorySystem]] = []
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, MemorySystem)
            and obj is not MemorySystem
            and getattr(obj, "__module__", None) == module.__name__
            and not inspect.isabstract(obj)
        ):
            candidates.append(obj)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            f"no concrete MemorySystem subclass found in agent module {module.__name__!r}"
        )
    raise ValueError(
        f"multiple concrete MemorySystem subclasses found in {module.__name__!r}: "
        f"{[c.__name__ for c in candidates]}"
    )
