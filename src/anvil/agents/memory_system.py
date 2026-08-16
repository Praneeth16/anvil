"""Abstract base class for code-mode agent implementations.

A code-mode candidate is a full Python module that implements this ABC.
FORGE's optimizer writes new subclasses (different retrieval algorithms,
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

from abc import ABC, abstractmethod
from typing import Any


class MemorySystem(ABC):
    """Base class for code-mode agent implementations.

    A code-mode candidate is a full Python module that implements this
    ABC. FORGE's optimizer writes new subclasses (different retrieval
    algorithms, learning strategies, memory structures) and the
    benchmark scores them.

    Inspired by meta-harness's MemorySystem pattern.

    Thread-safety: ``predict`` may be invoked concurrently from multiple
    threads when ``eval.n_workers > 1`` (code-mode parallel eval). A
    subclass must therefore keep ``predict`` free of unsynchronized
    shared-state mutation; stateful learning belongs in
    ``learn_from_batch``, which runs between batches (not per-row).
    """

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
