"""Baseline ``MemorySystem``: direct LLM call with no memory or retrieval.

The simplest code-mode agent. It demonstrates the ``MemorySystem`` ABC
and serves as the starting point that ANVIL's optimizer mutates from.
No retrieval, no learning — just a passthrough to the LLM (or, when no
client is injected, a literal echo of the input, which makes it testable
without any external service).
"""

from __future__ import annotations

from typing import Any

from anvil.agents.memory_system import MemorySystem


class BaselineExtractor(MemorySystem):
    """Baseline: direct LLM call with no memory or retrieval."""

    def __init__(self, llm_client: Any = None, model: str = "") -> None:
        self.llm_client = llm_client
        self.model = model

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        if self.llm_client is None:
            # Passthrough for testing — no external service required.
            return input, {"context_chars": len(input)}
        response = self.llm_client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": input}]
        )
        return response.choices[0].message.content, {"context_chars": len(input)}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        pass  # no learning in baseline
