"""Code-mode agent implementations.

In code mode (``harness/config.yaml > mode: code``) ANVIL optimizes
agent Python code — ``MemorySystem`` subclasses with different
retrieval algorithms, learning strategies, and memory structures —
instead of prompt scaffolds. The eval runner imports the active agent
module from this package (or a file path) and benchmarks it against
the golden set.

This is inspired by meta-harness's ``MemorySystem`` pattern.
"""
