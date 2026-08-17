"""ANVIL runtime plane.

Knows how to compose a prompt from a scaffold (filtered by
``applies_to``) and answer a request. Does not know about the
optimizer, evaluation, or git.

Public surface (will be filled in Phase 1):

* ``AnvilAgent`` — the ``ResponsesAgent`` subclass.
* ``compose_prompt`` — the rigorous composer that respects
  ``applies_to`` and requires an identity skill.
* ``build_gateway_client`` — the AI Gateway client (sole LLM route);
  ``build_databricks_client`` is a backward-compat wrapper that
  delegates to it.
"""
