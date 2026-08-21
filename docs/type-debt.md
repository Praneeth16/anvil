# Type debt

`mypy --strict` was turned on with 89 errors across 15 of 35 modules. Twenty modules were already
strict-clean, including the gate itself (`loop/frontier.py`, `loop/decision.py`) — good news, since
that is the code whose correctness matters most.

Two override blocks in `pyproject.toml` let the gate land green:

1. **Strictness relaxations** — missing annotations and bare generics. Ordinary debt, no
   behavioural risk, fixed opportunistically.
2. **`disable_error_code`** — real type *mismatches*. Itemised below, because a silenced type error
   with no paper trail is worse than no type checking at all.

Both lists are ratchets: the phase that rewrites a module fixes its errors and deletes its entry.
Never append.

## The mismatches that matter

### `eval/runner.py:481` — the code-mode constructor contract is unenforced

```
error: Unexpected keyword argument "llm_client" for "MemorySystem"  [call-arg]
error: Unexpected keyword argument "model" for "MemorySystem"  [call-arg]
```

`_instantiate_agent` ends in `cls(llm_client=llm_client, model=model)`, but the `MemorySystem` ABC
(`agents/memory_system.py`) declares **no `__init__` at all**. The constructor contract that every
optimizer-authored agent must satisfy exists only in a docstring.

Consequence: in `mode: code`, the optimizer can write a `MemorySystem` subclass whose `__init__`
takes different parameters, and the failure surfaces as a `TypeError` deep inside eval — where
Phase 2's error handling will class it as an infrastructure error and abort the round, when it is
really an invalid candidate that should have been rejected at validation time.

Fix (Phase 5, with `optimizer/code_validation.py`): declare `__init__` on the ABC and check the
candidate's signature against it during validation, so a bad candidate is rejected before eval.

### `runtime/agent.py:59` and `eval/runner.py:745` — `source` is a `Literal` fed a bare `str`

```
error: Incompatible default for parameter "source" (default has type "str",
       parameter has type "Literal['production', 'eval', 'optimizer']")
```

`source` tags every MLflow trace and is the field observability queries filter on. A typo produces
traces that silently match no query. The `Literal` is the right idea; the default and the caller
both bypass it.

Fix: correct the default and narrow at the `runner.py` call site. Cheap, do it in Phase 2 while
that file is open.

### `eval/runner.py:714-715`, `797`, `runtime/agent.py:70` — `GatewayClient` is not an `OpenAI`

```
error: Incompatible types in assignment (expression has type "OpenAI | GatewayClient",
       variable has type "OpenAI | None")
error: Argument "judge_client" to "build_scorers" has incompatible type "OpenAI | None";
       expected "OpenAI"
```

`GatewayClient` is a duck-typed stand-in for `OpenAI` that happens to expose the same
`chat.completions.create` surface. It works, but the annotations claim something false, and
`judge_client` is additionally typed non-optional while `None` reaches it.

Fix (Phase 5, when the provider boundary lands): define an explicit `ChatClient` protocol both
satisfy. This is exactly the seam that phase introduces, so the fix is free there.

### `optimizer/session.py:146` — the permission callback bypasses the typed contract

```
error: Argument "can_use_tool" ... expected Callable[..., Awaitable[PermissionResultAllow
       | PermissionResultDeny]]
```

`_allow_all_tool_calls` returns a raw `{"behavior": "allow", ...}` dict instead of the SDK's typed
`PermissionResultAllow`. It works by accident of the SDK's coercion, and it is the very function
Phase 1 deletes. Fixed there by construction.

### `eval/cache.py:74` — sorting a union that includes `None`

```
error: Argument "key" to "sorted" has incompatible type ... expected SupportsDunderLT
```

The scorer-fingerprint sort key can return `None`, which raises `TypeError` on comparison against
a `str` or `float`. Reachable only if a scorer entry is missing the sorted-on field. The fingerprint
guards gate integrity — a crash here is loud and safe, but it should be impossible by construction.

Fix: make the key total (coerce `None` to a sentinel) when Phase 3 touches the fingerprint.

### `runtime/agent.py:114,143` and `eval/runner.py:779` — `ResponsesAgent` payload shapes

```
error: Argument "output" to "ResponsesAgentResponse" has incompatible type
       "list[dict[str, Any]]"; expected "list[OutputItem]"
error: Module "mlflow.pyfunc" does not explicitly export attribute "ResponsesAgent"
error: Signature of "predict" incompatible with supertype "PythonModel"
```

Hand-built dicts passed where MLflow expects typed `OutputItem`s. MLflow accepts them today. The
`predict` override divergence and the non-exported `ResponsesAgent` are MLflow's own typing gaps,
so these stay silenced with a comment rather than being "fixed" against a moving target.

### `runtime/client.py:152`, `tools/search_knowledge_base.py:98-99` — minor

Untyped message dicts into the OpenAI SDK, and `doc_id`/`title` typed `str` while frontmatter
parsing can yield `None` (a KB doc missing a field produces `None` where `str` is promised).
Low severity; fix in passing.
