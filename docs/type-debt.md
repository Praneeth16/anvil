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

Last updated: 2026-08-24.

---

## Closed

Measured by deleting the `disable_error_code` block and re-running: **24 errors → 12**, and the 12
that remain are all MLflow's own typing. Four modules left the block outright —
`anvil.eval.cache`, `anvil.optimizer.session`, `anvil.runtime.client`,
`anvil.tools.search_knowledge_base` — and the two that stay now name only the codes they need, so a
new mismatch of any other kind is an error again.

### `GatewayClient` is not an `OpenAI` — now a `ChatClient` protocol

Every annotation said `openai.OpenAI`. Nothing in `src/` has ever constructed one and passed it: the
object is always a `GatewayClient`, a duck-typed stand-in. So the union `OpenAI | GatewayClient` was
not describing a real choice, and silencing it took the checker off the one boundary where a swapped
client is a genuine hazard.

`anvil.runtime.client.ChatClient` is a `Protocol` describing the single call ANVIL makes, and
`GatewayClient` satisfies it structurally. The plan called for "the protocol both satisfy" — that
turns out to be **impossible**, and the reason is worth recording: `openai`'s `create` is overloaded
with a narrow `Iterable[ChatCompletionMessageParam]` and no `**kwargs`, and protocol parameters are
contravariant, so no protocol loose enough to describe a hand-written client can be satisfied by it.
Describing what ANVIL uses is both achievable and the more useful contract.

One real mismatch remained inside `GatewayClient`: caller-supplied plain dicts into `openai`'s
narrowly-typed `messages`. That is a `cast` at the single point where the wire shape meets the SDK's
TypedDicts, not a silenced module.

### `source` is a `Literal` fed a bare `str`

`SOURCE_PRODUCTION` / `SOURCE_EVAL` / `SOURCE_OPTIMIZER` are now `Final[SourceTag]` rather than bare
`str`, so a typo is a type error at the constant rather than a trace that silently matches no
observability query. `tests/test_provider_boundary.py` asserts the constant set and the `Literal`'s
arguments are equal in *both* directions, so adding a tag to one and not the other is caught.

### The code-mode constructor contract is unenforced

`MemorySystem` now declares `__init__(self, *, llm_client: ChatClient | None = None, model: str = "")`
— the call `anvil.eval.runner._load_memory_system` actually makes — and
`anvil.optimizer.code_validation.check_constructor_contract` binds those arguments against the
candidate's signature during validation. A candidate the eval could not have constructed is now
*rejected*, which reverts the round, instead of raising a `TypeError` inside the eval, which
judgeability reads as infrastructure and aborts it.

The same check subsumes a second case that was silently identical: a candidate module with **no**
concrete `MemorySystem` in it. `write_agent` writes the module `agent_module` resolves to, so that
was never valid either; it just failed later and in the more expensive way. The subclass finder moved
from `eval/runner.py` to `agents/memory_system.py` so the optimizer plane can use it without
importing the eval plane.

**A behaviour bug fell out of this.** Prompt mode resolved `runtime_client or build_gateway_client()`;
code mode passed the unresolved *parameter*. Nobody injects a client outside tests, so an ordinary
code-mode round constructed every candidate with `llm_client=None` — `BaselineExtractor` reads that
as "echo the input", so the eval scored a passthrough rather than an agent, and any candidate that
really called the LLM died on an attribute of `None` deep inside the eval.

### `eval/cache.py` — sorting a union that includes `None`

Already fixed before this pass: the key is `lambda s: str(s["name"])`, which is total. The entry was
stale; `anvil.eval.cache` is now strict-clean.

### `optimizer/session.py` — the permission callback bypasses the typed contract

Fixed when the confinement work replaced `_allow_all_tool_calls`, as predicted. Strict-clean.

### `tools/search_knowledge_base.py` — `doc_id` / `title` promise `str`

Two separate `.get()` calls gave the checker nothing to narrow. Bound to locals first. This one was
covered by *no test at all* — a revert of the narrowing passed all 586 tests — so
`tests/test_kb_index.py` now exercises the fallbacks, including a `doc_id` that is present and
unusable (`doc_id: 42`). The golden set references documents *by* `doc_id`, so a document indexed
under `None` is one `expected_doc_ids` can never match.

---

## What is left, and why it stays

All twelve are MLflow's typing rather than ANVIL's, against a moving target.

### `runtime/agent.py` — `arg-type`, `attr-defined`, `override`

```
error: Module "mlflow.pyfunc" does not explicitly export attribute "ResponsesAgent"
error: Signature of "predict" incompatible with supertype "PythonModel"
error: Argument "output" to "ResponsesAgentResponse" has incompatible type
       "list[dict[str, Any]]"; expected "list[OutputItem]"
```

`ResponsesAgent` is absent from `mlflow.pyfunc.__all__`; its `predict` signature diverges from
`PythonModel.predict` by MLflow's own design; and `ResponsesAgentResponse(output=...)` is documented
to accept plain dicts while being annotated `list[OutputItem]`.

### `eval/runner.py` — `assignment`, `attr-defined`, `list-item`

`mlflow.genai.evaluation.harness` exports none of the internals the resilient eval shim must patch
(`batch_link_traces_to_run`, `construct_eval_result_df`, `_run_predict`,
`_get_new_expectations`), and the replacements are deliberately looser than the originals.
`list-item` is `ResponsesAgentRequest(input=[{...}])` — the same dict-vs-typed-item gap as above.

If MLflow tightens or exports these, delete the codes and re-measure. The measurement is one
command: remove the block, run `mypy`.

---

## Remaining strictness relaxations

The first override block (missing annotations, bare generics) still covers thirteen modules. No
behavioural risk; each is deleted by the phase that rewrites its module.
