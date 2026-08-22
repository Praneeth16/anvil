# Failure vs Error: what the code actually does today

Read from the installed mlflow 3.11.1 source rather than inferred, so this is
ground truth for the version in `uv.lock`. Phase 2 of the hardening work
implements against it.

## The claim

A **failure** is an expectation that was assessed and not met. An **error** is an
expectation that was never assessed. A failure is a fact about the agent; an
error is a fact about the infrastructure. Conflating them means a rate-limited
gateway and a bad answer move the promotion gate by the same amount, in the same
direction.

## What happens now

### ANVIL's own predict primitive

`eval/runner.py:641-668` (`_run_predictions_parallel`) catches any exception per
row and records the prediction as `""`. Its docstring calls this an "acceptance
contract":

> a prediction that raises is recorded as an empty string and does not abort the
> whole eval

The isolation is right; the representation is wrong. An empty string is not a
neutral value — it is a *very bad answer*, and the judges score it as one, at or
near 0.0. So the contract as written guarantees that infrastructure failures are
laundered into quality signal.

This primitive is not on the live path (the docstring says so, and it is
correct — mlflow's harness runs predict so that the per-row trace carries the
`RETRIEVER` span `RetrievalGroundedness` needs). It is exercised by tests and
offline paths.

### The live path

`mlflow/genai/evaluation/harness.py:771-782`:

```python
try:
    eval_item.outputs = call_with_retry(
        lambda: predict_fn(eval_item.inputs), rate_limiter, max_retries
    )
except Exception as e:
    eval_item.error_message = (
        f"Failed to invoke the predict_fn with {eval_item.inputs}: {e}"
    )
```

Three consequences:

1. **mlflow records the error properly** — `error_message` is set and `outputs`
   stays `None`. The information ANVIL needs exists.
2. **Nothing consults it before scoring.** `error_message` is assigned in exactly
   one place and read in none; the item proceeds to `_run_score` with `None`
   outputs and the judges score the absence of an answer as a wrong answer.
3. **mlflow already retries** via `call_with_retry(..., max_retries)`. ANVIL's
   shim (`_patch_harness`) passes the default `max_retries=0`, so today there is
   one attempt per row.

Scorer errors, by contrast, are handled correctly on both sides: mlflow records
an `AssessmentError(error_code="SCORER_ERROR")` feedback (`harness.py:879-881`),
and ANVIL's `_row_score` finds no numeric `value` and returns `None`, which
`_mean` skips. So a *judge* that crashes is already excluded from the aggregate.
Only a *prediction* that crashes is counted as a zero.

### What the aggregate does with it

`_aggregate_report` (`eval/runner.py:296-395`) prefers mlflow's
`metrics[f"{name}/mean"]` when present and falls back to its own `_mean` over
per-row scores. `EvalReport` has no error count, so a round in which half the
rows failed to execute is indistinguishable, downstream, from a round in which
the agent answered badly. `loop/round.py` then compares that number against the
frontier and keeps or reverts on it.

## Phase 2 implementation

All six steps have landed.

1. **`eval/outcome.py`** — `CaseOutcome` (`ok | failure | error | skipped |
   interrupted`), `Attempt`, `CaseRecord` with the invariants enforced in
   `__post_init__` (an error must name its `error_type`; a scorable outcome may
   not carry one), and `summarize()` → `OutcomeSummary.error_rate`. Pure.
2. **`_run_predictions_parallel` returns `CaseRecord`s**, not `""`. The tests
   that asserted the empty-string "acceptance contract" were rewritten: the
   contract was the defect, so the tests encoding it had to change with it.
   Retries are on error only (below). `RunInterrupted` carries the partial
   records.
3. **`_resilient_eval_harness(error_sink=...)`** captures
   `eval_item.error_message` per row, keyed by `trace_id`. The shim already
   wrapped `_run_predict` for the missing-trace fallback, so this cost no new
   interception point. `trace_id` rather than mlflow's internal `request_id`
   because it is the only row identifier that also appears in `result_df`, and
   the capture happens *after* the minimal-trace fallback so an errored row —
   which creates no span of its own — still has a key.
4. **`EvalReport.n_errors` / `.errors` / `.error_rate`**; errored rows are
   excluded from `per_judge`, `per_bucket`, and the aggregate, and are reported
   as errors rather than as judge failures. mlflow's `{name}/mean` is ignored
   whenever anything errored, since it is computed over every row including
   those. An error that cannot be joined back to a result row still counts
   toward `n_errors` (so the guard fires) and is logged as unattributable.
5. **`eval.max_error_rate` (default `0.2`)** — above it `loop/round.py` marks
   the round `eval_failed`, which short-circuits `gate_decision` to
   `INFRA_FAIL` *before any frontier I/O*. So a degraded gateway can neither
   revert a good mutation nor advance the frontier with a number that was never
   trustworthy. `mutated_score` is still recorded: "0.41, but half the cases
   never ran" is more useful six rounds later than a null.
6. **Exit codes in `anvil/cli.py`** — `0` clean, `1` assessed failures, `2`
   unusable run, `130` interrupted. `run_cli` maps `KeyboardInterrupt` and
   `RunInterrupted`, and lets `SystemExit` through so `--help` still works. The
   1↔2 boundary for an eval is `eval.max_error_rate`, the same number the round
   gate uses, so a red CI run and a reverted round mean the same thing.

Step 3 is the only part that touches mlflow internals, and it extends an
interception this codebase already owns and documents.

### Retry on error only

Retrying an error buys another sample of the *infrastructure*. Retrying a
failure buys another sample of the *agent*, which makes the score a function of
how many attempts were paid for — and a self-optimizing loop that can spend its
way to a better number will. So a returned answer is final however bad it is,
including an empty one, and only a raised exception is tried again. Failed
attempts are retained on the record: a case that only succeeded on its third try
is not the same as one that succeeded immediately, and the difference is a
degrading endpoint.

### The per-case timeout is not where the plan put it

The plan called for a per-case timeout in `_run_predictions_parallel`. What
shipped there is **nothing**, deliberately. A Python thread cannot be cancelled,
so a deadline in that pool would bound only how long we *wait*: the hung request
keeps running, and the pool's shutdown joins it anyway. It would read as a
guarantee while being none, which is worse than its absence.

The layer that can actually abandon the socket is the HTTP client —
`openai.OpenAI(timeout=...)` in `runtime/client.py`, currently the SDK default
of 600s with 2 built-in retries. That is where a per-request deadline belongs.
It changes live behaviour for the runtime agent and the judge, so it is left for
the live lane rather than smuggled in behind an offline test.
