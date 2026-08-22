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

1. `eval/outcome.py` — `CaseOutcome` (`ok | failure | error | skipped |
   interrupted`) and a per-case record carrying `case_id`, outcome, error type,
   redacted message, attempt count, and duration. Pure and unit-testable.
2. Change `_run_predictions_parallel` to return those records instead of `""`,
   and update the tests that currently assert the empty-string contract — the
   contract is the defect, so the tests encoding it have to change with it.
   Retry on error only, never on a returned failure; add a per-case timeout.
3. Extend the existing `_patch_harness` shim to capture `eval_item.error_message`
   per row into a side-channel. The shim already wraps `_run_predict` for the
   missing-trace fallback, so this costs no new interception point.
4. `EvalReport` gains `n_errors` and an `error_rate`; errored rows are excluded
   from `per_judge` and the aggregate rather than contributing zeros. This means
   preferring ANVIL's own `_mean` over mlflow's `{name}/mean` whenever any row
   errored, since mlflow's mean includes them.
5. `loop/round.py` fails the round as `INFRA_FAIL` when `error_rate` exceeds a
   configured ceiling, so a degraded gateway can never produce a revert. Reuse
   the Phase 1 wiring: a round that cannot be judged does not reach the gate.
6. Interrupt handling: SIGINT stops new cases, lets in-flight ones finish, marks
   the run `interrupted`, leaves it readable. Exit codes `0/1/2/130`.

Step 3 is the only part that touches mlflow internals, and it extends an
interception this codebase already owns and documents.
