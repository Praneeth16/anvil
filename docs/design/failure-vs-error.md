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
5. **`eval/judgeability.py`** — one definition of "may this report be compared
   to anything", consumed by the round gate, the CLI exit status, baseline
   generation, and held-out finalization. It refuses on three grounds, and the
   second and third were found by review *after* the first shipped (see
   "Exclusion is not enough" below). `loop/round.py` marks a refused round
   `eval_failed`, which short-circuits `gate_decision` to `INFRA_FAIL` *before
   any frontier I/O*. So a degraded gateway can neither revert a good mutation
   nor advance the frontier with a number that was never trustworthy.
   `mutated_score` is still recorded: "0.41, but half the cases never ran" is
   more useful six rounds later than a null.
6. **Exit codes in `anvil/cli.py`** — `0` clean, `1` assessed failures, `2`
   unusable run, `130` interrupted. `run_cli` maps `KeyboardInterrupt` and
   `RunInterrupted`, and lets `SystemExit` through so `--help` still works. The
   1↔2 boundary for an eval is `eval.max_error_rate`, the same number the round
   gate uses, so a red CI run and a reverted round mean the same thing.

Step 3 is the only part that touches mlflow internals, and it extends an
interception this codebase already owns and documents.

### Exclusion is not enough, and on one path it made things worse

Excluding errored cases fixes the *direction* of the bias. It does not fix the
*sample*, and code review found three consequences that the first pass missed —
two of them regressions introduced by the exclusion itself.

**An error that cannot be excluded.** The capture is keyed by `trace_id`; an
error whose row is absent from `result_df` has no row to exclude, so its
infrastructure zero stays in the mean. One such error in eight rows sits at an
error rate of 0.125 — under the 0.2 ceiling — so the round would have been judged
normally on a contaminated aggregate: the original bug, quietly reintroduced.
`n_unattributed_errors` now makes the report unjudgeable outright, because a
report that cannot honour its own contract must not be compared.

**No floor on surviving cases.** A rate is relative, so raising
`max_error_rate` to ride out a flaky endpoint also permits the aggregate to
become the score of one surviving row. Seven errors in eight rows used to score
~0.12 and be REVERTED; excluded, the same run can read 1.0 and *extend the
frontier*, becoming the bar every later round must beat. `max_error_rate: 1.0`
therefore reads as "restore the old behaviour" while doing something strictly
more dangerous. `eval.min_scorable_rows` (default 4, capped at the run's own row
count so a small smoke mode stays runnable) is the only instrument that catches
this — no rate can express "at least N cases actually ran".

**The baseline and the finalization were unguarded.** Both are worse than an
unguarded round, and the exclusion is what made them dangerous. A baseline run
that 429'd on six of eight rows used to produce a visibly broken ~0.25 that an
operator would rerun; excluded, it reads the mean of the two survivors —
*higher* than a healthy baseline and indistinguishable from one — and it is the
frontier's seed and the bar for the whole run. Held-out finalization is worse
still: the highest-stakes number the harness produces, run once, and write-once,
so a degraded run locks in until someone deletes the file by hand. Both now
refuse before writing. `CachedBaseline` also records `n_errors`, additively, so a
cached bar can be re-read later to tell a good one from a lucky one.

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

That absence has a visible consequence on the interrupt path, which review
caught: after a Ctrl-C the in-flight rows are waited for, and that wait is
bounded only by the 600s client default. An operator facing an apparent hang
will press Ctrl-C again, and a second interrupt raising out of `shutdown` would
discard every record collected — breaking the one guarantee `RunInterrupted`
makes. So the wait is announced on stderr and a second interrupt abandons it and
still returns the records. The abandoned threads run on until their own sockets
time out; Python cannot cancel them, but they no longer hold the evidence
hostage.

### What is *not* on the live path

`_run_predictions_parallel` has no production caller: the live eval delegates
predict parallelism to mlflow's own pool so that each row gets a trace carrying
the `RETRIEVER` span `RetrievalGroundedness` needs. So the retry-on-error-only
policy and `RunInterrupted` are real but currently exercised by tests and
offline paths only, and `run_cli`'s `RunInterrupted` branch is dead on the live
path. The live protections are the ones in steps 3-5: mlflow's own retries
(`call_with_retry`, currently `max_retries=0`), the error capture, and the
judgeability guards.
