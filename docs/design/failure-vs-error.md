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

## What the first live run against a real workspace found

Everything above passed 485 offline tests and then failed three times in a row
against `fe-vm-lakebase-praneeth`. Recorded because the *reason* the offline
suite was blind generalises.

`_resilient_eval_harness`'s fallback for a missing per-row trace is
`create_minimal_trace`, and that function ends in:

```python
return mlflow.get_trace(root_span.trace_id)
```

It depends on the very retrieval it exists to compensate for. Against a local
file store that always succeeds, so the fallback always works and every offline
test passes. Against the Databricks Tracing Server it returns `None` —
persistently, not transiently: the live run logged *"no retrievable trace after 3
fallback attempts"*. So nothing was fixed, and the crash relocated to the next
unguarded dereference of `eval_item.trace`.

It relocated to exactly the three places `_resilient_eval_harness`'s own
docstring had named in advance, which is worth sitting with — the failure modes
were documented and still unguarded:

| Site | Symptom |
|---|---|
| `batch_link_traces_to_run` (`trace_utils.py`:1014) | unguarded list comprehension; killed the run *after* 8 predictions and 24 judge calls were paid for |
| `construct_eval_result_df` (`trace_utils.py`:925) | swallows the error, returns `None`, which reaches `_aggregate_report` as `len(None)` |
| `mlflow.start_span` + `@trace_disabled` | a prediction failure thrown *into* the span's context manager emerged as `RuntimeError: generator didn't stop after throw()`, replacing a real 404 |

The third was a Phase 2 failure specifically: mlflow records `error_message` from
whatever escapes `predict_fn`, so the evidence would have recorded generator
plumbing instead of the endpoint failure. `_traced_predict` now catches the body's
exception, closes the span normally, and re-raises after — so `predict_fn` still
raises, and what it raises is real.

**And the fix re-created the bug it was fixing.** A row dropped for want of a
trace is not an *error* — the prediction succeeded — so `error_rate` never sees
it, and it is absent from `n_rows`. With the sample-size floor capped at
`n_rows`, losing six of eight rows leaves `error_rate` at `0.0` and a floor that
shrank to 2 along with the sample: no guard fires. Exactly the silent sample loss
that excluding errored rows introduced, one layer down. Hence `n_dropped_rows`
and `n_attempted`, with the floor computed against the latter.

mlflow's pre-flight `check_model_prediction` is now skipped (scoped and
restored). It is what masks the failure above, and it then aborts the whole run —
so the row never becomes an error record and none of the Phase 2 guards engage.
Skipping it lets the failure land on row 1, where it is captured as evidence. Its
other job, auto-wrapping an untraced `predict_fn`, is redundant against the root
span `evaluate_branch` already opens.

### Live verification

Against `fe-vm-lakebase-praneeth`, runtime and judge on
`databricks-claude-sonnet-4-6`:

| Case | Result |
|---|---|
| Healthy, `--mode quick` | exit **0**, 8/8 rows, `n_errors=0`, `n_dropped_rows=0`, `n_unattributed_errors=0` |
| Runtime endpoint set to a nonexistent model | exit **2**, `n_errors=8`, refused with `unmeasured rate 1.00 exceeds ceiling 0.20 (8/8 cases never scored: 8 errored)`; each `error_message` carries the real `404 RESOURCE_DOES_NOT_EXIST` |

The load-bearing number in the fault case is **`n_unattributed_errors=0`**: the
`trace_id` join between the shim's capture and `result_df` held for all eight
errored rows against the real tracing server. That was the one assumption the
offline tests could only check against a local file store.

### The healthy aggregate is not stable, and that is the Phase 3 argument

Two healthy runs of the **same 8 rows**, same scaffold, same model, no change
affecting scoring:

| Run | aggregate | correctness | groundedness | refusal |
|---|---|---|---|---|
| first | 0.875 | 0.625 | 1.000 | 1.000 |
| later | 0.722 | 0.500 | 0.667 | 1.000 |

**~0.15 of aggregate swing from judge noise alone**, on the row count the loop
actually uses per round. The gate promotes on any positive delta
(`gate.epsilon: 0.0`), so it would read that swing as a real improvement or a
real regression about as often as not. This is the empirical case for the paired,
noise-aware gate in Phase 3 — measured here rather than assumed.

The second run also logged `retrieval_groundedness: 4/8` scorer invocations
failing. Scorer errors are already excluded rather than scored 0.0, so the
aggregate is not corrupted by them — but half the groundedness judges failing is
a separate defect, not noise, and is not addressed here.

**Chased in `scorer-applicability.md`.** It was three defects, and the first run's
`groundedness 1.000` in the table above is the evidence: that 1.000 was a mean
over 4 rows. Excluding scorer errors turns out to reopen this document's own hole
one level down — the per-judge means had no sample floor — and the exclusion was
also paying the agent not to retrieve.
