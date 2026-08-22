# Scorer applicability, scorer errors, and one global switch

Closes the thread `failure-vs-error.md` ends on: `retrieval_groundedness` failing
3–4 of 8 scorer invocations on every healthy live run.

Four consecutive live runs against `fe-vm-lakebase-praneeth` failed 4/8, 3/8,
4/8, 3/8 — never zero, never all — with one message every time:

```
SCORER_ERROR: No retrieval context found in the trace. The RetrievalGroundedness
scorer requires the trace to contain at least one span with type 'RETRIEVER'.
```

Correctness and `refusal_appropriateness` never failed. That asymmetry is the
whole clue: those two read `inputs` and `outputs`, and groundedness reads the
*trace*.

It turned out to be three independent defects, one of which was hiding the other
two, plus a fourth found while writing the tests.

## 1. Two of the eight rows were never meant to retrieve

The golden set has a clean equivalence, verified by cross-tab over all 20 rows:

| category | rows | `should_refuse` | `expected_doc_ids` |
|---|---|---|---|
| `direct` | 6 | false | 1 |
| `multi_hop` | 6 | false | 2–4 |
| `distractor` | 4 | false | 1–2 |
| `out_of_scope` | 4 | **true** | **0** |

`expected_doc_ids` is empty exactly when the row should be refused. And quick
mode is `{rows: 8, buckets: {direct: 2, multi_hop: 2, distractor: 2,
out_of_scope: 2}}` — so **every quick run contains exactly 2 rows where the
agent correctly refuses, never calls `search_knowledge_base`, and emits no
`RETRIEVER` span.** Grounding does not apply to them.

`mlflow.genai.scorers.RetrievalGroundedness` has exactly one behaviour when a
trace has no retriever span: it raises. mlflow catches that into a `Feedback`
carrying a `SCORER_ERROR` and no value, anvil drops valueless rows from the
judge's mean, and the arithmetic comes out right — so a healthy run logged 2
scorer errors indefinitely with nobody able to tell them from real ones.

`prompts/anvil-round.md` had been telling the optimizer since round one that this
scorer is "only computed for in-scope rows with `expected_doc_ids`". It was a
description of intent that no code implemented.

## 2. The same exclusion paid the agent to stop retrieving

The dangerous half. Excluding a row because it has no retriever span does not
distinguish "this row never needed retrieval" from "this row needed retrieval and
the agent skipped it". The second is a finding about the agent, and excluding it
inverts the incentive.

Groundedness is binary, so on the observed live run — 1 of 8 rows grounded:

| behaviour | groundedness |
|---|---|
| retrieve on all 8 | `1/8 = 0.125` |
| retrieve only where already winning | `1/1 = 1.000` |

**An 0.875 swing available purely by searching less** — the largest single lever
in the scoring system, and it rewards the agent for doing less of the thing the
scaffold exists to do. `pareto.enabled` is `false`, so only `aggregate` gates a
round, but the aggregate is the weighted mean of the per-judge values, so the
lever reaches the gate either way.

This is the reward-hacking threat model applied one level down from Phase 1's:
there the optimizer could edit the grader, here it can shrink the grader's
*denominator*. An objective whose denominator the agent controls is gameable
whether or not anyone edits a file.

**Fix.** `_build_groundedness_scorer` wraps the mlflow scorer with the
applicability rule the prompt already promised:

| `expected_doc_ids` | retriever span | result |
|---|---|---|
| empty | — | `None` — not applicable, no assessment recorded |
| non-empty | absent | **`"no"`** — ungrounded, scored like a wrong answer |
| non-empty | present | mlflow's judge decides |

Abstaining now costs exactly what answering wrongly costs, which is the only
arrangement under which "the agent stopped retrieving" cannot read as an
improvement. Applicability is keyed on `expected_doc_ids` rather than
`should_refuse` because that is what grounding is *about*; the two agree in the
shipped golden set but they are separate columns, and a future row could
legitimately expect a refusal to cite policy.

Detection calls mlflow's own `extract_retrieval_context_from_trace` — the same
function `RetrievalGroundedness.__call__` uses — so "the judge will find context"
and "we predicted it would" cannot drift apart. Hand-rolling a
`span_type == RETRIEVER` scan would be a second definition of the same thing.

### The same lever again, one level down: which span gets to decide

`RetrievalGroundedness.__call__` returns **one feedback per retrieval span**, and
`construct_eval_result_df` flattens same-named assessments into a single
`{name}/value` column — last one wins. So the row's grounding score was decided
by whichever search happened to be flattened last, and the agent chooses that: a
final narrow search whose chunks trivially support a closing sentence carries the
row. Same shape as the applicability hole, with the agent picking the score's
*subject* rather than its denominator.

It was also mismeasuring. 6 of the 20 golden rows expect 2–4 documents, so the
agent searches several times and **no single search supports the whole answer** —
judging the complete answer against only the last search's chunks understates
those rows systematically. The legacy baseline's
`multi_hop: {retrieval_groundedness: 0.5}` is consistent with exactly that.

So the wrapper asks the grounding question **once, against the union of every
retrieved chunk**, via the same `judges.is_grounded` and the same
request/response extraction the built-in uses. Adding a span can then no longer
buy a verdict: extra context cannot make a hallucinated claim supported.

Rejected alternatives: `min` across spans fails every multi-hop row (no single
search grounds a multi-hop answer, so it would manufacture failures); `max` is
strictly more gameable than last-wins; "last span only" is what mlflow's own
docstring claims and its code contradicts, and it is the agent-chosen span.

## 3. A global tracing switch inside a concurrent pipeline

Applicability explains 2 failures per run. It does not explain 3 or 4, and it
does not explain why the count *varied* across runs of a deterministic row
selection.

`scorers.py` used to wrap the refusal judge's `chat.completions.create` in
`mlflow.tracing.disable()` / `enable()`, to stop `mlflow.openai.autolog` spawning
an orphan `CHAT_MODEL` trace per row. Local intent, global mechanism:

* `mlflow.tracing.provider.disable()` installs a **process-global**
  `NoOpTracerProvider`. mlflow 3.11.1 has no thread-local form — `trace_disabled`,
  `disable_autologging` and `disable_discrete_autologging` are all global, all
  three checked.
* `mlflow.genai.evaluate` runs **two** pools, `MlflowGenAIEvalPredict` and
  `MlflowGenAIEvalScore`, and submits a score task the moment one row's
  prediction returns. **Scoring runs concurrently with the remaining
  predictions.**

So every judge call blinded the tracer for whatever predictions were in flight,
and the damage landed two ways:

* A prediction **wholly** inside the window registers no trace under the
  `eval_request_id` that `_run_predict` resolves by, so `mlflow.get_trace`
  returns `None` and the row loses its trace. That is the failure
  `_resilient_eval_harness` was built to survive — which makes this a plausible
  root cause of the dropped-trace crashes PR #3 papered over, not just of the
  groundedness errors.
* A prediction **partly** inside it keeps its trace but loses whichever spans
  were emitted meanwhile. Lose the `search_knowledge_base` `RETRIEVER` span and
  groundedness raises while correctness and refusal score the row happily.

Which is exactly the observed signature, and exactly why the count varied.

**Fix.** `mlflow.tracing.context(enabled=False)` around the judge call. It sets a
**ContextVar**, so suppression is confined to the calling thread, and its
docstring states it "does not affect the global tracing state set by
`mlflow.tracing.disable`". Verified rather than trusted: inside the block a span
comes back as `MLFLOW_NO_OP_SPAN_TRACE_ID` while `is_tracing_enabled()` stays
`True`, and a thread started concurrently still gets a real trace id.

**Two wrong turns worth recording, because both looked right.**

*"No scoped suppression exists."* `trace_disabled`, `disable_autologging` and
`disable_discrete_autologging` were each checked and each found process-global,
and "none exists" was concluded from three misses rather than from the API.
`mlflow.tracing.context` was there the whole time. Three negative results are not
a proof of absence.

*The first replacement was `MLFLOW_GENAI_EVAL_ENABLE_SCORER_TRACING`.* It does
remove the global switch — mlflow wraps each scorer in an `EVALUATOR` span, so
the judge's autolog nests rather than orphaning — and it verified clean live. But
it costs 24 extra retained root traces per quick run and adds an **unguarded
`set_trace_tag` server call per scorer**, whose failure can abort scoring. Buying
a new dependency on the tracing server that `_resilient_eval_harness` exists to
survive is the wrong trade, so it was reverted. Turn it on by hand to debug
scorer behaviour.

*And one claim that needed narrowing.* An earlier reading had a stray root trace
"winning the request-ID → trace-ID mapping". `_run_predict` mints an explicit
`eval_request_id` and resolves by it, so a stray root trace from *another row*
cannot steal a row's trace — but "not last-root-wins" is too absolute. The
exporter assigns `_EVAL_REQUEST_ID_TO_TRACE_ID[eval_request_id] = trace_id`, so
multiple roots created inside the *same* prediction context overwrite each other.
That matters here: if the root span is a no-op because tracing was suppressed
mid-prediction, an autolog `CHAT_MODEL` span can become a new root inside that
same context and claim the mapping — yielding a row whose "trace" is a chat
completion with no `RETRIEVER` span. A third route to the observed symptom, and
another reason not to flip the provider.

### The stand-in trace must not be scored

A row whose trace the server will not return gets a synthesized minimal trace so
the run completes. That trace is root-span-only, so it has no `RETRIEVER` span —
and the applicability rule above would score it `"no"`, i.e. **infrastructure
damage recorded as an agent failure**, which is precisely the substitution the
failure-vs-error work exists to prevent. It is also the most likely error to
occur, since a lost trace is what this repo keeps hitting live.

So the harness tags the trace it synthesizes (`anvil.synthesized_trace`, injected
at creation via `mlflow.tracing.context(tags=...)` so it survives mlflow
re-fetching the trace), and the scorer returns *not applicable* on seeing it.
Unmeasured, never scored.

## 4. Every guard was blind to all of it

`construct_eval_result_df` flattens only a feedback's **value** into the
`{name}/value` column. A scorer error and a scorer abstention are therefore
indistinguishable there — both `None`, both dropped from the mean, and neither
visible to `unjudgeable_reason`, which counted prediction errors and dropped rows
only.

That is a hole in Phase 2's own argument. "Exclusion fixes the direction of the
bias but not the sample" applies per judge, and did not reach there. Each
per-judge value is a mean over only the rows that produced a score, so a judge
that broke on all but one row contributed that row as its verdict on the run —
with every prediction succeeding, every row in the frame, and every run-level
guard quiet. The live evidence is already in `failure-vs-error.md`: a run
reporting `groundedness 1.000` that was `4/8`.

**Fix.** `_row_scorer_error` reads the error out of the `assessments` column,
where it does survive; `EvalReport` gains `per_judge_assessed`,
`per_judge_errors` and `scorer_errors`; and `unjudgeable_reason` grows a
**per-judge floor**.

Both are measured against what each judge *attempted* — rows it scored plus rows
it errored on — not against the run's row count. A judge is allowed to decline
rows (groundedness scores 10 of 12 in standard mode by design), so measuring
against `n_rows` would make a correct eval unjudgeable, and a guard that fires on
correct usage gets switched off. A row a judge declines is not evidence of
anything wrong; a row it broke on is.

**Both**, because a floor alone is as blunt one level down as it was one level up.
`min(4, 8)` is cleared by 4 assessed rows, so a judge failing half its
invocations — the exact live symptom that started this — would pass a floor-only
check. The ceiling is the run's `max_error_rate` reused: "how much of this
measurement may be missing" is not a different question per judge.

This is also what makes a per-judge score *safe* to compute over a subset, which
Phase 3 needs before it leans on per-judge numbers.

## 5. Found while testing: NaN in the aggregate

Writing the per-judge counting test produced `aggregate=nan`.

pandas stores a missing entry in an otherwise-numeric column as `NaN`, not
`None`. `_coerce_score` accepted any `float`, so a scorer error on such a column
came back as a **number**: the row counted as assessed, and its NaN went into the
mean — where NaN propagates, taking the judge's mean and the entire weighted
aggregate with it. Every NaN comparison being `False`, a NaN aggregate then fails
the frontier's `>` check silently and reverts the round, or gets written to a
baseline as the bar every later round is compared against.

Live runs never hit it only because the shipped judges return `"yes"`/`"no"`,
which keeps the column `object` dtype and the missing value `None`. Any
float-valued scorer — a programmatic one — exposes it. `_clamp_score` already
rejected non-finite values on the programmatic path for exactly this reason; the
judge path had the same argument and not the same guard.

`_coerce_score` now maps non-finite to `None`, i.e. to "unscored".

## Ordering, and what it costs

**3 before 2.** While the tracing race can strip a `RETRIEVER` span from a row
that did retrieve, the new applicability rule would score that row `0.0` — a
correct rule reading corrupted evidence, and worse than the bug it replaces.

**The scorer fingerprint has to change.** `retrieval_groundedness` now means
something different while every field `compute_scorer_fingerprint` hashes — name,
type, weight, `check_function` — is byte-identical. So a baseline measured under
the old meaning would have stayed "compatible" and gone on being the bar a 50+
round run chases. `SCORER_SEMANTICS_VERSIONS` folds a per-scorer semantics
version into the fingerprint; only versioned scorers carry the key, so bumping
one does not invalidate baselines for configs that do not use it.

Consequences, which are deliberate and not side effects:

* the cached baseline must be regenerated;
* the persisted frontier must be reseeded;
* aggregates from rounds 1–5 are no longer comparable to later ones.

## Verification

Offline: 29 tests in `tests/test_groundedness_applicability.py`, covering each
branch of the applicability rule, the reward-hacking direction explicitly, the
synthesized-trace escape hatch *and* that it does not swallow that guard, the
span-union behaviour (and that the judge runs exactly once per row regardless of
span count), the thread-scoping of the judge's tracing suppression asserted from a
second thread started *inside* the judge call, scorer-error extraction, the
per-judge floor and ceiling including the cases where they must **not** fire, the
NaN coercion, and that the gate and `is_compatible` cannot drift apart again.

Live, against `fe-vm-lakebase-praneeth`:

| | before | after |
|---|---|---|
| groundedness scorer errors | 3–4 of 8, every run | **0** (3 runs) |
| groundedness rows scored | 4–5 of 8 | **6 of 8** quick / **10 of 12** standard — the `out_of_scope` rows abstain |
| `n_dropped_rows` | intermittent | **0** |
| exit | 0 | 0 |

Fault injection (`runtime_endpoint` pointed at a nonexistent model) still exits
**2** with `unmeasured rate 1.00 exceeds ceiling 0.20 (8/8 cases never scored: 8
errored)`, so none of this weakened the Phase 2 refusal path.

**On causation, honestly.** The applicability rule accounts for exactly 2 of the
3–4 failures, deterministically. The remaining 1–2 disappeared when the global
tracing switch did, across three runs — which is strong evidence rather than
proof: the race is probabilistic, and both changes shipped together. What can be
said precisely is that the structural cause is proven by construction and the
residual is consistent with the race and with nothing else identified. A
row-level timing trace would settle it; nobody has one.
