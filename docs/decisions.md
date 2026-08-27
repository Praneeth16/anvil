# Architectural decisions

The decisions the code already embodies, with the reasoning that produced
them. Each one names what it rules out, because that is the part that gets
forgotten and re-proposed.

These are numbered and stable. `scaffold/README.md` cites D8 by number;
`CLAUDE.md` treats this file as the architectural reference. Append new
decisions; do not renumber existing ones.

Design documents that argue a single decision in depth live in
`docs/design/`. This file is the index and the short form.

---

## D1 — One LLM SDK: `openai`, pointed at Databricks

**Decision.** Every LLM call in the runtime goes through the `openai` SDK
against a Databricks serving endpoint, via
`src/anvil/runtime/client.py`. Model choice is configuration
(`runtime_endpoint`, `judge_endpoint`, `optimizer_endpoint` in
`harness/config.yaml`), never code.

**Why.** The harness exists to measure the effect of changing an agent's
scaffold. Anything that varies between rounds and is not the scaffold is
noise in that measurement. A second provider SDK would put model identity
into code, where a round could change it and the resulting score delta would
mean nothing.

**Rules out.** Adding `anthropic`, `google-generativeai`, `cohere` or any
other provider SDK to runtime dependencies. The optimizer path is the one
exception and it is confined to the `optimizer` extra, because the Claude
Agent SDK is the optimizer *driver*, not a runtime LLM route.

---

## D2 — `ResponsesAgent`, not `ChatAgent`

**Decision.** The runtime agent is an `mlflow.pyfunc.ResponsesAgent`
(`src/anvil/runtime/agent.py`).

**Why.** It is the current MLflow agent interface at Databricks and the one
that carries tool-call and trace structure the eval plane depends on.
`ChatAgent` is legacy-ish there.

**Rules out.** Proposing `ChatAgent`, or a bare callable, as the runtime
interface.

---

## D3 — No agent framework inside the runtime

**Decision.** The tool-calling loop is plain Python. No LangGraph, no DSPy,
no CrewAI, no LlamaIndex.

**Why.** This is the decision the whole project rests on. The optimizer's job
is to read the agent's scaffold and rewrite it. A framework moves behaviour
out of the files the optimizer can see and into library internals it cannot,
so the thing being optimized stops being legible to the thing doing the
optimizing. A prompt-graph abstraction is exactly the wrong shape here.

**Rules out.** Any framework dependency in the runtime path, including "just
for the tool loop".

---

## D4 — Scaffold in files, operational state in Delta

**Decision.** The mutable agent — skills, rules, memory, sampling, tool
registry — is markdown and YAML under `scaffold/`, git-tracked, synced to the
`anvil.default.scaffold` UC Volume on deploy. Delta holds only operational
artifacts: the append-only round log `anvil.default.mutations`, and
MLflow-managed trace Delta (eval aggregates are a view over it, not a
bespoke table).

**Why.** The optimizer mutates the scaffold by editing text. Files are what
an LLM edits well, what `git diff` renders reviewably, and what a human can
inspect after the fact. A scaffold in Delta would give up all three for
nothing — there is no query workload over it. Follows the MiniMax M2.7
reference pattern, which is an external reference and is **not vendored here** --
cite this decision rather than a `research/` path, which does not exist in the
repository.

**Rules out.** A Delta table for scaffold content. Lakebase for anything in
current scope — reconsider only if high-QPS per-conversation runtime memory
is added.

---

## D5 — Keep and revert are git branch operations

**Decision.** Each round is a branch `anvil/round-N`. Keep is a
fast-forward merge; revert is `git branch -D`. Not Delta `RESTORE TABLE`,
not a snapshot directory.

**Why.** The unit of change is a set of file edits, so the right transaction
is a commit. It also means the mutation history is readable with ordinary
tools: `git log` is the round log, and every kept mutation is a reviewable
diff rather than a row describing one.

**Rules out.** Table-level restore as the revert mechanism; copying
directories to snapshot a round.

---

## D6 — Five planes, one-way imports

**Decision.** Runtime, Eval, Optimizer, Loop, Observability are physically
separate packages under `src/anvil/`, and the import direction is enforced:
the runtime never imports the optimizer; the eval never imports git; the loop
is the only orchestrator.

**Why.** Each plane has a different reason to change and a different blast
radius. The runtime is deployed; the optimizer is not. The eval must be
runnable with no git repo present. Collapsing any two makes it impossible to
say what a round actually changed.

**Consequence in practice.** `anvil.eval.cache` imports `ScorerConfig` from
`anvil.runtime.models`, so config types flow eval-ward, never the reverse.
When the judge's domain text became configurable, its defaults stayed in
`anvil.eval.scorers` and the config fields were typed `str | None` — putting
the default strings in `runtime.models` would have inverted that edge.

**Rules out.** A shared `utils` package that every plane imports. Circular
convenience imports.

---

## D7 — Failure and error are different outcomes

**Decision.** "The agent scored badly" and "the eval never ran" are distinct
results, distinguished all the way from `anvil.eval.outcome` up through the
exit codes in `anvil.cli`: `0` measured, `1` measured and below expectation,
`2` not measured, `130` interrupted.

**Why.** A caller that cannot tell a bad score from a broken harness cannot
automate anything on top of it. Concretely: errored cases used to be scored
zero, so a throttled endpoint looked like a quality regression and discarded
good mutations. Excluding them introduced the opposite failure — a run
surviving on two of twelve rows can read *higher* than a healthy one — which
is why judgeability has both a rate ceiling (`eval.max_error_rate`) and an
absolute floor (`eval.min_scorable_rows`). A rate alone cannot express it.

**Full argument.** `docs/design/failure-vs-error.md`.

**Rules out.** Treating any non-zero exit as failure. Scoring an errored case
as zero. Gating on failures by default — see the `anvil.cli` docstring for
why `1` is opt-in and `2` is not.

---

## D8 — The optimizer cannot edit its own grader

**Decision.** `harness/config.yaml` sits outside the optimizer's writable
scope. It holds model identities, the loop's meta-configuration, the eval and
gate thresholds, and the refusal judge's domain text. Everything
mutation-worthy — sampling, skills, rules, tools — lives in
`scaffold/harness.yaml` instead.

**Why.** The optimizer is scored by the eval and rewarded for the score going
up. Anything it can write that affects the score is a shortcut it will
eventually find, and the cheapest available shortcut is never "write a better
agent". Three specific holes this closes: rewriting `runtime_endpoint`
mid-loop invalidates every comparison; rewriting `critique_lookback` is the
loop editing its own recursion depth, which the gate is not designed to
catch; rewriting the judge's prompt is editing the grader directly.

That last one is why `judge_domain_name` and `judge_domain_context` are in
this file even though they are domain content and everything else domain-shaped
moved out to `examples/`. Convenience would put them next to the scaffold.

**Enforced by.** Four independent layers, then a check that does not depend on
the SDK at all: an OS-level `sandbox` (`allowUnsandboxedCommands=False`), an
`allowed_tools` allowlist, and two policy interception points — `can_use_tool`
and a `PreToolUse` hook — followed by post-hoc diff verification. A violation is
an `INFRA_FAIL` reported separately from a flaky endpoint, not a revert. See
`SECURITY.md` for the trust boundaries and `src/anvil/optimizer/policy.py`.

Both interception points call `ToolPolicy.decide`. That is the point rather than
an accident: defence in depth means two *enforcement points* for one rule. Two
copies of the rule is the failure mode D10 records — the gate kept its own copy
of the comparability rule and kept the bug after the eval's copy was fixed — so
`tests/test_optimizer_confinement.py` asserts the two verdicts are identical
across the interesting inputs instead of testing them separately.

**The secret set must follow the domain.** `ToolPolicy`'s denied paths are
matched by exact relative path, which was airtight while the domain was a
constant and silently wrong the moment it became a parameter: a second domain
inside the repo puts its golden set at a path the policy has never heard of, and
the session can then read the reference answers and judge notes for every case
it is about to be graded on. Reads leave no diff, so the post-hoc scope check
cannot catch it. `run_round` therefore builds the secret set from the paths the
round is actually using, keeping the built-in domain's paths alongside them so a
typo cannot unprotect the real answer key.

**Rules out.** Moving any threshold, endpoint, or judge prompt into
`scaffold/`. Trusting scope confinement to prompt instructions alone. Any
hardcoded list of secret paths that does not track the active domain.

---

## D9 — A Pareto frontier gate, not a frozen baseline delta

**Decision.** A mutation is kept only if it improves at least one tracked
objective (each per-judge score, plus the aggregate) without regressing
another by more than `gate.epsilon`. The frontier persists to
`eval/runs/frontier.json`. The legacy `gate.type: delta` behaviour — keep
anything beating the cached baseline — remains available and is not the
default.

**Why.** Against a frozen baseline, a round worse than an earlier *kept*
round still passes as long as it clears the original bar, so quality can
ratchet downward while every individual decision looks correct. Comparing
against best-so-far per objective removes that. Tracking objectives
separately also stops one judge's gain from silently paying for another's
loss inside a single averaged number.

**Rules out.** Gating on the aggregate alone. Re-adding "beats the baseline"
as the default.

---

## D10 — A scorer that does not apply reports nothing, not zero

**Decision.** Applicability is a first-class outcome. `out_of_scope` rows
have no retrieved context to be grounded in, so `retrieval_groundedness`
returns *no score* for them rather than `0.0`, and per-judge aggregates are
means over the buckets where the scorer applies. One definition, in
`anvil.eval.judgeability`, which the gate calls rather than re-deriving.

**Why.** A zero for an inapplicable bucket is a constant subtracted from the
aggregate that reads as a quality regression, and it moves when the *bucket
mix* changes rather than when the agent does. The related trap: a per-judge
score computed over whatever subset the agent happened to produce is not a
score at all, because the agent's own behaviour selects the denominator.

**Comparability.** Because none of this is visible in a scorer's name,
weight, or check function, `anvil.eval.cache` versions scorer *semantics*
(`SCORER_SEMANTICS_VERSIONS`) and folds the version into the baseline
fingerprint. Groundedness is at v4. A baseline from before a semantics change
is refused rather than silently chased. The same argument extends to the
judge's configurable domain text, which is folded in when set.

**Full argument.** `docs/design/scorer-applicability.md`.

**Rules out.** Defaulting an inapplicable score to `0.0` or `1.0`. Any second
copy of the applicability rule — the gate had one once, and it kept the bug
after the eval's copy was fixed.

---

## D11 — The domain is configuration, not library code

**Decision.** A knowledge base, a golden set, a programmatic evaluator module
and a scaffold together constitute a domain, and all four are paths supplied
at the boundary: `--kb-dir`, `--golden-set-path`, `--evaluator-path`,
`--scaffold`. The refusal judge's domain description is two config keys. No
part of `src/` names a domain.

**Why.** The harness's claim is that it optimizes agents, not that it
optimizes one utility-support agent. That claim is only testable if a second
domain can be run without editing library source. `examples/pyloom-docs/` is
that test.

**A baseline records which domain it measured.** Making the domain a parameter
created a new way for two incomparable numbers to look comparable: mode, scorer
names, endpoints and scorer fingerprint are all identical between two domains,
and only the questions differ. So `CachedBaseline` and `EvalReport` carry a
content fingerprint of the knowledge base and golden set, and the gate refuses a
cross-domain comparison the same way it refuses a changed scorer semantics
version. It is checked twice: once before the round spends anything, from local
files, and once after the eval against the fingerprint the eval actually
produced. Absent on either side means unchecked, so baselines written before the
field stay usable.

**Rules out.** Hardcoding a domain name, KB path, or golden-set path in
`src/`. Adding a flag to one entry point and not the others — the held-out
finalization is single-use, so scoring the wrong domain there locks in a
wrong number. Comparing a round against a baseline from another domain.

---

## D12 — Promotion requires a paired test, not a bigger threshold

**Decision.** After the frontier gate says KEEP, the improvement must also clear
a one-sided paired sign test over the **per-row** scores
(`anvil.eval.significance`). `gate.test: paired` is the default; `none` restores
the legacy "any delta clearing epsilon promotes". `gate.replicates` (default `1`)
evaluates each candidate *K* times and averages the per-row scores before the
test runs.

**Why.** Two healthy runs of the *same* scaffold, the same rows and the same
model scored `0.875` and `0.722` — about 0.15 of aggregate, from judge noise
alone. Every per-round gain the loop has actually produced was 0.03–0.06. A gate
that promotes on any positive delta was therefore reading noise as signal about
as often as not, and fifty rounds of that is a random walk with a plausible story
attached.

The obvious fix is the wrong one. Raising `gate.epsilon` to 0.15 rejects every
real gain ever observed here, trading a loop that promotes noise for a loop that
promotes nothing. What makes the difference measurable is that both runs answer
**the same questions**: the scores pair row by row, and a per-row difference
cancels the row-difficulty variance that dominates the aggregate. Only the rows
where the two runs *disagree* carry information — which is a sign test, and it
assumes nothing about the noise distribution, the right posture when the noise
source is an LLM judge nobody has characterised.

**A veto, never a promotion.** The test runs only when the frontier already
decided to keep. It cannot rescue a mutation the frontier rejected: direction
("does this regress an objective") and significance ("is this distinguishable
from noise") are different questions, and a mutation that regressed does not
become acceptable by regressing insignificantly.

**Two ways to fail to conclude, and they get different answers.** No pairable
rows means the *test* could not run — a baseline written before per-row scores
existed. That is the situation an empty `scorer_fingerprint` describes and it
gets the same answer: unchecked, said out loud, frontier decision stands.
Reverting there would be a migration disguised as a gate. Rows that paired but
produced too few disagreements is the opposite: the test ran and reports that
this row count cannot distinguish this mutation from noise. That reverts, and the
reason names `gate.replicates`, which is the knob that buys the power back.

**The cost is explicit, and so is the consequence.** At 12 rows most real
mutations do not flip five rows, so `replicates: 1` will revert many rounds as
underpowered. That is honest rather than pessimistic — the alternative is
promoting noise — and replication is the lever, at exactly proportional spend.

**Activation requires regenerating the baseline.** `eval/runs/baseline.json` as
shipped has no `per_row`, so the gate is inert (and says so, every round) until
`scripts/make_baseline.py` is re-run. Deliberate: back-compat that fails loudly
beats a forced migration, but "inert" must not be quiet.

**Postscript (issues #10/#15).** The baseline has since been regenerated twice
— once on the repaired scorer semantics (2026-08-26), once on the MultiHopRAG
domain — so the gate is operative. And the 12-row constraint above was the
dataset's, not the test's: the golden set had 20 rows and `standard` read 12
of them. The MultiHopRAG migration raised the dev partition to 50 rows, so
`replicates: 1` now detects q=0.65–0.70 effects with power ~0.5–0.7 before
replication enters the picture.

**Rules out.** Tuning `epsilon` as the answer to judge noise. Treating
"underpowered" and "not significant" as the same outcome. Letting the paired test
promote anything. Reading `gate.replicates` from a config the round could have
rewritten — it is read after `verify_changed_paths` has restored out-of-scope
writes, so a round cannot buy itself statistical power.

## D13 — The paired test compares against the kept parent, not the frozen baseline

**Decision.** The paired sign test's comparator is the **current parent
scaffold's most recent eval draw**, persisted to `eval/runs/parent.json` on
every KEEP and loaded by the next round (`anvil.eval.cache.load_parent`).
Before the first KEEP — or right after `scripts/make_baseline.py` re-anchors,
which deletes `parent.json` — the frozen baseline stands in, because it is the
parent of the first candidate by definition. The frozen baseline itself is
never overwritten: it stays the round-1 anchor and the frontier's seed.

**Why.** D12 made promotion require a paired test, but the test paired each
candidate against the frozen original baseline forever (`save_baseline` ran
only from `scripts/make_baseline.py`). After A→B was kept, candidate C was
paired against A while the frontier judged C against B — so from round two on
the veto answered "does the candidate differ from the original scaffold", not
"does it improve on its parent". Those are different questions, and the loop
was being graded on the one nobody asked (issue #19).

**Why persist-on-KEEP and not contemporaneous re-eval.** The statistically
cleanest design re-evaluates parent and candidate in the same judge session
each round, controlling cross-session judge drift by construction. It also
doubles eval spend per round — at 50 dev rows × 50+ rounds, that is the
difference between an affordable campaign and an unaffordable one. The cheaper
design accepts that the parent's draw comes from an earlier judge session:
pairing cancels row difficulty, not session drift. That risk is already borne
by the frontier gate, which compares best-so-far scores measured across many
sessions, so the veto is no weaker than the gate it vetoes. The gate-validation
harness (#8) will measure the drift empirically rather than assume it away.

**Why a new file.** `baseline.json` is git-tracked reference data with a
comparability contract (mode, endpoints, fingerprints) that tests and tooling
depend on. `parent.json` is round state: rewritten wholesale on every KEEP,
written by nothing else. A superseded parent's draw is never consulted again,
and reuse across a revert streak is correct by construction — the parent is
genuinely still that scaffold. `parent.json` follows `frontier.json`'s
persistence pattern exactly: an uncommitted working-tree file that branch
operations carry over.

**Enforced by.** `round.py` loads `load_parent() or load_baseline()` as the
single comparator for the prompt, the score delta, and the gate call; the KEEP
path writes it via `save_parent(report_to_baseline(...))` before the git
verdict; a pre-flight `dataset_incomparability_reason` check on `parent.json`
raises before any spend if the comparator went stale out-of-band (mid-campaign
baseline regen, hand edit); `make_baseline.py` deletes `parent.json` whenever
it writes a new baseline.

**Rules out.** Pairing against the frozen baseline once a KEEP exists.
Persisting a vetoed or reverted candidate's draw as anyone's comparator.
Overwriting `baseline.json` from the round loop. Failing open when the parent
comparator is dataset-incompatible (the pre-flight raises instead).
