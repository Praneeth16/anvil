# Plan: loop design, current state, open work

What the loop does, where the project actually is, and what is left. For the
reasoning behind the architecture, read `docs/decisions.md`. For the security
model, `SECURITY.md`.

Last updated: 2026-08-24.

---

## The loop

One round takes the agent as it stands, asks an optimizer to change it, and
keeps the change only if the evidence supports it.

```
                    ┌──────────────────────────────────────┐
                    │  parent branch (anvil/exp)           │
                    │  the best agent found so far         │
                    └───────────────┬──────────────────────┘
                                    │  branch anvil/exp-round-N
                                    ▼
   ┌────────────────────────────────────────────────────────────────┐
   │ 1. PLAN      optimizer session reads: the scaffold, the last    │
   │              `critique_lookback` critiques, the last            │
   │              `revert_lookback` reverted mutations, and the      │
   │              failing cases from the previous eval              │
   │              → emits one OptimizerAction + a critique          │
   ├────────────────────────────────────────────────────────────────┤
   │ 2. APPLY     the action is applied under scaffold/ ONLY, and    │
   │              the writes are verified against that scope after   │
   │              the fact. Out-of-scope write → INFRA_FAIL, and it  │
   │              is reported separately from a flaky endpoint      │
   ├────────────────────────────────────────────────────────────────┤
   │ 3. COMMIT    the diff is committed to the round branch, so the  │
   │              mutation is reviewable as a diff, not a log line   │
   ├────────────────────────────────────────────────────────────────┤
   │ 4. EVAL      mlflow.genai.evaluate over the mode's row subset:  │
   │              correctness, retrieval_groundedness,               │
   │              refusal_appropriateness. Scorers that do not apply │
   │              to a bucket report nothing, not zero               │
   ├────────────────────────────────────────────────────────────────┤
   │ 5. JUDGE-    is this run a measurement at all? error rate ≤     │
   │    ABILITY   max_error_rate AND assessed rows ≥                 │
   │              min_scorable_rows AND the baseline's semantics    │
   │              version and scorer fingerprint still match        │
   │              → no: INFRA_FAIL. The round is never reverted on   │
   │                unmeasured evidence                             │
   ├────────────────────────────────────────────────────────────────┤
   │ 6. GATE      frontier: improves ≥1 objective, regresses none    │
   │              by > epsilon → KEEP, else REVERT                  │
   ├────────────────────────────────────────────────────────────────┤
   │ 7. RECORD    eval/runs/round_NNN.json,                          │
   │              scaffold/memory/round_NNN_critique.md,             │
   │              eval/mutations.jsonl, anvil.default.mutations      │
   └───────────────────────────┬────────────────────────────────────┘
                               │
              KEEP ────────────┴──────────── REVERT / INFRA_FAIL
      ff-merge into parent                 git branch -D
```

Four outcomes, and the distinctions between them are the point: **KEEP** (the
change helped), **REVERT** (it did not), **NOOP** (the optimizer produced no
usable action), **INFRA_FAIL** (the round produced no measurement, so it says
nothing about the agent either way).

Run it:

```bash
python scripts/make_baseline.py          # the bar, once
python scripts/run_round.py --rounds 50  # the loop
python scripts/round_show.py 12          # inspect a round
```

---

## Current state

**The gate is noise-aware.** Promotion requires the improvement to survive a
paired sign test over the per-row scores, not merely to clear `epsilon`
(`docs/decisions.md` D12). It is inert until the baseline is regenerated — see
open item 2, which is one command.

**Hardening complete.** The eval layer is correct as far as it has been
pushed: failure is distinguished from error end to end, applicability is a
first-class outcome, the gate defers to one definition of comparability
instead of keeping a second copy, and scorer semantics are versioned into the
baseline fingerprint (groundedness at v4).

**Domain-portable.** A knowledge base, golden set, evaluator module and
scaffold are all supplied at the boundary; nothing in `src/` names a domain.
`examples/pyloom-docs/` is the second domain that proves it, and
`tests/test_example_domains.py` keeps it honest.

**Offline CI.** The whole suite runs with no credentials and no network, and
that is enforced rather than asserted: `tests/conftest.py` refuses
off-machine sockets for any test not marked `live`.

**Measured, once.** Ten rounds on the NeoVolt domain, 7 kept / 3 no-op:

| Round | Aggregate | Decision |
|---:|---:|---|
| 1 | 0.750 | keep |
| 2 | 0.778 | keep |
| 3 | 0.833 | keep |
| 4 | 0.819 | keep |
| 5 | — | noop |
| 6 | 0.875 | keep |
| 7 | — | noop |
| 8 | 0.875 | keep |
| 9 | — | noop |
| 10 | 0.819 | keep |

Read this as evidence the machinery runs end to end and moved the agent, and
**not** as a benchmark. Those rounds were measured under scorer semantics
v1–v3, on a 20-row golden set whose 12-row `standard` mode capped the paired
gate's power at 0.185 and left no room for a holdout. The NeoVolt domain now
lives at `examples/neovolt/` with its full history; the primary domain is
MultiHopRAG — 120 rows split 40 train / 50 dev / 30 test, rounds evaluated on
the whole dev partition, finalization alone on test (issues #15/#21). The
curve also stops well short of the 50-round target, where the interesting
question — whether gains keep coming or the optimizer runs out of ideas —
actually lives.

---

## Open work, in priority order

### 1. Judge alignment

The unaligned LLM judge is the root cause of the noise the paired gate now works
around, not merely correlated with it. MLflow exposes
`Judge.align(traces, optimizer=None)` and `judges.AlignmentOptimizer`
(`docs/verified-api-surface.md`). Aligning the judge against human labels would
shrink the noise floor itself, which is strictly better than compensating for it
— a smaller floor means the paired test detects smaller real effects at
`replicates: 1`.

**Blocked on labels, not on code.** Alignment needs human verdicts on real
traces, and there is no honest way to synthesize those: a judge aligned against
labels produced by a judge measures nothing. The prerequisite is a person
labelling a few dozen rows from a live run.

### 2. Chunk-level retrieval (#26)

`search_knowledge_base` ranks whole documents but returns a prefix snippet.
That fit NeoVolt's short policy pages; MultiHopRAG articles average ~10.3k
chars with the answer mid-body, and the migration's live probe measured rows
failing correctness while groundedness passed — the fact sat past the prefix.
The migration ships a stopgap (500 → 3000 chars). The proper fix is chunk-level
retrieval: BM25 over deterministic ~1-2k chunks, results labeled with their
parent `doc_id` so the citation contract is unchanged. Land it before a
campaign — snippet shape changes what a baseline measures.

### 3. Regenerate the baseline to activate the paired gate — DONE

Done 2026-08-26 (baseline with `per_row`, scorer semantics v1/v5), and done
again on the new primary domain: the MultiHopRAG migration regenerates the
baseline on the 50-row dev partition as part of the same change. The paired
gate is operative, not documented.

### 4. Parent-anchored paired test (#19) — DONE

The paired test now compares each candidate against its actual parent:
`eval/runs/parent.json` is rewritten from the kept candidate's eval report on
every KEEP, with the frozen baseline standing in only until the first KEEP.
The contemporaneous alternative (re-evaluate the parent every round) was
rejected on cost — the trade and its drift caveat are recorded as D13 in
`docs/decisions.md`. #8's A/A harness remains the empirical check on
cross-session judge drift.

---

## Closed

**Type debt (was items 3, 4, 5).** Measured rather than assumed: deleting the
`disable_error_code` block showed 24 errors, and two were behaviour, not
annotations. `ChatClient` is a protocol at the provider boundary — describing
what ANVIL calls, because a protocol `openai.OpenAI` could also satisfy turns out
to be impossible and the reason is recorded in `docs/type-debt.md`.
`MemorySystem.__init__` is declared and checked during candidate validation, so a
candidate the eval could not construct is rejected instead of failing inside the
eval. The `source` constants are `Final[SourceTag]`. Four modules left the
suppression block; the two that remain name only the codes they need, and all
twelve remaining errors are MLflow's own typing.

Two bugs fell out of it: code mode passed the *unresolved* client, so every
code-mode round built its candidate with `llm_client=None` and scored a
passthrough rather than an agent; and the KB frontmatter narrowing was covered by
no test at all, so reverting it passed all 586.

**Optimizer confinement (was item 6).** Mostly already landed — the OS `sandbox`,
the `allowed_tools` allowlist, `max_budget_usd` wiring the declared cost budget,
and the typed permission result were all in place and the entry had gone stale.
What was missing was the `PreToolUse` hook, now added as a second independent
enforcement point for the same `ToolPolicy.decide`. Two enforcement points, one
rule; the tests assert the two verdicts never disagree.

**Licensing.** Apache-2.0, `LICENSE` verbatim. The bracketed
`Copyright [yyyy] [name of copyright owner]` line is inside Apache's own APPENDIX
— instructions for applying the license, not a field to fill in — so it stays as
shipped and no owner is asserted. No `NOTICE` file: Apache-2.0 requires one only
if the work already carries attribution notices, and nothing here does.

**`CONTRIBUTING.md`.** Written.

**The `research/minimax-m27-*.md` citations.** The files exist nowhere, so
`docs/decisions.md` D4 and `CLAUDE.md` now say the reference is external and not
vendored, and point at the decision instead of a dead path. Writing the files
would have meant inventing an "authoritative reference", which is worse than a
missing one.

---

## Not planned

- **Lakebase**, for anything in current scope. Revisit only for high-QPS
  per-conversation runtime memory.
- **An agent framework** in the runtime. See `docs/decisions.md` D3 — this is
  the one decision the project cannot trade away.
- **A second provider SDK** in runtime dependencies. D1.
