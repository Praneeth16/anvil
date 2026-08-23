# Plan: loop design, current state, open work

What the loop does, where the project actually is, and what is left. For the
reasoning behind the architecture, read `docs/decisions.md`. For the security
model, `SECURITY.md`.

Last updated: 2026-08-23.

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
v1–v3. v4 changed which buckets a scorer applies to, so these numbers are not
comparable to the current baseline (0.828 on `standard`, 12 rows) and a rerun
would not reproduce them. The curve also stops well short of the 50-round
target, where the interesting question — whether gains keep coming or the
optimizer runs out of ideas — actually lives.

---

## Open work, in priority order

### 1. A noise-aware promotion gate

The largest correctness problem left. Two runs of the *same* scaffold, same
rows, same model differed by ~0.15 of aggregate from judge noise alone
(0.875 vs 0.722; `docs/design/failure-vs-error.md`). `gate.epsilon` is
`0.0`, so a strict positive delta promotes — meaning noise of that size is
read as signal roughly half the time.

Mitigated today only by using `standard` (12 rows) over `quick` (8). The
actual fix is a paired, noise-aware comparison: repeat measurement, or a
significance test, or both. MLflow's `evaluate()` offers no `seed`, no
repetition count, and no paired mode (`docs/verified-api-surface.md`), so
this is ANVIL's to build.

Blocks leaning on per-judge numbers for anything finer-grained than the
aggregate.

### 2. Judge alignment

The unaligned LLM judge is the root cause of item 1, not merely correlated
with it. MLflow exposes `Judge.align(traces, optimizer=None)` and
`judges.AlignmentOptimizer` (`docs/verified-api-surface.md`). Aligning the
judge against human labels on a handful of rows would shrink the noise floor
that item 1 works around.

### 3. A `ChatClient` protocol at the provider boundary

`GatewayClient` is a duck-typed stand-in for `OpenAI`, exposed as
`OpenAI | GatewayClient` or `OpenAI | None` and silenced in mypy
(`docs/type-debt.md`). Define the protocol both satisfy, and the type checker
starts seeing that boundary again. Unblocks deleting two entries from the
`pyproject.toml` exemption ratchet.

### 4. `MemorySystem.__init__` as a real contract

`eval/runner.py` instantiates code-mode agents as
`cls(llm_client=..., model=...)`, but the `MemorySystem` ABC declares no
`__init__`. A code-mode optimizer can therefore write a subclass with an
incompatible signature, and the failure surfaces deep inside the eval as an
infrastructure error rather than as a rejected candidate. Declare the
constructor on the ABC and validate the candidate's signature in
`optimizer/code_validation.py`, before the eval spends money.

### 5. Small, cheap type debt

Each is itemised with its consequence in `docs/type-debt.md`:

- `source` is `Literal["production","eval","optimizer"]` fed a bare `str`; a
  typo in the default silently produces traces no observability query matches.
- `eval/cache.py` sorts scorer fingerprints with a key that can return
  `None`, which raises `TypeError` on comparison. Touches the gate.
- `runtime/client.py` and `tools/search_knowledge_base.py` promise `str`
  where a missing KB frontmatter field yields `None`.

### 6. Stronger optimizer confinement

`ClaudeAgentOptions` turns out to expose more than the current design uses:
`sandbox: SandboxSettings` (OS-level, not policy-only), `allowed_tools` (an
allowlist rather than a denylist), `hooks` as an independent interception
point, and `max_budget_usd` / `task_budget` as hard ceilings
(`docs/verified-api-surface.md`). Confinement can therefore be four
independent layers instead of one callback plus post-hoc diff verification.

### 7. Reusability queue

- `CONTRIBUTING.md` — not yet written.
- Copyright holder for the `LICENSE` appendix is undecided; the shipped
  Apache-2.0 text is verbatim and asserts no owner.
- `research/minimax-m27-*.md` is cited by `docs/decisions.md` D4 and
  `CLAUDE.md` but is not in the repository.

---

## Not planned

- **Lakebase**, for anything in current scope. Revisit only for high-QPS
  per-conversation runtime memory.
- **An agent framework** in the runtime. See `docs/decisions.md` D3 — this is
  the one decision the project cannot trade away.
- **A second provider SDK** in runtime dependencies. D1.
