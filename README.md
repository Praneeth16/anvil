# ANVIL

**A self-mutating agent harness on Databricks.** An optimizer LLM rewrites a
support agent's prompt scaffold, round after round, and an evaluation gate
decides which rewrites survive.

[![CI](https://github.com/Praneeth16/anvil/actions/workflows/ci.yml/badge.svg)](https://github.com/Praneeth16/anvil/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

---

## Why this exists

An agent's behaviour lives in its prompt scaffold — its skills, its rules, its
sampling settings, the descriptions of its tools. That scaffold is normally
tuned by a human editing text and forming an impression of whether it got
better.

ANVIL replaces the impression with a measurement. Each round, an optimizer LLM
reads the agent's traces and failures, proposes one change, and commits it to a
git branch. The change is then evaluated against a frozen golden set, and kept
only if the numbers support it. Rejected rounds are `git branch -D`; accepted
rounds are fast-forward merges. The mutation history is a git log, and every
change the optimizer ever made is a reviewable diff.

The design constraint that follows from this, and that shapes everything else:
**the thing being optimized must stay legible to the thing doing the
optimizing.** That is why the runtime is plain Python with no agent framework,
and why the scaffold is markdown and YAML rather than rows in a table. An
optimizer cannot rewrite what it cannot read.

## Does it work?

Ten rounds on the built-in support domain — 7 kept, 3 no-op:

| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Aggregate | 0.750 | 0.778 | 0.833 | 0.819 | — | **0.875** | — | **0.875** | — | 0.819 |
| Decision | keep | keep | keep | keep | noop | keep | noop | keep | noop | keep |

> **Read this as machinery, not as a benchmark.** Those rounds ran under scorer
> semantics v1–v3. v4 changed which buckets a scorer applies to, so these
> numbers are **not comparable** to the current baseline (0.828 on `standard`,
> 12 rows), and a rerun would not reproduce them. They are the record of a real
> run that moved the agent end to end. The run also stops well short of the
> 50-round target, which is where the interesting question lives — whether
> gains keep coming, or the optimizer runs out of ideas.

And the honest limitation, up front: two runs of the *same* scaffold on the
same rows differed by **~0.15 of aggregate from judge noise alone**. The gate's
`epsilon` is `0.0`, so noise of that size is sometimes read as signal. See
[Limitations](#limitations).

## How a round works

```
   parent branch (anvil/exp) ── the best agent found so far
              │
              ├─ 1. PLAN     optimizer reads the scaffold, recent critiques,
              │              past reverted mutations, and the last eval's
              │              failures → proposes one action
              │
              ├─ 2. APPLY    writes under scaffold/ only; the writes are
              │              verified against that scope afterwards
              │
              ├─ 3. COMMIT   to branch anvil/exp-round-N
              │
              ├─ 4. EVAL     mlflow.genai.evaluate over the mode's rows:
              │              correctness · groundedness · refusal
              │
              ├─ 5. JUDGEABLE?  enough rows actually scored? baseline still
              │                 comparable?  no → INFRA_FAIL, never a revert
              │
              ├─ 6. GATE     improves ≥1 objective, regresses none → KEEP
              │
              └─ 7. RECORD   round JSON · critique · mutations log
                             KEEP → ff-merge      REVERT → branch -D
```

Four outcomes, and the distinctions carry weight. **KEEP** and **REVERT** are
results about the agent. **NOOP** means the optimizer produced no usable
action. **INFRA_FAIL** means the round produced no measurement at all — so it
says nothing about the agent, and is never allowed to discard a mutation. A
throttled endpoint used to look exactly like a quality regression; making these
four distinct is most of what `docs/design/failure-vs-error.md` argues.

## Architecture

Five planes, physically separated, with a one-way import rule:

| Plane | Path | Knows about | Produces |
|---|---|---|---|
| Runtime | `src/anvil/runtime/` | composing a prompt and answering | trace + response |
| Eval | `src/anvil/eval/` | running `mlflow.genai.evaluate` | `EvalReport` + JSON |
| Optimizer | `src/anvil/optimizer/` | proposing a mutation | `OptimizerAction` + critique |
| Loop | `src/anvil/loop/` | git, branches, baselines, decisions | round artifacts + Delta row |
| Observability | `src/anvil/observability.py` | autolog + a standard tag set | tagged traces |

The runtime never imports the optimizer. The eval never imports git. The loop
is the only orchestrator. Each plane has a different reason to change and a
different blast radius; collapsing any two makes it impossible to say what a
round actually changed.

## Quickstart

Requires Python 3.12 and a Databricks workspace with model serving.

```bash
uv venv --python 3.12
uv sync --extra dev --extra optimizer
```

<details>
<summary>Behind a corporate package index?</summary>

`pyproject.toml` pins PyPI at the project level so `uv.lock` stays portable —
a lock regenerated behind an internal proxy records that host in every wheel
URL, which breaks CI and external contributors. If your machine only resolves
through a proxy, set `UV_INDEX` for a single run rather than re-locking. The
full reasoning is in the comment in `pyproject.toml`. External users need
nothing extra.

</details>

The test suite needs no credentials and no network — that is enforced, not
merely intended, so this works with the wifi off:

```bash
uv run pytest
```

Then, against a real workspace:

```bash
# what is configured right now: scaffold, endpoints, cached baseline.
# Reads files only -- no LLM call, no cost.
uv run python scripts/round_show.py

# smallest real eval (8 rows, 3 judges)
uv run python scripts/evaluate.py --mode quick

# the mode the gate actually uses (12 rows, ~3-5 min)
uv run python scripts/evaluate.py --mode standard

# establish the bar, then optimize
uv run python scripts/make_baseline.py
uv run python scripts/run_round.py --rounds 1     # ~15-20 min

# inspect what happened
uv run python scripts/round_show.py 1
uv run streamlit run scripts/round_dashboard.py
```

Credentials come from `DATABRICKS_HOST` + `DATABRICKS_TOKEN`, or from a
`~/.databrickscfg` profile named with `--profile`. Model choice is
configuration, in `harness/config.yaml`.

Every script exits `0` when it measured the agent, `1` when it measured and the
agent fell short, `2` when it could not measure at all, and `130` on
interrupt — so any of them can gate CI. A revert is not a failure: a 50-round
run that keeps two mutations exits `0`.

## Bring your own domain

A domain is four things, all supplied at the boundary. Nothing in `src/` names
a domain, and pointing ANVIL at your own problem requires no library changes:

```bash
uv run python scripts/evaluate.py \
  --scaffold      examples/pyloom-docs/scaffold \
  --kb-dir        examples/pyloom-docs/data/kb \
  --golden-set-path examples/pyloom-docs/data/golden_set.jsonl \
  --mode quick
```

**[`examples/pyloom-docs/`](examples/pyloom-docs/)** is a complete worked
example: a documentation-support agent for a fictional Python library, with 14
knowledge-base pages, a 20-row golden set, and a starting scaffold with real
headroom left in it.

Its golden set is built around traps, because a golden set without traps proves
nothing. The knowledge base documents a deprecated v1 client *and* the current
v2 one, so "how do I construct a client?" has a plausible wrong answer sitting
right next to the right one. Rows record the wrong values in
`must_not_include`, and `tests/test_example_domains.py` asserts that each of
those strings really does appear in some other document — a forbidden string
that appears nowhere is a trap with nothing to catch, and would let the row
pass unconditionally.

To build your own, copy that directory's shape:

| You provide | What it is |
|---|---|
| `data/kb/*.md` | knowledge base; YAML frontmatter with `doc_id`, then prose |
| `data/golden_set.jsonl` | one row per case, bucketed `direct` / `multi_hop` / `distractor` / `out_of_scope` |
| `scaffold/` | the starting agent: `harness.yaml`, `skills/`, `rules/` — this is what the optimizer rewrites |
| `harness/config.yaml` | endpoints, thresholds, gate, and the judge's domain description |
| `data/evaluator.py` | optional deterministic check functions, for programmatic scorers |

## Configuration: what the optimizer may touch

The split is the load-bearing safety property, not a filing convention:

| | `scaffold/` | `harness/config.yaml` |
|---|---|---|
| **Optimizer may write** | yes | **no** |
| Holds | skills, rules, sampling, tool registry, memory | endpoints, eval thresholds, gate, judge domain |

The optimizer is *scored* by the eval and rewarded for the score rising.
Anything it can write that affects the score is a shortcut it will eventually
find, and the cheapest shortcut is never "write a better agent". So the
endpoints stay out of its reach (rewriting `runtime_endpoint` mid-run
invalidates every comparison), the loop's own lookback stays out (that is the
loop editing its own recursion depth), and the judge's prompt stays out — that
one is editing the grader directly.

This is enforced, not requested: writes are verified against the scope after
the fact, and an escape is reported as a containment failure distinct from a
flaky endpoint. See [`SECURITY.md`](SECURITY.md), and note what it says
plainly — ANVIL runs a language model whose job is to write code this
repository then executes. Read it before pointing this at anything you care
about.

## Limitations

- **Judge noise is the binding constraint.** ~0.15 of aggregate between two
  identical runs at 8 rows, and reproduced again while writing
  `examples/pyloom-docs/`: two runs of one unchanged scaffold scored 0.875 and
  0.917, differing on a single refusal row. Mitigated only by preferring
  `standard` (12 rows) over `quick`; the real fix is a paired, noise-aware gate,
  which is the top item in `docs/plan.md`. MLflow's `evaluate()` offers no seed,
  no repetition count and no paired mode, so that work is ANVIL's.
- **The judge is unaligned.** `Judge.align` exists and has not been used yet.
- **50 rounds is the target; 10 is what has been run.** The shape of the curve
  past that point is unknown.
- **`mode: code`** — where the optimizer rewrites agent Python rather than
  prompts — works, but the `MemorySystem` constructor is not yet a validated
  contract, so a bad candidate fails inside the eval instead of being rejected
  before it costs anything.

## Documentation

| | |
|---|---|
| [`docs/decisions.md`](docs/decisions.md) | the architectural decisions, and what each rules out |
| [`docs/plan.md`](docs/plan.md) | loop design, current state, prioritized open work |
| [`docs/design/failure-vs-error.md`](docs/design/failure-vs-error.md) | why a bad score and a broken run must not look alike |
| [`docs/design/scorer-applicability.md`](docs/design/scorer-applicability.md) | why an inapplicable scorer reports nothing, not zero |
| [`docs/verified-api-surface.md`](docs/verified-api-surface.md) | what MLflow and the Agent SDK actually expose, verified against pinned versions |
| [`docs/type-debt.md`](docs/type-debt.md) | every mypy exemption, with its consequence |
| [`SECURITY.md`](SECURITY.md) | trust boundaries and the optimizer's blast radius |
| [`scaffold/README.md`](scaffold/README.md) | the mutable directory, from the optimizer's point of view |

## License

[Apache License 2.0](LICENSE).
