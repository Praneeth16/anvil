# ANVIL

**A self-mutating agent harness on Databricks.** An optimizer LLM rewrites an
agent's prompt scaffold, round after round, and an evaluation gate decides
which rewrites survive.

[![CI](https://github.com/Praneeth16/anvil/actions/workflows/ci.yml/badge.svg)](https://github.com/Praneeth16/anvil/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

---

## Why this exists

An agent's behaviour lives in its prompt scaffold: its skills, its rules, its
sampling settings, the descriptions of its tools. That scaffold is normally
tuned by a human editing text and forming an impression of whether it got
better. ANVIL replaces the impression with a measurement.

Each round, an optimizer LLM reads the agent's traces and failures, proposes one
change, and commits it to a git branch. The change is evaluated against a frozen
golden set and kept only if the numbers support it. Accepted rounds are
fast-forward merges, rejected rounds are `git branch -D`, and every change the
optimizer ever made is a reviewable diff.

What makes that work in practice:

- **The agent is text you can read.** Skills, rules, sampling and the tool
  registry are markdown and YAML under `scaffold/`. An optimizer cannot rewrite
  what it cannot read, which is also why the runtime is plain Python with no
  agent framework hiding behaviour in library internals.
- **The optimizer cannot edit its own grader.** Endpoints, thresholds, the gate
  and the judge's prompt live in a file outside its writable scope, and its
  writes are verified against that scope after the fact.
- **A bad score and a broken run are different outcomes.** A throttled endpoint
  produces `INFRA_FAIL`, which never discards a mutation. Exit codes carry the
  same distinction, so any script can gate CI.
- **Promotion survives judge noise.** A candidate has to beat the baseline
  row by row, on a paired sign test, not merely post a higher average.
- **A scorer that does not apply reports nothing.** Groundedness returns no
  score on a refusal row rather than `0.0`, so per-judge means move when the
  agent changes and not when the bucket mix does.
- **Any domain, no library changes.** A knowledge base, a golden set, a scaffold
  and an optional evaluator module are all paths passed at the boundary.

## What it has done

Ten rounds on the NeoVolt support domain, 7 kept and 3 no-op — measured
history, preserved with that domain at [`examples/neovolt/`](examples/neovolt/).
Those rounds also exposed the harness's binding constraints: the 20-row golden
set capped the paired gate's power at 0.185, and the optimizer was graded on
the same 12 rows every round, so "held-out" finalization was 60% seen rows.

The primary domain is now **MultiHopRAG**: a 120-row golden set over a
350-article news knowledge base, vendored by
[`scripts/build_multihop_domain.py`](scripts/build_multihop_domain.py) and
split 40 train / 50 dev / 30 test. Rounds evaluate against the whole dev
partition, which is what gives the gate real power; finalization alone touches
the test partition. See `data/ATTRIBUTION.md` for dataset licensing.

Two things to know before reading anything into the historical numbers. They
ran under scorer semantics v1–v3, incomparable to any current baseline. And two
runs of the *same* scaffold on the same rows differed by about 0.15 of
aggregate from judge noise alone, which is larger than most per-round gains —
the reason the gate requires a paired sign test. See [Limitations](#limitations).

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
              ├─ 6. GATE     improves ≥1 objective, regresses none, AND the
              │              improvement clears a paired sign test → KEEP
              │
              └─ 7. RECORD   round JSON · critique · mutations log
                             KEEP → ff-merge      REVERT → branch -D
```

Four outcomes, and each one licenses a different next step. KEEP and REVERT are
results about the agent. NOOP means the optimizer produced no usable action.
INFRA_FAIL means the round produced no measurement at all, so it says nothing
about the agent and is never allowed to discard a mutation. A throttled endpoint
used to look exactly like a quality regression; making these four distinct is
most of what [`docs/design/failure-vs-error.md`](docs/design/failure-vs-error.md)
argues.

## Quickstart

Requires Python 3.12 and a Databricks workspace with model serving.

```bash
uv venv --python 3.12
uv sync --extra dev --extra optimizer
```

<details>
<summary>Behind a corporate package index?</summary>

`pyproject.toml` pins PyPI at the project level so `uv.lock` stays portable. A
lock regenerated behind an internal proxy records that host in every wheel URL,
which breaks CI and external contributors. If your machine only resolves through
a proxy, set `UV_INDEX` for a single run rather than re-locking. The full
reasoning is in the comment in `pyproject.toml`. External users need nothing
extra.

</details>

The test suite needs no credentials and no network. A socket guard in
`tests/conftest.py` refuses off-machine connections for every test not marked
`live`, so the suite runs with the wifi off:

```bash
uv run pytest
```

Then, against a real workspace:

```bash
# What is configured right now: scaffold, endpoints, cached baseline.
# Reads files only, so no LLM call and no cost.
uv run python scripts/round_show.py

# Smallest real eval (9 dev rows, 3 judges)
uv run python scripts/evaluate.py --mode quick

# The mode the gate actually uses (50 dev rows, ~15-25 min)
uv run python scripts/evaluate.py --mode full

# Establish the bar, then optimize
uv run python scripts/make_baseline.py
uv run python scripts/run_round.py --rounds 1     # ~15-20 min

# Inspect what happened
uv run python scripts/round_show.py 1
uv run streamlit run scripts/round_dashboard.py
```

Credentials come from `DATABRICKS_HOST` plus `DATABRICKS_TOKEN`, or from a
`~/.databrickscfg` profile named with `--profile`. Model choice is
configuration, in `harness/config.yaml`.

Every script exits `0` when it measured the agent, `1` when it measured and the
agent fell short, `2` when it could not measure at all, and `130` on interrupt.
Reverting a bad mutation is the loop working, so a 50-round run that keeps only
two of them still exits `0`.

## Bring your own domain

A domain is four things, all supplied at the boundary. Nothing in `src/` names a
domain, and pointing ANVIL at your own problem requires no library changes:

```bash
uv run python scripts/evaluate.py \
  --scaffold        examples/pyloom-docs/scaffold \
  --kb-dir          examples/pyloom-docs/data/kb \
  --golden-set-path examples/pyloom-docs/data/golden_set.jsonl \
  --mode quick
```

[`examples/pyloom-docs/`](examples/pyloom-docs/) is a complete worked example: a
documentation-support agent for a fictional Python library, with 14
knowledge-base pages, a 20-row golden set, and a starting scaffold with real
headroom left in it. [`examples/neovolt/`](examples/neovolt/) preserves the
original support domain with its ten rounds of measured history.

Its golden set is built around traps, because a golden set without traps proves
nothing. The knowledge base documents a deprecated v1 client alongside the
current v2 one, so "how do I construct a client?" has a plausible wrong answer
sitting right next to the right one. Rows record the wrong values in
`must_not_include`, and `tests/test_example_domains.py` asserts that each of
those strings really does appear in some other document. A forbidden string that
appears nowhere is a trap with nothing to catch, and would let the row pass
unconditionally.

To build your own, copy that directory's shape:

| You provide | What it is |
|---|---|
| `data/kb/*.md` | knowledge base; YAML frontmatter with `doc_id`, then prose |
| `data/golden_set.jsonl` | one row per case, bucketed `direct` / `multi_hop` / `distractor` / `out_of_scope` |
| `scaffold/` | the starting agent: `harness.yaml`, `skills/`, `rules/`. This is what the optimizer rewrites |
| `harness/config.yaml` | endpoints, thresholds, gate, and the judge's domain description |
| `data/evaluator.py` | optional deterministic check functions, for programmatic scorers |

## What the optimizer may touch

The split between these two directories is a safety property, and it is the
reason the numbers mean anything:

| | `scaffold/` | `harness/config.yaml` |
|---|---|---|
| Optimizer may write | yes | no |
| Holds | skills, rules, sampling, tool registry, memory | endpoints, eval thresholds, gate, judge domain |

The optimizer is scored by the eval and rewarded for the score rising. Anything
it can write that affects the score is a shortcut it will eventually find, and
the cheapest shortcut is never "write a better agent". So the endpoints stay out
of its reach, because rewriting `runtime_endpoint` mid-run invalidates every
comparison. The loop's own lookback stays out, because that is the loop editing
its own recursion depth. The judge's prompt stays out, because that is editing
the grader directly.

Four independent layers enforce it: an OS-level sandbox, a tool allowlist, and
two policy interception points, followed by a diff check that holds even if the
SDK stops honouring the other four.

Read [`SECURITY.md`](SECURITY.md) before pointing this at anything you care
about. ANVIL runs a language model whose job is to write code this repository
then executes.

## Architecture

Five planes, physically separated, with a one-way import rule:

| Plane | Path | Knows about | Produces |
|---|---|---|---|
| Runtime | `src/anvil/runtime/` | composing a prompt and answering | trace + response |
| Eval | `src/anvil/eval/` | running `mlflow.genai.evaluate` | `EvalReport` + JSON |
| Optimizer | `src/anvil/optimizer/` | proposing a mutation | `OptimizerAction` + critique |
| Loop | `src/anvil/loop/` | git, branches, baselines, decisions | round artifacts + Delta row |
| Observability | `src/anvil/observability.py` | autolog + a standard tag set | tagged traces |

The runtime never imports the optimizer. The eval never imports git. The loop is
the only orchestrator. Each plane has a different reason to change and a
different blast radius; collapsing any two makes it impossible to say what a
round actually changed.

## Limitations

- **Judge noise is the binding constraint.** About 0.15 of aggregate between two
  identical runs at 8 rows, reproduced again while building
  `examples/pyloom-docs/`, where two runs of one unchanged scaffold scored 0.875
  and 0.917 and differed on a single refusal row. The gate requires a paired
  sign test over the per-row scores, which is what makes a real gain
  distinguishable from that noise. Power scales with rows: the 50-row dev
  partition detects a q=0.65–0.70 mutation with power ~0.5–0.7, the `quick`
  and `standard` modes are far weaker, and `gate.replicates` buys more at
  proportional cost.
- **The judge is unaligned.** `Judge.align` exists and has not been used. It
  would shrink the noise floor itself, which beats compensating for it, but it
  needs human labels on real traces and there is no honest way to synthesize
  those.
- **50 rounds is the target and 10 is what has been run.** The shape of the curve
  past that point is unknown.
- **`mode: code` is the less-travelled path.** The optimizer rewriting agent
  Python works, and a candidate the eval could not construct is now rejected
  during validation rather than failing inside the eval. It has had far less live
  exercise than `mode: prompt`.

## Documentation

| | |
|---|---|
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | setup, the checks, and what a change needs beyond a passing suite |
| [`docs/decisions.md`](docs/decisions.md) | the architectural decisions, and what each rules out |
| [`docs/plan.md`](docs/plan.md) | loop design, current state, prioritized open work |
| [`docs/design/failure-vs-error.md`](docs/design/failure-vs-error.md) | why a bad score and a broken run must not look alike |
| [`docs/design/scorer-applicability.md`](docs/design/scorer-applicability.md) | why an inapplicable scorer reports nothing, not zero |
| [`docs/verified-api-surface.md`](docs/verified-api-surface.md) | what MLflow and the Agent SDK actually expose, verified against pinned versions |
| [`docs/type-debt.md`](docs/type-debt.md) | every mypy exemption, with its consequence |
| [`SECURITY.md`](SECURITY.md) | trust boundaries and the optimizer's blast radius |
| [`scaffold/README.md`](scaffold/README.md) | the mutable directory, from the optimizer's point of view |

## License

[Apache License 2.0](LICENSE). The shipped evaluation domain derives from the
MultiHopRAG dataset (ODC-BY 1.0) — see [`data/ATTRIBUTION.md`](data/ATTRIBUTION.md).
