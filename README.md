# ANVIL

A self-mutating agent harness on Databricks. The runtime is an
`mlflow.pyfunc.ResponsesAgent`; the optimizer is a Claude Agent SDK
session that mutates the agent's **scaffold** (skills, rules, memory,
sampling, tool registry) round by round, with `git` + MLflow evals
as gate.

## Architecture (5 planes, physically separated)

| Plane | Path | Knows about | Output |
|---|---|---|---|
| Runtime | `src/anvil/runtime/` | how to compose a prompt and answer | trace + response |
| Eval | `src/anvil/eval/` | how to run `mlflow.genai.evaluate` | `EvalReport` + JSON |
| Optimizer | `src/anvil/optimizer/` | how to propose a mutation | `OptimizerAction` + critique |
| Loop | `src/anvil/loop/` | git, branches, baselines, decisions | round artifacts + Delta row |
| Observability | `src/anvil/observability.py` | autolog + standard tag set | tagged traces |

The runtime never imports from the optimizer. The eval never imports
from git. The loop is the only orchestrator.

## Quickstart

```bash
uv venv --python 3.12
uv sync --extra dev --extra optimizer

# Smoke runtime (~30s)
uv run python scripts/run_round.py --smoke

# Quick eval (8 rows, 3 scorers, ~3-5 min)
uv run python scripts/evaluate.py --mode quick

# One round end-to-end (~15-20 min)
uv run python scripts/run_round.py --rounds 1
```

## Storage

- **Scaffold** → `scaffold/` (markdown + YAML, git-tracked).
- **Mutations log** → `anvil.default.mutations` (Delta append-only).
- **Traces** → MLflow native Delta sync, experiments
  `anvil-exp-runtime`, `anvil-exp-eval`, `anvil-exp-optimizer`.
- **Per-round eval JSON** → `eval/runs/round_NNN.json`.
- **Per-round critique** → `scaffold/memory/round_NNN_critique.md`.
