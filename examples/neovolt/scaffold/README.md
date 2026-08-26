# ANVIL scaffold

The mutable agent harness. **This directory is the scaffold.** Every file
here is something the ANVIL optimizer may rewrite autonomously.

The immutable runtime configuration (model endpoints + optimizer loop
meta-config) lives in `harness/config.yaml`, OUTSIDE this directory.
The optimizer's filesystem write scope must not include that file.
See `docs/decisions.md` Decision 8 for the rationale.

On deploy, a DAB task syncs this directory into the UC Volume
`anvil.default.scaffold`. The runtime `ResponsesAgent` loads from the Volume
at startup.

## Layout

```
<repo>/
├── scaffold/                     # Optimizer-mutable
│   ├── harness.yaml              # sampling, active skills[], rules[], tools[]
│   ├── skills/
│   │   └── <skill_name>.md       # One prompt template per skill
│   ├── memory/
│   │   ├── round_000_seed.md     # Seed memory (hand-written starter lessons)
│   │   ├── round_NNN_critique.md # Self-criticism emitted by the optimizer each round
│   │   └── round_NNN_lesson.md   # Distilled lessons carried forward
│   └── rules/
│       └── <rule_name>.md        # One workflow rule per file (guardrails, thresholds, templates)
└── harness/
    └── config.yaml               # IMMUTABLE: runtime_endpoint, optimizer_endpoint, loop.*
```

## How mutations work

1. Optimizer creates a git branch `anvil/round-N`
2. Edits files in this directory (or adds new ones, or deletes)
3. Commits
4. Eval gate re-deploys the branch's scaffold to a test Volume path and
   runs `mlflow.genai.evaluate()` against the frozen golden eval set
5. Pass → fast-forward merge to `main`, re-sync primary Volume, append
   row to `anvil.default.mutations`
6. Fail → delete branch, append row to `anvil.default.mutations` with
   `decision='revert'`

## Reference

See `docs/plan.md` for the full loop design and `docs/decisions.md`
for the architectural rationale.
