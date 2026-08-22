# Security model

ANVIL runs a large language model whose job is to **write files and code that this repository
then executes**. Read this before pointing it at anything you care about.

## What ANVIL does by design

Each round, an optimizer LLM is given a working tree and asked to mutate the agent's scaffold —
markdown skills and rules, YAML sampling config, and (in `mode: code`) Python modules that the
evaluator imports and runs. The output of a language model is therefore, by construction, code
that executes on your machine or in your Databricks workspace.

There is no way to make that safe in the abstract. What ANVIL can do is bound it.

## Trust boundaries

| Component | Trust | Enforcement |
|---|---|---|
| Harness source (`src/anvil/**`) | Trusted. Human-authored, reviewed, versioned. | Off-limits to the optimizer. |
| Gate, evaluator, golden set | Trusted **and adversarially relevant** — the optimizer is graded by these, so it has an incentive to edit them. | Hard denylist + post-hoc diff verification. |
| Scaffold (`scaffold/**`) | Untrusted. LLM-authored. | The optimizer's only writable scope. |
| Optimizer-authored agent code | Untrusted, and *executed*. | AST denylist + isolated import (`optimizer/code_validation.py`). |
| Golden set answers | Secret from the optimizer. | Denylist on `data/golden_set.jsonl`; import denylist blocks `golden_set`, `answer_key`, `ground_truth`. |

The load-bearing rule: **a self-modifying system must not be able to modify its own grader.**
If the optimizer can weaken a scorer, raise its own score, or append easy cases to the golden
set, every number this harness produces is meaningless — and it will find that path, because
it is optimizing for the score.

## Known limitations

- **Reward hacking is the primary threat, not exfiltration.** The optimizer is not assumed
  malicious; it is assumed to be an optimizer. Treat gate integrity as a security property.
- **The permission callback is best-effort.** It depends on the Claude Agent SDK version honoring
  it. Diff verification before commit is the check that holds regardless, and it is the one to
  keep working.
- **Evidence redaction is advisory.** A regex pass over transcripts before they hit disk reduces
  accidental credential capture. It is not a guarantee. Do not run ANVIL with credentials you
  would mind seeing in a round transcript.
- **The harness authenticates to Databricks.** Anything the optimizer can execute inherits that
  identity. Use a service principal scoped to exactly the catalog, schema, and endpoints the
  harness needs — never a workspace admin.
- **Isolation is process-level, not adversarial.** A git worktree and a tool policy stop mistakes
  and reward hacking. They are not a sandbox against code that is actively trying to escape one.
  For that, run rounds in a disposable serverless job, not on a laptop with your SSH keys.

## Reporting

Open a private security advisory on the repository rather than a public issue. Include the round
evidence directory if the finding involves the optimizer escaping its writable scope — that is
the class of bug most worth fixing fast.
