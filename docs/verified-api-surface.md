# Verified API surface

Recorded by introspecting the **installed** packages, not by reading docs. Public documentation for
the MLflow GenAI evaluation APIs was unreachable (403) during this pass, and doc-derived API names
have a habit of being aspirational, so everything below was read off the objects themselves.

Re-run the introspection after any dependency bump and update this file. Verified against
`uv.lock` on 2026-08-22.

| Package | Version |
|---|---|
| Python | 3.12.13 |
| mlflow | 3.11.1 |
| claude-agent-sdk | 0.1.64 |

## mlflow.genai (3.11.1)

`evaluate(data, scorers, predict_fn=None, model_id=None) -> EvaluationResult`

Note what is **absent**: no `seed`, no repetition count, no paired-comparison mode, and no
significance testing anywhere in the signature. Confirms the plan's premise — statistical rigour
for promotion decisions is ours to build, not MLflow's to provide.

Also present and relevant:

- `scorer` / `Scorer` — custom scorer decorator and base class.
- Built-in judge scorers: `Correctness`, `RetrievalGroundedness`, `RetrievalRelevance`,
  `RetrievalSufficiency`, `Safety`, `Guidelines`, `ExpectationsGuidelines`, `Equivalence`,
  `Completeness`, `Fluency`, `RelevanceToQuery`, `Summarization`, `ToolCallCorrectness`,
  `ToolCallEfficiency`, `UserFrustration`, `KnowledgeRetention`, plus conversational variants.
  The three scorers this harness configures (`correctness`, `retrieval_groundedness`,
  `refusal_appropriateness`) map onto the first two plus a guidelines-based judge.
- `make_judge(name, instructions, model=None, ...)` — custom judge construction.
- `Judge.align(traces, optimizer=None) -> Judge` and `judges.AlignmentOptimizer` — judge alignment
  from labelled traces. This is the MemAlign path, and it matters here: an unaligned LLM judge is
  the largest single noise source in the gate. Worth a later phase, out of scope for hardening.
- `datasets` (`create_dataset`, `get_dataset`, `search_datasets`) — UC-backed evaluation datasets,
  a candidate home for the golden set once the split is enforced.
- `optimize_prompt` / `optimize_prompts` — MLflow ships its own prompt optimizer. Worth knowing
  as prior art and as a baseline to benchmark ANVIL's loop against; not a replacement, since it
  does not mutate skills, rules, tool registries, or agent code.
- `register_prompt` / `load_prompt` / prompt aliases and versions — prompt registry. The harness
  deliberately uses scaffold-commit-SHA trace tags instead; keep that decision, it is stronger.
- `to_predict_fn`, `git_versioning`, `label_schemas`, `labeling`, `scheduled_scorers`.

## claude_agent_sdk 0.1.64 — `ClaudeAgentOptions`

The confinement plan assumed the permission callback was the only lever. It is not. Fields that
change Phase 1:

- **`sandbox: SandboxSettings`** — an actual OS-level sandbox, not a policy hook.
  Keys: `enabled`, `autoAllowBashIfSandboxed`, `excludedCommands`, `allowUnsandboxedCommands`,
  `network`, `ignoreViolations`, `enableWeakerNestedSandbox`.
  `SandboxNetworkConfig`: `allowUnixSockets`, `allowAllUnixSockets`, `allowLocalBinding`,
  `httpProxyPort`, `socksProxyPort` — so network egress is controllable, which is how a round
  becomes reproducible and non-exfiltrating.
  `SandboxIgnoreViolations`: `file`, `network` — deliberate, auditable exceptions.
- **`allowed_tools: list[str]`** — an allowlist, the correct default posture. The current code
  only sets `disallowed_tools` (deny two tools, permit everything else).
- **`hooks: {PreToolUse|PostToolUse|...: [HookMatcher]}`** — `PreToolUse` is a second, independent
  interception point. Belt and braces alongside `can_use_tool`.
- **`max_budget_usd: float`** and **`task_budget: TaskBudget`** — hard cost ceilings enforced by
  the SDK. `harness/config.yaml` declares `cost_budget_usd_per_round: 5.0` and currently enforces
  it nowhere; this wires it up for real.
- **`permission_mode`** — includes `'dontAsk'` and `'auto'` beyond the modes the code comments
  discuss.
- **`add_dirs: list[str | Path]`** — explicit extra readable directories, so `cwd` can be a bare
  worktree while the optimizer still reads what it legitimately needs.
- `can_use_tool` returns typed `PermissionResultAllow(updated_input, updated_permissions)` or
  `PermissionResultDeny(message, interrupt)`. The existing code returns a raw dict
  (`{"behavior": "allow", ...}`), which happens to work but bypasses the typed contract.
- `env: dict[str,str]` — scoped environment for the subprocess, rather than mutating
  `os.environ` process-wide as `setup_anthropic_env` does today.

**Consequence for the plan:** Phase 1 gets stronger than designed. Layer four independent
controls — `sandbox` (OS), `allowed_tools` (allowlist), `PreToolUse` hook + `can_use_tool` (path
policy), and post-hoc diff verification (the one that holds if the SDK changes) — instead of
relying on a single best-effort callback. `max_budget_usd` also closes the unenforced cost budget.
