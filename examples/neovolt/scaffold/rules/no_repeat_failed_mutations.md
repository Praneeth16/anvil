---
rule_id: no_repeat_failed_mutations
kind: loop_detection
applies_to: optimizer
priority: high
created_at: 2026-04-20
---

# Loop-detection: do not re-propose a mutation that was already reverted

Before proposing a mutation, the optimizer must query the `mutations` Delta
table for recent rows where `decision='revert'` that touched any of the
files the new proposal would touch. If a near-identical mutation was already
tried and reverted, either skip or propose a materially different change.

**Operational check (executed by the Mutation Planner):**

```sql
SELECT mutation_id, git_commit_sha, files_changed, diff_summary, proposed_at
FROM anvil.default.mutations
WHERE decision = 'revert'
  AND arrays_overlap(files_changed, :proposed_files)
ORDER BY proposed_at DESC
LIMIT :revert_lookback
```

The resulting rows are included in the optimizer prompt with the instruction:
*"These mutations touched the same files and were rolled back. Do not propose
a mutation that is semantically equivalent to any of these. If your proposed
change is similar, either (a) explain what is materially different this time,
or (b) propose a different change."*
