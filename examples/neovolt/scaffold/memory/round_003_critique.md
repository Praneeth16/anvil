---
round: 3
branch: anvil/exp-round-3
decision: keep
action_kind: edit_rule
parse_status: ok_last_of_many
baseline_score: 0.7440
mutated_score: 0.8333
score_delta: +0.0893
---

# Round 3 critique

## Action applied
edit_rule rules/answer_specificity.md: Round 2 kept at 0.778 (correctness 0.5, groundedness 0.833, refusal 1.0) but introduced a new distractor-groundedness re

## Rationale (from optimizer)
Round 2 kept at 0.778 (correctness 0.5, groundedness 0.833, refusal 1.0) but introduced a new distractor-groundedness regression: golden_013 dropped 1.0 → 0.5 because Section 3's directive to explain 'where [the user's value] came from' invited speculation beyond retrieved chunks (enrollment cohorts, closure motivation) even though the tokens themselves (Plan A / legacy / closed) are in the retrieved chunks. Sections 1, 2, and 4 are untouched — they delivered the round-2 wins and should be preserved. The surgical edit adds a 'Groundedness guardrails for distractor answers' block inside Section 3 that: (a) conditions legacy-token naming on verbatim presence in retrieved chunks, (b) bans invented dates / closure reasons / generalisations from one plan to all legacy plans, and (c) tells the model to keep every clause traceable to a chunk clause. No clash with answer_scope_discipline (all prohibitions restated), no new skill (avoids round-6 failure mode), no sampling change (avoids round-1 double-change trap). Predicted per-bucket deltas: distractor groundedness 0.5 → 1.0; all other cells unchanged; correctness still benefits from Sections 1–4 of round 2. Predicted aggregate: 0.778 → ~0.833 (+0.055 vs round-2, +0.089 vs cached baseline).

## Outcome
Decision: **KEEP**. Score delta vs cached baseline:
+0.0893.
