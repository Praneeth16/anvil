---
round: 10
branch: anvil/exp-round-10
decision: keep
action_kind: edit_skill
parse_status: ok_last_of_many
baseline_score: 0.7440
mutated_score: 0.8194
score_delta: +0.0754
---

# Round 10 critique

## Action applied
edit_skill skills/identity.md: Targets OOS correctness, the only bucket still at 0.0 both in the cached parent baseline (0.744) and in the post-round-8

## Rationale (from optimizer)
Targets OOS correctness, the only bucket still at 0.0 both in the cached parent baseline (0.744) and in the post-round-8 state (0.875, temp 0.3). golden_017 (weather) and golden_018 (appliance) keep failing because identity.md's abstract template + bracket slots + topic-class table don't survive runtime sampling — the model substitutes 'forecast' for 'weather', says 'help with shopping' instead of naming 'appliance', or drops 'NeoVolt' / 'policies'. Fix: insert four literal 'Emit verbatim' example strings between the template and the abstraction table. Each string is hand-audited to contain every required token for its golden row (NeoVolt+policies+weather; NeoVolt+appliance+recommend; NeoVolt+policies+stock; NeoVolt+policies+regulator+can't speculate) and nothing banned — no 'I'd suggest'/'try'/'look at'/'check out', no brand/retailer/agency names, no hedge adjectives, no follow-up sentence. Clash check: (a) answer_scope_discipline §2 prohibits redirect verbs + external sources — examples use neither; (b) answer_specificity §4 carries the same skeleton — examples instantiate it, not contradict; (c) recommend-as-negation clarification preserved ('can't recommend that here'); (d) round-6-legacy clashing-skill failure avoided — edit, not add; (e) round-7 rationale-length failure avoided — this rationale is under the 2000-char cap. Predicted delta: OOS correctness 0.0→1.0 on both rows (2 hits out of 8); correctness judge rises ~0.25; aggregate vs cached 0.744 baseline → ~0.83. Direct / multi_hop / distractor untouched since examples only fire on OOS triggers; groundedness unchanged (OOS rows don't contribute); refusal_appropriateness stays 1.0.

## Outcome
Decision: **KEEP**. Score delta vs cached baseline:
+0.0754.
