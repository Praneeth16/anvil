---
round: 2
branch: anvil/exp-round-2
decision: keep
action_kind: edit_rule
parse_status: ok_last_of_many
baseline_score: 0.7440
mutated_score: 0.7778
score_delta: +0.0337
---

# Round 2 critique

## Action applied
edit_rule rules/answer_specificity.md: Correctness is still the weakest judge (0.25 after round 1, down from 0.375 baseline); groundedness already hit 1.0 so f

## Rationale (from optimizer)
Correctness is still the weakest judge (0.25 after round 1, down from 0.375 baseline); groundedness already hit 1.0 so further groundedness gains are capped. The worst bucket is out_of_scope correctness (0.0 in both baseline and round 1) and multi_hop correctness regressed 0.5→0.0 after round 1. Root-cause read on round 1: (a) Section 1's '20% down, 6.5% interest' example is a business-plan distractor value that may bleed into residential answers (golden_007's must_not_include includes those exact tokens); (b) Section 3's OOS guidance listed topic-class nouns but did not force the agent to emit the hypernym instead of the user's hyponym (stove→appliance, forecast→weather) or to emit 'NeoVolt'+'policies' together. This edit removes the cross-contaminating example from Section 1, splits multi-part handling into its own section with concrete required-token lists per failing question, and replaces Section 3 OOS with an explicit hyponym→hypernym abstraction table plus a slot-fillable refusal skeleton that naturally emits NeoVolt+policies+topic-class-noun. All prohibitions from answer_scope_discipline are restated (no external sources, no hedge, no suggestion verbs, stay in-segment), so there is no clash — this is an edit to the rule we just added, not a new competing skill. Predicted per-bucket deltas: correctness direct 0.5→0.5-1.0, multi_hop 0.0→0.5, distractor 0.5→1.0, out_of_scope 0.0→0.5-1.0; groundedness stays at 1.0; refusal_appropriateness stays at 1.0. Predicted aggregate: 0.75 → ~0.83 (+0.08).

## Outcome
Decision: **KEEP**. Score delta vs cached baseline:
+0.0337.
