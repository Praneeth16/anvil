---
round: 1
branch: anvil/exp-round-1
decision: keep
action_kind: add_rule
parse_status: ok_last_of_many
baseline_score: 0.7440
mutated_score: 0.7500
score_delta: +0.0060
---

# Round 1 critique

## Action applied
add_rule rules/answer_specificity.md: Correctness is the weakest judge (0.375) and the failure pattern is uniform across all four buckets: the agent's respons

## Rationale (from optimizer)
Correctness is the weakest judge (0.375) and the failure pattern is uniform across all four buckets: the agent's responses are missing the specific verbatim tokens the judge's must_include list checks for. Direct (golden_001) drops $0.142 / $11.50; multi_hop (golden_008) drops 'free' or '10 business days'; distractor (golden_013) drops 'Plan A' / 'legacy' / 'closed'; out_of_scope (golden_017, golden_018) drops the topic-class noun ('weather', 'appliance', 'recommend') and the word 'NeoVolt'. A new rule answer_specificity.md encodes three surgical constraints: (1) verbatim inclusion of retrieved numeric/named facts, (2) explicit acknowledgement of a distractor's near-miss premise alongside the correct value, (3) explicit 'NeoVolt' + topic-class noun in OOS refusals. It is designed to reinforce, not override, answer_scope_discipline Section 2 — every prohibition (no external sources, no hedging, no suggestion verbs) is restated. This avoids the round-6 failure mode (OOS skill vs. refusal rule clash). Predicted per-bucket deltas: correctness direct 0.5→1.0, multi_hop 0.5→1.0, distractor 0.5→1.0, out_of_scope 0.0→1.0; groundedness neutral or slightly positive (verbatim quoting improves grounding); refusal_appropriateness unchanged at 1.0. Predicted aggregate: 0.744 → ~0.83 (+0.08).

## Outcome
Decision: **KEEP**. Score delta vs cached baseline:
+0.0060.
