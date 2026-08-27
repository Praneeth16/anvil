# NeoVolt utility-support domain

NeoVolt is ANVIL's original domain: a fictional electricity & gas utility
whose runtime agent answers customer questions from a policy knowledge base.
The first ten optimization rounds ran here; the baseline, round records, and
mutation log ship in `eval/` as measured history. NeoVolt became an example
when MultiHopRAG took over as the primary domain — the 20-row golden set caps
the paired gate's power and leaves no room for a holdout (issue #15).

Everything needed for the domain lives in this directory: 24 policy pages, a
20-row golden set, the scaffold as the optimizer left it after round 10
(including its critique memory), and the domain's own immutable
`harness/config.yaml`. No library code is changed or copied.

```
examples/neovolt/
├── data/kb/*.md            24 policy pages; frontmatter doc_id + prose
├── data/golden_set.jsonl   20 rows: 6 direct, 6 multi-hop,
│                           4 distractor, 4 out of scope
├── scaffold/               what the optimizer may rewrite
│   ├── harness.yaml        sampling, active skills, rules, tools
│   ├── skills/             identity, refund, greeting, classification, escalation
│   ├── rules/              answer scope and specificity discipline
│   └── memory/             the optimizer's own round-by-round critique history
├── harness/config.yaml     what it may NOT: endpoints, thresholds,
│                           gate, and the judge's domain description
└── eval/                   measured history: baseline, rounds 1-10, mutations log
```

## Evaluation traps

| Trap | Evaluation bucket | Failure being tested |
| --- | --- | --- |
| Current $0.142/kWh tariff versus legacy Plan A $0.085 | direct, distractor | Confirming the grandfathered rate as current |
| Business versus residential payment-plan terms | multi_hop | Quoting the wrong plan's installments or down payment |
| Planned maintenance versus unplanned outage | distractor | Telling the customer to report a pre-announced outage |
| Policy knowledge versus absent knowledge | out_of_scope | Answering weather, competitor, or account-state questions the KB cannot support |

## Run the evaluation

From the repository root, run:

```bash
python scripts/evaluate.py \
  --scaffold examples/neovolt/scaffold \
  --kb-dir examples/neovolt/data/kb \
  --golden-set-path examples/neovolt/data/golden_set.jsonl \
  --mode quick
```

Use `standard` or `full` after a quick run succeeds.

There is no flag for the runtime config, and none is needed: the loader
resolves `harness/config.yaml` as a sibling of whatever `--scaffold` points
at. That file is this domain's own — it sets `judge_domain_name` and
`judge_domain_context` explicitly (they used to be the library's built-in
defaults; writing them out keeps the shipped baseline scorer-comparable now
that the default is MultiHopRAG), and it points the MLflow experiments at
separate paths so two domains' runs do not land in one experiment where their
aggregates invite comparison.

## What the measured history shows

The shipped `eval/runs/baseline.json` (aggregate 0.817 on `standard`, 12
rows, per-row scores complete) is the bar the paired gate compares against.
`eval/mutations.jsonl` and `eval/runs/round_001..010.json` record the first
campaign: ten rounds of scaffold mutations, kept and reverted, under the
pre-split 12-row regime.

Two caveats when reading that history. The earliest rounds predate the
current scorer semantics (correctness v1, groundedness v5), so their scores
are not comparable with the current baseline — see `docs/decisions.md`. And
the 12-row regime is exactly what issue #15 measured as underpowered: this
domain is kept for learning and regression-testing the harness, not for new
campaigns.
