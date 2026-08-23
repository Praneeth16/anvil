# Pyloom documentation domain

This example gives ANVIL a second evaluation domain. Pyloom is a fictional
Python library, and the runtime agent answers developer questions from a small
documentation knowledge base. The optimizer can rewrite the example's prompt
scaffold while the golden set checks whether each mutation improves answers
without accepting legacy APIs, near-match pages, or unsupported claims.

Everything needed for the domain lives in this directory: 14 documentation
pages, a 20-row golden set, a starting scaffold, and the domain's own immutable
`harness/config.yaml`. No library code is changed or copied.

```
examples/pyloom-docs/
├── data/kb/*.md            14 pages; frontmatter doc_id + prose
├── data/golden_set.jsonl   20 rows: 6 direct, 6 multi-hop,
│                           4 distractor, 4 out of scope
├── scaffold/               what the optimizer may rewrite
│   ├── harness.yaml        sampling, active skills, rules, tools
│   ├── skills/             identity, citation, deprecation, escalation
│   └── rules/              citation and scope discipline
├── harness/config.yaml     what it may NOT: endpoints, thresholds,
│                           gate, and the judge's domain description
└── eval/runs/              measured results
```

## Evaluation traps

| Trap | Evaluation bucket | Rows | Failure being tested |
| --- | --- | --- | --- |
| Current v2 client versus deprecated v1 client | direct, multi-hop, distractor | `pyloom_001`, `pyloom_007`, `pyloom_013` | Repeating `Loom(key='...')`, the v1 endpoint, or the final v1 release as current |
| Bearer tokens versus legacy API keys | multi-hop, distractor | `pyloom_008`, `pyloom_014` | Using the old header, environment variable, or key prefix for v2 |
| Streaming versus pagination | multi-hop, distractor | `pyloom_011`, `pyloom_015`, `pyloom_016` | Treating render events as collection pages, or cursors as stream events |
| Timeouts versus retries | direct, multi-hop | `pyloom_003`, `pyloom_005`, `pyloom_009` | Substituting a retry budget for a timeout or a stream idle limit for a normal request limit |
| Current exception hierarchy versus v1 names | direct, multi-hop | `pyloom_004`, `pyloom_010`, `pyloom_012` | Returning `LoomAuthError` or `LoomRateLimitError` for v2 code |
| Documented facts versus absent knowledge | out of scope | `pyloom_017` through `pyloom_020` | Answering for another library, inventing benchmark or retention data, or claiming access to private token state |

## Run the evaluation

From the repository root, run:

```bash
python scripts/evaluate.py \
  --scaffold examples/pyloom-docs/scaffold \
  --kb-dir examples/pyloom-docs/data/kb \
  --golden-set-path examples/pyloom-docs/data/golden_set.jsonl \
  --mode quick
```

Use `standard` or `full` after a quick run succeeds.

There is no flag for the runtime config, and none is needed: the loader resolves
`harness/config.yaml` as a sibling of whatever `--scaffold` points at, so
`--scaffold examples/pyloom-docs/scaffold` picks up
`examples/pyloom-docs/harness/config.yaml` automatically. That file is this
domain's own — it sets `judge_domain_name` and `judge_domain_context` so the
refusal judge knows it is grading a Python library rather than the built-in
utility-support domain, and it points the MLflow experiments at separate paths
so two domains' runs do not land in one experiment where their aggregates
invite comparison.

The agent runtime, search tool, judges, gate and mutation loop are the root
harness's, unchanged. That is the claim this example exists to test.

## What a first run looks like

The starting scaffold, `--mode quick` (8 rows), measured on a live workspace.
The record of run B is checked in at `eval/runs/quick_first_run.json`.

```
                            run A     run B
aggregate                   0.875     0.917
              correctness   0.625     0.750
   retrieval_groundedness   1.000     1.000   (6/8 scored)
  refusal_appropriateness   1.000     1.000
```

**Those are two runs of the same scaffold, on the same rows, against the same
model, with nothing changed between them.** The only difference in the two
records is `pyloom_017`, a refusal row, which one judge invocation passed and the
other failed. One row out of eight moved the aggregate by 0.042 and correctness
by 0.125.

That is the harness's most important open problem, reproduced here by accident:
judge noise is comparable in size to the improvements a round is trying to
detect, and `gate.epsilon` is `0.0`, so a strict positive delta promotes. Prefer
`--mode standard` (12 rows) over `quick` for anything you intend to act on, and
read `docs/design/failure-vs-error.md` before trusting a single round's delta.

Two more things in that output are worth understanding before you read your own.

**Groundedness scored 6 of 8, and that is correct.** The two `out_of_scope`
rows have no retrieved context to be grounded in, so the scorer reports
*nothing* for them rather than `0.0`. A zero there would be a constant
subtracted from the aggregate that looks like a quality regression and moves
with the bucket mix rather than with the agent. See `docs/decisions.md` D10.

**Correctness well below 1.000 is headroom, not breakage.** The starting
scaffold is deliberately unfinished — it is what the optimizer is supposed to
improve. `pyloom_002` (a version-support question) and `pyloom_014` (a
bearer-token distractor) failed in both runs, which makes them real targets
rather than noise.

The first attempt at this example did **not** produce a usable measurement, and
it is instructive. All four `out_of_scope` rows shipped with
`must_include: []`, which becomes an empty `expected_facts`, and MLflow's
Correctness judge raises "Missing input fields" rather than scoring. Two of
eight cases errored, 0.25 against a ceiling of 0.20, so the run refused itself
as unjudgeable and exited `2` instead of reporting a confident 0.889 built on
six surviving rows. That is the failure-vs-error machinery doing its job — see
`docs/design/failure-vs-error.md` — and `tests/test_example_domains.py` now
fails offline on an empty `must_include`, so the next person does not pay for
a live run to find it.

## Read the results

Check whether answers cite the pages that actually contain each claim. The
hard cases should show whether the scaffold keeps v2 guidance separate from
legacy docs, combines both halves of multi-hop questions, and refuses when the
knowledge base is silent.

Across optimization rounds, look for gains that survive all four buckets. A
higher aggregate caused by better direct answers is not enough if deprecated
constructors start leaking into current guidance or refusals become guesses.
