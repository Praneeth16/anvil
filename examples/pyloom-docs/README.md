# Pyloom documentation domain

This example gives ANVIL a second evaluation domain. Pyloom is a fictional
Python library, and the runtime agent answers developer questions from a small
documentation knowledge base. The optimizer can rewrite the example's prompt
scaffold while the golden set checks whether each mutation improves answers
without accepting legacy APIs, near-match pages, or unsupported claims.

Everything needed for the domain lives in this directory: 14 documentation
pages, a 20-row golden set, and a starting scaffold. The root runtime and
evaluation code are reused unchanged.

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

Use `standard` or `full` after a quick run succeeds. The same root harness
configuration, agent runtime, search tool, judges, and mutation loop apply to
this domain.

## Read the results

Check whether answers cite the pages that actually contain each claim. The
hard cases should show whether the scaffold keeps v2 guidance separate from
legacy docs, combines both halves of multi-hop questions, and refuses when the
knowledge base is silent.

Across optimization rounds, look for gains that survive all four buckets. A
higher aggregate caused by better direct answers is not enough if deprecated
constructors start leaking into current guidance or refusals become guesses.
