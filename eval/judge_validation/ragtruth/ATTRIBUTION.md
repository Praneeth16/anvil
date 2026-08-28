# RAGTruth

The rows in this directory are a stratified slice of **RAGTruth**
(`ParticleMedia/RAGTruth` on GitHub), pinned at revision
`c103204b9ce28d6bbad859304bf30de72b8ed8fe`.

- **Paper:** Niu et al., "RAGTruth: A Hallucination Corpus for Developing
  Trustworthy Retrieval-Augmented Language Models" (ACL 2024).
- **Source:** https://github.com/ParticleMedia/RAGTruth
- **License:** MIT, https://github.com/ParticleMedia/RAGTruth/blob/main/LICENSE

## Derivation

Sampled by `scripts/build_ragtruth_slice.py` (seed-pinned, reproducible via
`--check`): 50 rows per (task_type, supported) cell = 300 answer rows,
plus every `incorrect_refusal` row. `truncated` rows are excluded. The
slice validates the harness's judges against human labels; it is not an
eval domain and carries no fabricated expectations.
