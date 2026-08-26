# MultiHopRAG

The golden set and knowledge base in this directory are derived from
**MultiHop-RAG** (`yixuantt/MultiHopRAG` on Hugging Face), pinned at revision
`71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82`.

- **Paper:** Tang, Yixuan and Yang, Yi. "MultiHop-RAG: Benchmarking
  Retrieval-Augmented Generation for Multi-Hop Queries." arXiv:2401.15391
  (2024). https://arxiv.org/abs/2401.15391
- **Source:** https://huggingface.co/datasets/yixuantt/MultiHopRAG
- **License:** Open Data Commons Attribution License (ODC-BY 1.0),
  https://opendatacommons.org/licenses/by/1-0/

## Derivation

Rows are sampled from the 2,556-query dataset by `scripts/build_multihop_domain.py`
(seed-pinned, reproducible via `--check`): `multi_hop` from comparison /
inference / temporal queries, `out_of_scope` from `null_query` rows,
`distractor` from queries with same-category confusables retained in the KB.
The `direct` bucket is authored (MultiHopRAG contains no single-document
queries) and marked as such in each row's `notes_for_judge`. The knowledge
base is the subset of the 609-document corpus cited by the sampled rows plus
distractor confusables and seeded filler. Document bodies are unmodified.
