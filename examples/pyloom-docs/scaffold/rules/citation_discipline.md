---
rule_id: citation_discipline
kind: answer_constraint
applies_to: runtime
priority: high
created_at: 2026-08-23
---

# Citation discipline

Search before making an in-scope factual claim. Cite only `doc_id` values that
the tool returned and place each citation next to the claim it supports.

Do not cite a related page merely because it appeared in search results. For
multi-part questions, retrieve enough pages to support every part. If current
and deprecated pages conflict, the page whose scope matches the user's version
controls the answer.
