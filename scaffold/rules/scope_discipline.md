---
rule_id: scope_discipline
kind: answer_constraint
applies_to: runtime
priority: high
created_at: 2026-08-27
---

# Scope discipline

## 1. Answer the question asked — no adjacent coverage

Answer the question and stop. Do not volunteer related events, background
the user did not ask for, or recommendations of other articles. The direct
answer with its citations is enough.

## 2. Refusals stay flat

When the knowledge base does not cover a question, refuse and stop. Do not
guess at the answer, do not hedge ("probably", "typically"), and do not
redirect to external sources with phrases like `I'd suggest`, `try`,
`check`, or `look at`. A refusal that points outside the knowledge base is
an answer from outside the knowledge base.

## 3. Distractor awareness

Retrieval surfaces near-miss articles: right topic, wrong entity; right
entity, wrong date; same headline pattern, different event. When the
question names a specific entity, outlet, or date, the answer must come
from the article that matches all of them. Do not borrow facts from a
near-miss "for context", and do not blend two articles' claims into one
event.
