---
rule_id: citation_discipline
kind: answer_constraint
applies_to: runtime
priority: high
created_at: 2026-08-27
---

# Citation discipline

Three constraints apply to every response that uses the knowledge base.

## 1. No uncited facts

Every article-derived statement names its `doc_id`. A claim without a
citation reads as invented, even when it happens to be true.

## 2. Citations must be real and retrieved

A `doc_id` that was not returned by `search_knowledge_base` in this
conversation is a fabrication, whatever it resembles. Cite only doc_ids
from actual search results.

## 3. Same category is not same content

The knowledge base holds many confusable articles: same outlet, same
topic, different entity, event, or date. Retrieval will surface them
together. Before citing, check that the article contains the specific
claim — a title match on the topic is not evidence of the fact.
