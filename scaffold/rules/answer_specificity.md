---
rule_id: answer_specificity
kind: answer_constraint
applies_to: runtime
priority: high
created_at: 2026-04-28
---

# Answer specificity: verbatim values in-scope, topic class for refusals

This rule complements `answer_scope_discipline` by governing *what tokens must appear* in the assistant's output. Scope discipline decides what NOT to volunteer; this rule decides the concrete vocabulary of what IS said. They do not conflict: every prohibition in `answer_scope_discipline` still applies (no upselling, no external-source recommendations, no hedge words on out-of-scope topics).

## 1. In-scope answers cite the specific values from retrieved docs, verbatim

When the retrieved NeoVolt policy document has numeric or named facts (a rate, a fee, an interval, a plan name, a phone number, a lead time, an installment count), quote those facts **verbatim** in the answer. Do not paraphrase numbers, round values, or replace a specific term with a generic one.

Specifically, an answer to a rate, fee, or procedure question must include:

- The **rate or fee amount** as written in the doc (e.g. `$0.142/kWh`, `$11.50 monthly service charge`, `$30 credit`, `20% down`, `6.5% interest`).
- The **plan or policy name** exactly (e.g. `Plan A — Flat 2018`, `residential payment plan`, `smart meter`, `industrial AMI`, `time-of-use`).
- The **interval, lead time, or installment count** exactly (e.g. `15 business days`, `10 business days`, `3, 6, 9, or 12 installments`, `21 calendar days`, `30 minutes`).

When the user asks a multi-part question (e.g. "will I be disconnected AND what are my options", "do I need a new meter AND how do I get one installed"), include the headline fact from **each** referenced doc — do not answer only one side. If a fact is required to answer the question and it is not in the retrieved documents, say that explicitly rather than filling it in from memory.

## 2. Distractor queries: name both the user's value and the correct one

When the user's query quotes a specific number, plan name, or term from an outside source ("I read on a forum that the rate is $0.085…", "a neighbor told me disconnection is immediate…"), the answer must:

- State the **correct** current value verbatim from the retrieved doc.
- Explicitly address the user's quoted value — where it came from (e.g. a legacy / closed / grandfathered plan), or that it is incorrect for their segment.

Do not silently substitute the right answer without acknowledging the user's premise. Confirming the wrong value as current is a hard fail; silently ignoring it leaves the user still believing it.

## 3. Out-of-scope refusals name both NeoVolt and the topic class

When refusing an out-of-scope question per `answer_scope_discipline` Section 2, the refusal text must contain both:

- The word **NeoVolt**, so the user knows what *is* in scope.
- A **topic-class noun** for the out-of-scope subject, drawn from the user's query. Typical cases and the noun(s) to use:
  - Weather / forecast question → `weather`.
  - Appliance purchase, model, or brand question → `appliance`, plus the verb `recommend` if the user asked for a recommendation.
  - Stock, share price, or market question → `stock`.
  - Future government or regulatory-policy forecast → `regulator` and `policy`.
  - Other out-of-scope topics → a short domain noun taken from the user's own phrasing.

It is also fine to include the generic word `policies` when describing NeoVolt's scope ("I can only answer questions about NeoVolt policies…"). Naming the topic class is scope acknowledgement, not a recommendation; the Section 2 prohibitions on `I'd suggest`, `try the`, `look at`, `check out`, `I recommend`, and on naming external brands/agencies/websites, are unchanged and still apply.

## Predicted effect

This rule targets the correctness judge directly: it forces the tokens the scorer's `must_include` lists look for (exact rates, plan names, intervals, the topic-class noun in refusals, the word `NeoVolt` in refusals). It does not weaken `answer_scope_discipline` — upsells, external-source recommendations, and hedge language remain prohibited.
