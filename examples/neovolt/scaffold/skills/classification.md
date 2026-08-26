---
skill_id: classification
applies_to: first_turn_after_intent
priority: 2
created_at: 2026-04-21
---

# Classification

Classify every inbound request into exactly one of these categories
before proposing an action:

- `billing` — charges, invoices, refunds, plan changes, failed payments
- `technical` — bugs, outages, feature-not-working, integration errors
- `account` — login issues, access, permissions, profile updates
- `abuse` — policy violations, harassment, fraud reports
- `other` — anything that does not fit above; escalate if unsure

Call `route_ticket` with the chosen category once you are confident.
If the request spans two categories (e.g. "I can't log in AND I was
charged twice"), pick the one with higher customer impact and note the
secondary category in the ticket.

When a request is ambiguous, ask **one** clarifying question before
classifying — not a list. Do not guess silently.
