---
skill_id: refund
applies_to: billing_requests
priority: 4
created_at: 2026-04-21
---

# Refund

Use this skill only after `classification` has routed a request as
`billing` and the customer is asking for money back.

Decision logic:

1. Call `apply_policy` with `policy=refund_eligibility` and the
   customer's `order_id`. Trust its response.
2. If the policy result is `eligible` and the amount is `<= $500`,
   issue the refund by calling `apply_policy` with
   `policy=issue_refund` and the approved amount.
3. If `amount > $500`, do not self-approve. Escalate per the
   `escalation` skill.
4. If the policy result is `ineligible`, tell the customer the specific
   reason returned by the policy call. Do not paraphrase it into
   something softer — the customer should be able to act on it.

Never promise a refund before `apply_policy` has responded. Never issue
a refund for an order you did not verify via `lookup_customer` first.
