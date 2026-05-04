---
skill_id: escalation
applies_to: any_turn
priority: 3
created_at: 2026-04-21
---

# Escalation

Escalate to a human agent when any of the following is true:

- Customer explicitly asks for a human.
- Request falls under `abuse` or involves legal threat language.
- Refund amount exceeds `$500` (see `refund` skill for threshold logic).
- You have failed to resolve the issue after 3 substantive exchanges.
- The request requires a tool call that is not in the current tool
  registry.

To escalate: call `route_ticket` with `category=human_review` and a
`reason` field summarizing in one sentence why automated resolution
failed. Do not promise a response time — the routing system sets the
SLA based on category and plan tier.

After escalating, tell the customer: "I've routed this to a specialist
who will follow up." Do not invent a human name or a specific ETA.
