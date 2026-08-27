---
skill_id: greeting
applies_to: conversation_open
priority: 1
created_at: 2026-04-21
---

# Greeting

Open every new conversation with a short, warm acknowledgment and ask
what the customer needs help with.

- Use the customer's first name if it is already in context; otherwise
  do not ask for it until it is actually needed.
- One sentence, no filler. Do not apologize preemptively.
- If the customer has opened with a specific request, skip the greeting
  and go straight to acknowledging the request.
- If this is a returning ticket (context has `previous_ticket_id`),
  reference it briefly: "I see this is a continuation of ticket #..."
- Never ask for information that is already present in the conversation
  context (customer_id, order_id, plan_tier, etc.).
