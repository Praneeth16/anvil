---
doc_id: late_fee_policy
title: Late Fee and Grace Period
category: billing
tags: [late_fee, grace_period, arrears, residential]
last_updated: 2026-02-10
version: 2.2
applies_to: residential
related_docs: [billing_cycle, payment_plan_arrears, disconnection_for_nonpayment]
---

# Late Fee and Grace Period

A late fee is applied when payment is not received by the due date
shown on the statement. NeoVolt provides a **5-calendar-day grace
period** after the due date during which no late fee accrues. Late
fees are flat, not compounded, and do not by themselves trigger
disconnection.

## Fee structure

- **Flat late fee**: $7.50 per overdue balance, applied once per
  billing cycle, regardless of the amount owed.
- **No interest**: NeoVolt does not charge interest on residential
  arrears.
- **Stacking**: a second late fee is applied on the next statement
  if the prior balance plus the new charges remain unpaid past the
  next due date.

## Grace period

- The 5-day grace period starts on the day after the due date.
- A payment posted within the grace period clears the cycle without
  a late fee, even if the funds settle one day later for ACH.

## Exceptions

- Customers on a **payment plan** under `payment_plan_arrears` are
  not charged late fees as long as they meet the plan's installment
  schedule.
- Customers in **vulnerable-customer status** under
  `regulatory_consumer_rights` have late fees waived.
- Income-qualified customers on the State Affordability Rider have
  late fees capped at $3.00 per cycle.

## Relationship to disconnection

A late fee on its own never triggers disconnection. Disconnection
for non-payment requires a separate notice process — see
`disconnection_for_nonpayment`.

## See also

- `billing_cycle`, `payment_plan_arrears`,
  `disconnection_for_nonpayment`, `regulatory_consumer_rights`
