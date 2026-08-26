---
doc_id: payment_plan_business
title: Small-Business Payment Plan for Arrears
category: payments
tags: [payment_plan, business, arrears, small_business]
last_updated: 2026-03-01
version: 2.0
applies_to: small_business
related_docs: []
---

# Small-Business Payment Plan for Arrears

Small-business accounts have a separate payment-plan program from
residential customers. Terms are **less generous** than the
residential plan in `payment_plan_arrears`; this is a regulator
requirement, not a NeoVolt policy choice.

## Eligibility

- Account is classified as "Small Business" — annual peak demand
  under 200 kW.
- The arrears balance is between **$200 and $15,000**.
- The business has been a NeoVolt customer for at least 12 months.

## Plan terms

- **Length**: 3 or 6 installments only. Longer terms are not
  available.
- **Down payment**: **20% of the arrears** must be paid up front to
  start the plan.
- **Interest**: a **6.5% annual rate** is applied to the unpaid
  arrears balance over the plan period.
- **Late fees**: continue to accrue on the current monthly bill;
  only the arrears portion is shielded by the plan.

## How to enroll

- Online: portal.neovolt.example/business → "Billing" → "Set up
  payment plan".
- Account manager: business customers with a dedicated account
  manager should request enrollment through them.

## Default

- Missing **one** installment defaults the plan immediately. The
  remaining balance plus accrued interest is added to the next
  bill, and disconnection notice timelines apply.

## Important

- Vulnerable-customer protections (`regulatory_consumer_rights`) do
  **not** apply to business accounts.
- Residential customers running a small business out of their home
  are still eligible for the residential plan in
  `payment_plan_arrears` — this page does not apply to them.

## See also

- `payment_methods` (note: business accounts have separate accepted
  payment methods)
