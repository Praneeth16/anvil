---
doc_id: disconnection_for_nonpayment
title: Disconnection for Non-Payment Process
category: regulatory
tags: [disconnection, non_payment, notice, residential]
last_updated: 2026-03-15
version: 3.0
applies_to: residential
related_docs: [late_fee_policy, payment_plan_arrears, regulatory_consumer_rights]
---

# Disconnection for Non-Payment Process

NeoVolt may not disconnect a residential customer for non-payment
without first issuing a **15-business-day disconnection notice**.
This timeline is set by State Energy Regulator Order 2024-082 and
applies to every residential account regardless of balance.

## Notice timeline

1. **Day 0** — disconnection notice issued by mail and email after
   the balance has been past due for at least 30 days. Notice
   includes the amount owed and the earliest disconnection date.
2. **Day 5** — courtesy reminder by SMS to phone numbers on file.
3. **Day 10** — second written notice with options to cure
   (full payment, payment plan enrollment, vulnerable-customer
   request).
4. **Day 15** — earliest date a field disconnection may occur.

Disconnections are not performed on Fridays, weekends, public
holidays, or the business day before a public holiday.

## How to cure

- **Pay in full** (see `payment_methods`): the notice is voided
  immediately on payment posting.
- **Enroll in a payment plan** (`payment_plan_arrears`): the notice
  is suspended as long as the plan remains current.
- **Apply for vulnerable-customer status**
  (`regulatory_consumer_rights`): the notice is held during the
  review and waived if status is granted.

## Reconnection after disconnection

- Once disconnected, reconnection requires payment of the cleared
  balance plus a **$60 reconnection fee**.
- Reconnection occurs within 1 business day of payment for remote
  meters; up to 3 business days when a field visit is needed.

## Exceptions

- A customer with **medically-essential equipment registered**
  (life support, dialysis) cannot be disconnected for non-payment;
  the account is referred to the State Energy Regulator's
  hardship-protection desk instead.
- Extreme-weather moratoria pause all disconnection actions during
  declared heatwaves and cold snaps.

## See also

- `late_fee_policy`, `payment_plan_arrears`,
  `regulatory_consumer_rights`
