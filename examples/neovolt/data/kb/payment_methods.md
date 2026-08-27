---
doc_id: payment_methods
title: Accepted Payment Methods
category: payments
tags: [payment, autopay, card, bank_transfer, wallet]
last_updated: 2026-03-04
version: 3.1
applies_to: residential
related_docs: [billing_cycle, payment_plan_arrears]
---

# Accepted Payment Methods

NeoVolt accepts six payment methods for residential accounts. All
methods post to the account within 1 business day; autopay and the
NeoVolt wallet post same-day.

## Accepted methods

- **Bank transfer (ACH)**: free, posts in 1 business day.
- **Debit card**: free for amounts up to $500/month; a 1.5%
  convenience fee applies above that threshold.
- **Credit card** (Visa, Mastercard, Amex, Discover): a 2.5%
  convenience fee applies regardless of amount.
- **Autopay** (ACH only): scheduled for the statement issue date,
  no convenience fees.
- **NeoVolt wallet**: prepaid balance loaded by ACH or card; wallet
  payments post same-day.
- **In-person at NeoVolt service centers**: cash, check, money order;
  no convenience fees but allow 2 business days to post.

## Setting up autopay

- Online: portal.neovolt.example → "Billing" → "Manage autopay".
- Requires a verified bank account (NeoVolt makes two micro-deposits
  to confirm).
- You can pause autopay for one billing cycle without canceling.

## Exceptions

- Money orders over $1,000 require manager approval at the service
  center.
- International credit cards may incur a foreign-issuer surcharge
  passed through from the card network.
- Business accounts have separate accepted methods — see
  `payment_plan_business`.

## See also

- `billing_cycle`, `payment_plan_arrears`, `late_fee_policy`
