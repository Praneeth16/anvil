---
doc_id: outage_compensation
title: Outage Compensation Credits
category: outages
tags: [outage, compensation, credit, residential, sla]
last_updated: 2026-03-08
version: 2.4
applies_to: residential
related_docs: [outage_reporting, regulatory_consumer_rights]
---

# Outage Compensation Credits

Residential customers are eligible for an automatic bill credit when
an **unplanned** outage at their premises lasts more than **12
continuous hours**. The credit is set by State Energy Regulator
Order 2024-082 and applied to the customer's next statement without
the need to file a claim.

## Credit structure

- **First 12 hours**: no credit (within service-level
  expectations).
- **12–24 hours**: $30 credit.
- **24–48 hours**: $60 credit.
- **Beyond 48 hours**: $30 per additional 24-hour block.

## How NeoVolt determines duration

- Smart meters: outage start and end timestamps from the meter.
- Non-smart meters: outage report time from the customer (see
  `outage_reporting`) plus restoration time logged by dispatch.

## Eligibility conditions

- Outage must be **unplanned** — planned-maintenance outages are
  excluded; see `outage_planned_maintenance`.
- Outage must be **at the customer's premises**, confirmed by the
  meter or dispatch records.
- The customer must have reported the outage within the active
  window if they have a non-smart meter.

## Vulnerable customers

Vulnerable-customer status (see `regulatory_consumer_rights`)
doubles all credit amounts above and grants priority restoration.

## Force majeure

Credits do not apply during outages caused by declared natural
disasters (e.g. major storms with state-of-emergency declarations).
A statement-level note explains the exclusion when applicable.

## See also

- `outage_reporting`, `regulatory_consumer_rights`,
  `outage_planned_maintenance`
