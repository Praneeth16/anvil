---
doc_id: smart_meter_required_for_tou
title: Smart Meter Requirement and Installation for TOU
category: meters
tags: [meter, smart_meter, tou, installation, residential]
last_updated: 2026-03-22
version: 1.6
applies_to: residential
related_docs: [tariff_time_of_use, meter_readings_self]
---

# Smart Meter Requirement and Installation for TOU

Time-of-use billing requires interval data, which only smart meters
provide. If your premises has a legacy electromechanical meter, you
must request a **smart-meter upgrade** before enrolling in
`tariff_time_of_use`. The upgrade is **free** for residential
customers.

## Eligibility

- Standard residential service in any of NeoVolt's metro and
  suburban service zones.
- The premises must have a meter base accessible from outside or a
  utility-room location reachable without the customer's presence.

## Installation process

1. Request a smart-meter install via portal.neovolt.example →
   "Meters" → "Request smart meter".
2. NeoVolt offers a 4-hour appointment window within **10 business
   days** of the request.
3. Installation takes about 30 minutes; power is interrupted for
   roughly 5 minutes during the swap.
4. Once installed, interval data starts flowing within 24 hours.
   Enrollment in TOU can then be activated for the next billing
   cycle.

## What changes after install

- No more self-readings (see `meter_readings_self` exceptions).
- Outage detection becomes automatic for your premises.
- Hourly usage data appears in the portal within ~24 hours of
  consumption.

## Exceptions

- **Rural service zones** without smart-meter coverage are listed
  on a network-build-out schedule; TOU is not available there until
  coverage reaches the address.
- **Industrial AMI meters** are different from residential smart
  meters — see `meter_industrial_advanced`.
- A customer may **opt out** of smart-meter installation, in which
  case TOU is not available.

## See also

- `tariff_time_of_use`, `meter_readings_self`,
  `meter_industrial_advanced`
