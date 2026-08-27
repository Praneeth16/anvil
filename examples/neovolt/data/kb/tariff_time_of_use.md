---
doc_id: tariff_time_of_use
title: Time-of-Use (TOU) Tariff
category: tariffs
tags: [tariff, time_of_use, peak, off_peak, residential]
last_updated: 2026-03-20
version: 2.0
applies_to: residential
related_docs: [smart_meter_required_for_tou, tariff_standard_residential]
---

# Time-of-Use (TOU) Tariff

The Time-of-Use (TOU) tariff charges different per-kWh rates by
hour of day. It rewards shifting flexible loads (laundry, dishwasher,
EV charging) outside peak hours. **Enrollment in TOU requires a
smart meter** — see `smart_meter_required_for_tou` for the install
process.

## Rate schedule (weekdays)

- **Peak**: 16:00–21:00 — **$0.281/kWh**.
- **Mid-peak**: 07:00–16:00 and 21:00–23:00 — **$0.158/kWh**.
- **Off-peak**: 23:00–07:00 — **$0.082/kWh**.

## Rate schedule (weekends and public holidays)

- All hours billed at the **mid-peak** rate of $0.158/kWh.

## Service charge

- $13.50 monthly service charge (slightly higher than the standard
  flat tariff to fund the smart-meter back-end).

## Effective dates

- Current rates effective 2026-01-01, set by State Energy Regulator
  Order 2025-114.
- Rates are reviewed annually.

## How to enroll

1. Verify you have a smart meter — `smart_meter_required_for_tou`
   covers eligibility and installation.
2. Online: portal.neovolt.example → "Tariffs" → "Switch to TOU".
3. Switch takes effect on the next billing cycle. There is no fee
   to switch in or out, but only one tariff change per 90-day
   window is allowed.

## Exceptions

- TOU is not available to industrial customers; industrial sites
  use the demand-charge tariff.
- Pre-2023 grandfathered TOU plans have different peak windows —
  see `tariff_legacy_grandfathered`.

## See also

- `smart_meter_required_for_tou`, `tariff_standard_residential`,
  `meter_readings_self`
