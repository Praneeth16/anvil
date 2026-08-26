---
doc_id: meter_industrial_advanced
title: Industrial AMI Meters
category: meters
tags: [meter, industrial, ami, large_load]
last_updated: 2026-01-30
version: 1.3
applies_to: industrial
related_docs: []
---

# Industrial AMI Meters

Industrial customers (those with peak demand exceeding 200 kW) use
**Advanced Metering Infrastructure (AMI) meters**, which are
distinct from residential smart meters. AMI meters measure both
real and reactive power and provide 1-minute interval data to
NeoVolt's network operations center.

## Capabilities

- 1-minute interval data on real (kW) and reactive (kVAR) power.
- Power-quality logging: voltage sags, transients, harmonic
  distortion.
- Bi-directional metering for sites with on-site generation.
- Demand-charge calculation in real-time, used by industrial
  tariffs.

## Tariff applicability

- AMI meters are paired with the **industrial demand tariff**, which
  has a per-kW peak-demand charge in addition to per-kWh energy
  charges.
- Industrial customers are **not eligible** for the residential
  Time-of-Use tariff (`tariff_time_of_use`); the demand tariff
  serves the equivalent purpose for industrial loads.
- Residential smart meters (`smart_meter_required_for_tou`) are not
  installable at industrial sites — voltage class and current
  ratings are different.

## Installation

- AMI meter installation is coordinated through the customer's
  NeoVolt industrial account manager.
- Lead time is typically **30–60 business days** from order to
  install, including engineering review and current-transformer
  selection.

## Data access

- Industrial customers can pull interval data via the dedicated
  industrial portal (industrial.neovolt.example) or NeoVolt's
  energy-data API.
- Data retention: 13 months on the portal; longer-term archives
  available on request.

## Important

This page applies **only** to industrial sites. Residential
customers — including small businesses operating out of a home —
should consult `meter_readings_self` and
`smart_meter_required_for_tou` instead. Putting an AMI meter on a
residential service is not supported.
