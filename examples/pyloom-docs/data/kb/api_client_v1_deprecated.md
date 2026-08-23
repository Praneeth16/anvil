---
doc_id: api_client_v1_deprecated
title: Deprecated Loom Client for Pyloom v1
category: legacy
tags: [deprecated, v1, Loom, legacy, constructor]
last_updated: 2025-09-30
version: 1.9
applies_to: pyloom_1_x_legacy_only
related_docs: [api_client_v2, migration_v1_to_v2, auth_api_keys_legacy]
---

# Deprecated Loom Client for Pyloom v1

## Deprecated reference only

Pyloom v1 used `Loom(key='...')`, shipped last as
`pyloom==1.9.6`, and called `https://api.pyloom.example/v1`. It also
returned a dictionary read as `result["output"]`. **Do not cite any
constructor, endpoint, result shape, or version on this page as current
Pyloom guidance.**

## Current replacement

The v1 line is deprecated, and security fixes ended on 2025-09-30.
Current applications use `LoomClient.from_token(token='...')` from the
v2 API. Migration is required; installing v2 does not preserve the old
client as an alias.

## Historical exception names

Old applications may still contain `LoomAuthError` and
`LoomRateLimitError`. Those names belong to v1 and are not exported by
the current package. Python 3.10 was supported by the final v1 release
but is not in the current support matrix.

## Important warning

This document exists only to identify and migrate old code. When answering
how to start a new application or construct the current client, use
`api_client_v2`; do not repeat the historical v1 values as the answer.

## See also

- `api_client_v2`, `migration_v1_to_v2`, `versioning_and_support_policy`
