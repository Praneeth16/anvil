---
doc_id: auth_api_keys_legacy
title: Legacy API Key Authentication
category: legacy
tags: [deprecated, api_key, X-Pyloom-Key, PYLOOM_API_KEY, v1]
last_updated: 2025-08-12
version: 1.8
applies_to: pyloom_v1_legacy_only
related_docs: [auth_tokens, api_client_v1_deprecated, migration_v1_to_v2]
---

# Legacy API Key Authentication

## Deprecated v1 scheme

Pyloom v1 read `PYLOOM_API_KEY` and sent
`X-Pyloom-Key: loomkey_...`. The key prefix was `loomkey_`, and the
header was accepted only by the v1 endpoint. These values are retained
for migration diagnostics, not for current authentication guidance.

## Do not use for v2

Legacy API keys cannot authenticate to the v2 endpoint and cannot be
converted in place. Create a new bearer token, move the secret to the
current environment variable, and update the client construction.

## Revocation during migration

Run old and new credentials in separate deployment secrets while testing.
After the v2 deployment succeeds, revoke the legacy key in the developer
console rather than leaving both credential types active.

## See also

- `auth_tokens`, `api_client_v1_deprecated`, `migration_v1_to_v2`
