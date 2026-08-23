---
doc_id: auth_tokens
title: Bearer Token Authentication
category: authentication
tags: [token, bearer, Authorization, PYLOOM_TOKEN, scopes]
last_updated: 2026-08-20
version: 2.5
applies_to: pyloom_v2
related_docs: [api_client_v2, auth_api_keys_legacy, logging_and_debugging]
---

# Bearer Token Authentication

## Current authentication

Pyloom v2 uses bearer tokens. Store the token in `PYLOOM_TOKEN` and send
`Authorization: Bearer pyl_live_...` over HTTPS. Live tokens begin with
`pyl_live_`; sandbox tokens begin with `pyl_test_`. A render normally
needs both `templates:read` and `renders:write`.

## Token handling

Create and revoke tokens in the Pyloom developer console. The SDK reads
the token value you pass to `LoomClient.from_token`; it does not discover
credentials from a browser session. Never log a complete token or commit
one to a repository.

## Scope failures

A valid token without the required scope receives a permission response.
An expired or revoked token receives an authentication response. The
exception mapping is documented in `errors_and_exceptions`.

## See also

- `api_client_v2`, `auth_api_keys_legacy`, `errors_and_exceptions`
