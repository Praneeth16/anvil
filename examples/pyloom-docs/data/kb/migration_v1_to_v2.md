---
doc_id: migration_v1_to_v2
title: Migrate from Pyloom v1 to v2
category: migration
tags: [migration, v1, v2, LoomClient, RenderResult]
last_updated: 2026-07-28
version: 2.4
applies_to: applications_upgrading_from_1_x
related_docs: [api_client_v1_deprecated, api_client_v2, auth_tokens, errors_and_exceptions]
---

# Migrate from Pyloom v1 to v2

## Required code changes

Replace the legacy client import with `from pyloom import LoomClient`,
construct it with `LoomClient.from_token(...)`, and replace the old run
call with `client.render(...)`. The v2 return type is `RenderResult`, and
rendered content is read from `RenderResult.text`.

## Credential change

Replace the legacy API-key secret with `PYLOOM_TOKEN` and issue a bearer
token with the scopes your application needs. The old credential is not
accepted by the v2 endpoint.

## Error handling and lifecycle

Replace v1 exception imports with the current hierarchy documented in
`errors_and_exceptions`. Also add explicit client cleanup: a context
manager is appropriate for scripts, while services should close their
shared client during shutdown.

## Rollout sequence

Test the new result shape before switching production traffic. Do not run
v1 and v2 clients against the same mutable render job because retries from
both deployments can duplicate work.

## See also

- `api_client_v2`, `auth_tokens`, `errors_and_exceptions`
