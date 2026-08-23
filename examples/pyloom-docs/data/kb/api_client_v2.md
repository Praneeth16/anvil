---
doc_id: api_client_v2
title: LoomClient API for Pyloom v2
category: api
tags: [LoomClient, client, v2, constructor, lifecycle]
last_updated: 2026-08-19
version: 2.7
applies_to: pyloom_2_x
related_docs: [auth_tokens, migration_v1_to_v2, timeouts]
---

# LoomClient API for Pyloom v2

## Current construction

The supported v2 client is created with
`LoomClient.from_token(token='pyl_live_...')`. Pass the keyword
`token`, not an API-key argument. `LoomClient` targets
`https://api.pyloom.example/v2` unless an explicit test endpoint is
configured. Use a **context manager** for scripts so the transport closes.

```python
from pyloom import LoomClient

with LoomClient.from_token(token='pyl_live_...') as client:
    result = client.render(template="receipt", context={"total": "19.75"})
```

## Lifecycle

Services may keep one thread-safe client for the process lifetime and call
`client.close()` during shutdown. Creating one client per render adds
unnecessary connection setup.

## Core methods

- `client.render(...)` returns one `RenderResult`.
- `client.stream(...)` yields incremental stream events.
- `client.templates.list(...)` returns one page of template metadata.

## See also

- `auth_tokens`, `streaming`, `pagination`, `migration_v1_to_v2`
