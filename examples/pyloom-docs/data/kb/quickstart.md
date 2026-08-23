---
doc_id: quickstart
title: Pyloom Quickstart
category: getting_started
tags: [quickstart, render, LoomClient, token]
last_updated: 2026-08-18
version: 2.7
applies_to: pyloom_v2
related_docs: [install, api_client_v2, auth_tokens]
---

# Pyloom Quickstart

## First render

Pyloom's current release is **2.7.3**. Create the client with
`LoomClient.from_token(token=os.environ["PYLOOM_TOKEN"])`, then call
`client.render(template="welcome", context={"name": "Ada"})`. The call
returns a `RenderResult`; the rendered string is available as
`result.text`.

```python
import os
from pyloom import LoomClient

with LoomClient.from_token(token=os.environ["PYLOOM_TOKEN"]) as client:
    result = client.render(
        template="welcome",
        context={"name": "Ada"},
    )
    print(result.text)
```

## Before you run it

Install the pinned package described in `install`, and create a bearer
token with the `templates:read` and `renders:write` scopes described in
`auth_tokens`. Do not paste tokens into source control.

## What the example proves

The context manager closes the underlying HTTP transport after the render.
For long-lived services, create one client during application startup and
close it during shutdown.

## See also

- `install`, `api_client_v2`, `auth_tokens`
