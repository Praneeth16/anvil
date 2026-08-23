---
doc_id: errors_and_exceptions
title: Errors and Exceptions
category: api
tags: [exceptions, errors, PyloomError, RateLimitError, request_id]
last_updated: 2026-08-16
version: 2.2
applies_to: pyloom_v2
related_docs: [retries_and_backoff, timeouts, logging_and_debugging]
---

# Errors and Exceptions

## Exception hierarchy

All public SDK errors inherit from `PyloomError`. Current subclasses are
`AuthenticationError` for HTTP 401, `PermissionDeniedError` for HTTP 403,
`ValidationError` for HTTP 422, `RateLimitError` for HTTP 429,
`RequestTimeoutError` for client timeouts, `StreamDisconnectedError` for
interrupted streams, and `PageExpiredError` for HTTP 410 cursors.

## Diagnostic fields

Server-backed exceptions expose `.request_id`, `.status_code`, and
`.response_body`. The response body may contain user input, so sanitize it
before attaching it to a public issue.

## Catching errors

Catch the narrow class you can recover from, then catch `PyloomError` at
the application boundary. Do not catch `Exception` around a render and
silently convert programming errors into empty output.

```python
from pyloom import PyloomError, RateLimitError

try:
    result = client.render(template="digest", context=payload)
except RateLimitError as exc:
    queue_for_later(exc.request_id)
except PyloomError as exc:
    record_failure(exc.request_id)
```

## See also

- `retries_and_backoff`, `timeouts`, `logging_and_debugging`
