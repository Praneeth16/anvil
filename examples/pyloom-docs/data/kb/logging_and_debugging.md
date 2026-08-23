---
doc_id: logging_and_debugging
title: Logging and Debugging
category: operations
tags: [logging, debug, PYLOOM_LOG_LEVEL, X-Request-ID, request_id]
last_updated: 2026-08-11
version: 1.6
applies_to: pyloom_v2
related_docs: [errors_and_exceptions, auth_tokens, retries_and_backoff]
---

# Logging and Debugging

## Enable SDK logs

Set `PYLOOM_LOG_LEVEL=DEBUG` to enable diagnostic output from the
`pyloom` logger. Each server response includes `X-Request-ID`; the same
value is available as `.request_id` on a `PyloomError`. Record that value
when reproducing a failed request.

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("pyloom").setLevel(logging.DEBUG)
```

## Secret safety

Debug logs redact the `Authorization` header and recognized token values.
Application-level logging can still expose context data, so do not log
complete render payloads unless your own data policy permits it.

## Issue report checklist

Include the Pyloom version, Python version, smallest reproducible example,
exception class, and request ID. Remove tokens and user data before filing
the issue.

## See also

- `errors_and_exceptions`, `auth_tokens`, `retries_and_backoff`
