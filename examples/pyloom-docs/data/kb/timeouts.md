---
doc_id: timeouts
title: Request Timeouts
category: reliability
tags: [timeouts, connect, read, write, pool, RequestTimeoutError]
last_updated: 2026-08-12
version: 1.7
applies_to: pyloom_v2_http_requests
related_docs: [retries_and_backoff, streaming, errors_and_exceptions]
---

# Request Timeouts

## Default thresholds

The v2 defaults are a **6.5 seconds** connect timeout, **41 seconds**
read timeout, **19 seconds** write timeout, and **2.25 seconds** pool
timeout. A breached threshold raises `RequestTimeoutError`. There is no
unbounded request mode in the supported client.

## Configuration

Pass `Timeout(connect=6.5, read=41.0, write=19.0, pool=2.25)` when
creating the client. Set each phase deliberately; a large read timeout
does not help a connection that cannot be established.

## Retries and streams

A timeout does not increase the configured retry budget. The read timeout
covers the streaming handshake until the first event; after that,
`streaming` defines the idle limit.

## See also

- `retries_and_backoff`, `streaming`, `errors_and_exceptions`
