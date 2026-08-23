---
doc_id: retries_and_backoff
title: Retries and Backoff
category: reliability
tags: [retries, backoff, Retry-After, rate_limit, transient]
last_updated: 2026-08-14
version: 2.3
applies_to: pyloom_v2_http_requests
related_docs: [timeouts, errors_and_exceptions, logging_and_debugging]
---

# Retries and Backoff

## Default policy

The v2 client defaults to `max_retries=3`, which permits
**4 total attempts** including the initial request. Backoff waits are 0.4 seconds,
1.2 seconds, and 3.6 seconds before jitter. A server `Retry-After` value
is honored but capped at 27 seconds.

## Retried failures

The client retries connection resets and HTTP 408, 429, 502, 503, and
504 responses. It does not retry HTTP 400, 401, 403, or 422 responses.
Once a stream has yielded its first content event, stream recovery follows
`streaming` rather than this request policy.

## Configuration

Set a different budget with `RetryPolicy(max_retries=...)` and pass the
policy when creating the client. Keep retry budgets bounded in worker
queues so an upstream incident does not consume every worker slot.

## Exhaustion

After the budget is exhausted, the client raises the specific exception
for the final response. See `errors_and_exceptions` for the class mapping
and `logging_and_debugging` for request correlation.

## See also

- `timeouts`, `errors_and_exceptions`, `streaming`
