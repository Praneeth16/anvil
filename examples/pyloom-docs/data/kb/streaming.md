---
doc_id: streaming
title: Streaming Render Output
category: api
tags: [streaming, client.stream, TextDelta, StreamHeartbeat, events]
last_updated: 2026-08-15
version: 2.1
applies_to: pyloom_v2_streaming_extra
related_docs: [install, timeouts, errors_and_exceptions]
---

# Streaming Render Output

## Event stream

`client.stream(...)` yields `TextDelta` events; append each event's
`.text` value to build the output. The server emits a `StreamHeartbeat`
every 17 seconds while work is active and ends with `StreamCompleted`.
An idle stream is closed after 83 seconds.

```python
from pyloom import TextDelta

with client.stream(template="report", context=payload) as events:
    for event in events:
        if isinstance(event, TextDelta):
            display(event.text)
```

## Disconnect behavior

If the connection breaks after content has arrived, the SDK raises
`StreamDisconnectedError` and does not replay the render automatically.
The application may restart only when its render operation is safe to
repeat.

## Timeout boundary

The normal read timeout covers the opening response before the first
event. After that, the stream's idle limit applies. Configure request
timeouts as described in `timeouts`.

## See also

- `install`, `timeouts`, `errors_and_exceptions`
