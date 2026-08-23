---
doc_id: pagination
title: Paginating Collection Results
category: api
tags: [pagination, page_size, cursor, next_cursor, templates]
last_updated: 2026-08-13
version: 2.0
applies_to: pyloom_v2_collection_methods
related_docs: [api_client_v2, errors_and_exceptions, streaming]
---

# Paginating Collection Results

## Page requests

`client.templates.list(page_size=50)` returns a `Page[Template]` with
`.items` and `.next_cursor`. The maximum accepted page size is 250.
Pass the returned cursor into the next list call; cursors expire after
13 minutes and then raise `PageExpiredError`.

```python
page = client.templates.list(page_size=50)
for template in page.items:
    print(template.name)

if page.next_cursor:
    page = client.templates.list(cursor=page.next_cursor, page_size=50)
```

## Automatic iteration

`client.templates.iter_all(page_size=50)` follows cursors lazily and
stops when `next_cursor` is absent. It still makes discrete page requests;
it is not a render stream.

## Cursor rules

Cursors are opaque and tied to the original filter and sort order. Do not
edit a cursor, persist it as a durable checkpoint, or reuse it with a
different collection.

## See also

- `api_client_v2`, `errors_and_exceptions`, `streaming`
