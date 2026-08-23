---
skill_id: deprecation
kind: answer_policy
applies_to: runtime
priority: high
created_at: 2026-08-23
---

# Deprecation handling

When a question or search result touches a deprecated API, identify it as
deprecated and answer with the current API. Use the current document for the
recommended code and the legacy document only to explain or migrate old code.

Do not present an old constructor, credential type, endpoint, exception name,
or supported runtime as the default for a new application. If the user must
remain on v1, state that the guidance is historical and point to the migration
page.
