---
skill_id: answer_with_citation
kind: response_format
applies_to: runtime
priority: high
created_at: 2026-08-23
---

# Answer with citations

Support each factual claim with the `doc_id` of the retrieved page that states
it. Write citations in square brackets, such as `[timeouts]`, after the claim.

For a question with several parts, search for every part and cite each source
where it is used. A citation to one page does not support facts taken from a
different page.

Quote exact class names, method names, environment variables, versions, and
numeric defaults. If the retrieved text does not contain the value, do not fill
it in from memory.
