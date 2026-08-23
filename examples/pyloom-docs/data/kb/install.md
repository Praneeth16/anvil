---
doc_id: install
title: Install Pyloom
category: getting_started
tags: [install, pip, Python, version]
last_updated: 2026-08-17
version: 2.6
applies_to: pyloom_v2
related_docs: [quickstart, versioning_and_support_policy, streaming]
---

# Install Pyloom

## Supported installation

Install the current stable release with `pip install "pyloom==2.7.3"`.
Pyloom 2.7.3 supports **CPython 3.11, 3.12, and 3.13** on Linux, macOS,
and Windows. Other Python runtimes are not part of the supported test
matrix.

```bash
python -m pip install "pyloom==2.7.3"
python -c "import pyloom; print(pyloom.__version__)"
```

## Optional streaming extra

For asynchronous streaming support, install
`pip install "pyloom[streaming]==2.7.3"`. The base package supports
synchronous rendering without this extra.

## Upgrade checks

Read `migration_v1_to_v2` before upgrading an application from the 1.x
line. A clean virtual environment is recommended when changing major
versions because the public client and result types changed.

## See also

- `quickstart`, `migration_v1_to_v2`, `versioning_and_support_policy`
