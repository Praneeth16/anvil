---
doc_id: versioning_and_support_policy
title: Versioning and Support Policy
category: project_policy
tags: [versioning, support, semantic_versioning, release, deprecation]
last_updated: 2026-08-10
version: 1.5
applies_to: all_pyloom_releases
related_docs: [install, migration_v1_to_v2, api_client_v1_deprecated]
---

# Versioning and Support Policy

## Current release

The current stable release is **2.7.3**. Pyloom follows
**semantic versioning**: major releases may break public APIs, minor releases add
backward-compatible features, and patch releases contain compatible fixes.
Each v2 minor line receives fixes for **18 months** after its first release.

## Supported environments

Only the Python versions listed in `install` are tested for the current
release. A package importing on an older interpreter does not make that
runtime supported.

## Deprecation process

Public APIs are announced as deprecated in release notes before removal in
a later major release. The v1 client is already outside its fix window;
`api_client_v1_deprecated` exists as a migration reference, not as a
supported alternative.

## Reporting support issues

For behavior that contradicts these docs, collect the diagnostic details
from `logging_and_debugging` and file a minimal issue. Maintainers do not
provide private account or hosted-service status through the SDK docs.

## See also

- `install`, `migration_v1_to_v2`, `api_client_v1_deprecated`
