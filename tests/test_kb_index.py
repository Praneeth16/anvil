"""Indexing a knowledge base whose frontmatter is incomplete.

``doc_id`` and ``title`` are annotated ``str``, but they come from YAML
frontmatter a document may simply not have -- and the golden set references
documents *by* ``doc_id``, so a document indexed under ``None`` is one the eval
can never match to an ``expected_doc_ids`` entry. The fallbacks were already
written; nothing exercised them, which is why a revert of the narrowing passed
the whole suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anvil.tools.search_knowledge_base import _load_kb_index


def _write(kb: Path, name: str, text: str) -> None:
    (kb / name).write_text(text, encoding="utf-8")


@pytest.mark.unit
def test_missing_doc_id_falls_back_to_the_filename_stem(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    kb.mkdir()
    _write(kb, "retries.md", "---\ntitle: Retries\n---\nexponential backoff\n")

    (doc,) = _load_kb_index(kb)
    assert doc.doc_id == "retries"


@pytest.mark.unit
def test_missing_title_falls_back_to_the_doc_id(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    kb.mkdir()
    _write(kb, "retries.md", "---\ndoc_id: retry_policy\n---\nexponential backoff\n")

    (doc,) = _load_kb_index(kb)
    assert doc.doc_id == "retry_policy"
    assert doc.title == "retry_policy"


@pytest.mark.unit
def test_no_frontmatter_at_all_still_indexes(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    kb.mkdir()
    _write(kb, "retries.md", "exponential backoff\n")

    (doc,) = _load_kb_index(kb)
    assert doc.doc_id == "retries"
    assert doc.title == "retries"


@pytest.mark.unit
@pytest.mark.parametrize("value", ["[a, b]", "42", "null", "{k: v}"])
def test_non_string_frontmatter_values_fall_back(tmp_path: Path, value: str) -> None:
    """A YAML value that parses to something other than a string is not an id.

    The narrowing is an ``isinstance`` check, not a presence check, and that is
    the difference: ``doc_id: 42`` is present and unusable.
    """
    kb = tmp_path / "kb"
    kb.mkdir()
    _write(kb, "retries.md", f"---\ndoc_id: {value}\n---\nexponential backoff\n")

    (doc,) = _load_kb_index(kb)
    assert doc.doc_id == "retries"
