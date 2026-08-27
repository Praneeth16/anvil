"""Integrity checks for the primary domain at ``data/`` (MultiHopRAG).

The example-domain suite only covers ``examples/``; the domain the loop
actually runs had no offline guards of its own. These mirror that rule set
(required fields, cited docs exist, expected facts present, real traps,
refusal hygiene) and add the two things only the primary domain has:

* the exact per-bucket train/dev/test partition table the vendoring script
  pinned. ``full`` mode is defined as the whole dev partition and
  ``select_subset`` raises when a bucket is short, so a partition that drifts
  detonates at eval time — after money is spent; and
* per-mode selection satisfiability under the shipped config.

No LLM, no network. Everything here is file parsing plus the harness's own
loaders.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from anvil.data.golden_set import REQUIRED_FIELDS, load_golden_set
from anvil.eval import runner
from anvil.runtime.loader import load_harness

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# The vendoring script (scripts/build_multihop_domain.py, seed 42) pins these
# per-bucket (train, dev, test) counts for the 120-row variant. Changing the
# golden set, the ratios, or the seed without re-running it fails here first.
EXPECTED_PARTITION = {
    "direct": (8, 10, 6),
    "multi_hop": (16, 20, 12),
    "distractor": (8, 10, 6),
    "out_of_scope": (8, 10, 6),
}


def _docs() -> dict[str, str]:
    docs: dict[str, str] = {}
    for path in sorted((DATA_DIR / "kb").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
        assert match, f"{path.name}: no YAML frontmatter"
        front = yaml.safe_load(match.group(1)) or {}
        assert front.get("doc_id") == path.stem, (
            f"{path.name}: doc_id {front.get('doc_id')!r} != filename stem"
        )
        docs[path.stem] = text
    assert docs, "data/kb is empty"
    return docs


def _rows() -> list[dict]:
    return load_golden_set(DATA_DIR / "golden_set.jsonl")


@pytest.mark.unit
def test_attribution_file_is_present() -> None:
    """MultiHopRAG is ODC-BY: attribution is a license obligation, not courtesy."""
    attribution = DATA_DIR / "ATTRIBUTION.md"
    assert attribution.is_file(), "data/ATTRIBUTION.md missing (ODC-BY requires attribution)"
    text = attribution.read_text(encoding="utf-8")
    assert "ODC-BY" in text and "MultiHop-RAG" in text


@pytest.mark.unit
def test_golden_set_loads_and_carries_every_required_field() -> None:
    rows = _rows()
    assert len(rows) == sum(t + d + s for t, d, s in EXPECTED_PARTITION.values())
    for row in rows:
        missing = [f for f in REQUIRED_FIELDS if f not in row]
        assert not missing, f"{row.get('example_id')}: missing {missing}"
    ids = [r["example_id"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate example_ids"


@pytest.mark.unit
def test_every_cited_doc_exists() -> None:
    docs = _docs()
    for row in _rows():
        for field in ("expected_doc_ids", "expected_citations"):
            for doc_id in row[field]:
                assert doc_id in docs, f"{row['example_id']}: {field} names unknown doc {doc_id!r}"
        assert set(row["expected_citations"]) <= set(row["expected_doc_ids"]), (
            f"{row['example_id']}: expected_citations not a subset of expected_doc_ids"
        )


@pytest.mark.unit
def test_every_must_include_string_is_present_in_a_cited_doc() -> None:
    """Same rule as the example domains: refusal rows are checked against
    their reference_answer, answer rows against their cited docs."""
    docs = _docs()
    for row in _rows():
        if row["should_refuse"]:
            source = row["reference_answer"] + "\n" + row["query"]
        else:
            source = "\n".join(docs[d] for d in row["expected_doc_ids"] if d in docs)
        for needle in row["must_include"]:
            assert needle in source, (
                f"{row['example_id']}: must_include {needle!r} not present in its sources"
            )


@pytest.mark.unit
def test_no_row_has_an_empty_must_include() -> None:
    for row in _rows():
        assert row["must_include"], (
            f"{row['example_id']}: empty must_include -> empty expected_facts, "
            "and the Correctness judge errors instead of scoring"
        )


@pytest.mark.unit
def test_every_must_not_include_is_a_real_trap() -> None:
    """Each forbidden string on an answer row exists in some non-cited KB doc
    and in none of the cited ones. Refusal rows carry behavioral traps or
    none — they catch fabrication, not retrieval confusion — so the
    KB-residency rule does not apply to them."""
    docs = _docs()
    for row in _rows():
        if row["should_refuse"]:
            continue
        cited = set(row["expected_doc_ids"])
        cited_text = "\n".join(docs[d] for d in cited)
        others = {k: v for k, v in docs.items() if k not in cited}
        assert row["must_not_include"], f"{row['example_id']}: no traps on an answer row"
        for needle in row["must_not_include"]:
            assert needle not in cited_text, (
                f"{row['example_id']}: must_not_include {needle!r} appears in a cited doc"
            )
            assert any(needle in text for text in others.values()), (
                f"{row['example_id']}: must_not_include {needle!r} appears in no "
                "non-cited KB doc — fake trap"
            )


@pytest.mark.unit
def test_refusal_rows_are_exactly_the_out_of_scope_rows() -> None:
    for row in _rows():
        out_of_scope = row["category"] == "out_of_scope"
        assert bool(row["should_refuse"]) == out_of_scope, (
            f"{row['example_id']}: should_refuse={row['should_refuse']} "
            f"but category={row['category']}"
        )
        if out_of_scope:
            assert not row["expected_doc_ids"], f"{row['example_id']}: refusal row cites docs"
            assert not row["expected_citations"], (
                f"{row['example_id']}: refusal row expects citations"
            )


@pytest.mark.unit
def test_the_partition_table_is_exact() -> None:
    """`full` mode IS the dev partition; a drifting partition detonates it."""
    cfg = load_harness(REPO_ROOT / "scaffold").config.eval
    assert cfg.split.enabled, "shipped config must enable the split (#21)"
    train, dev, test = runner.partition_dataset(_rows(), cfg.split)
    counts = {
        part: {b: sum(1 for e in examples if e["category"] == b) for b in EXPECTED_PARTITION}
        for part, examples in (("train", train), ("dev", dev), ("test", test))
    }
    for bucket, want in EXPECTED_PARTITION.items():
        got = (counts["train"][bucket], counts["dev"][bucket], counts["test"][bucket])
        assert got == want, f"partition {bucket}: want {want}, got {got}"


@pytest.mark.unit
def test_every_mode_selects_what_the_config_promises() -> None:
    rows = _rows()
    cfg = load_harness(REPO_ROOT / "scaffold").config.eval
    _, dev, test = runner.partition_dataset(rows, cfg.split)
    for mode in ("quick", "standard", "full"):
        with pytest.warns(UserWarning, match=f"scaled {mode!r} bucket counts"):
            selected = runner._select_mode_examples(rows, cfg=cfg, selected_mode=mode)
        assert len(selected) == cfg.modes[mode].rows, (
            f"{mode}: selected {len(selected)}, config promises {cfg.modes[mode].rows}"
        )
        assert {r["example_id"] for r in selected} <= {r["example_id"] for r in dev}, (
            f"{mode}: selected rows from outside the dev partition"
        )
    test_selected = runner._select_mode_examples(rows, cfg=cfg, selected_mode="test")
    assert list(test_selected) == test, "test mode must return the whole test partition"
    assert cfg.modes["test"].rows == len(test)
