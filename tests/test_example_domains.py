"""Integrity checks for the worked example domains under ``examples/``.

An example that has quietly stopped working is worse than no example: it is the
first thing a reader runs, and its failure reads as the harness being broken.
These checks run offline against every domain in ``examples/``, so a new one is
covered the moment it is added -- no per-domain test to remember to write.

The load-bearing check is :func:`test_every_must_not_include_is_a_real_trap`.
A ``must_not_include`` string that appears nowhere else in the knowledge base is
a trap with nothing to catch: the row passes whatever the agent retrieves, and
the distractor bucket silently stops measuring distractor resistance.

No LLM, no network. Everything here is file parsing plus the harness's own
loaders.
"""

from __future__ import annotations

import collections
import re
from pathlib import Path

import pytest
import yaml

from anvil.data.golden_set import REQUIRED_FIELDS, load_golden_set, select_subset
from anvil.runtime.loader import default_runtime_config_path, load_harness
from anvil.runtime.models import RuntimeYAML

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

# Every directory under examples/ that looks like a domain. Collected at import
# time so each domain becomes its own test id.
EXAMPLE_DOMAINS = sorted(
    p for p in EXAMPLES_DIR.glob("*") if p.is_dir() and (p / "data" / "kb").is_dir()
)

# Fails loudly rather than passing vacuously: an empty examples/ would otherwise
# make every test below a silent no-op.
assert EXAMPLE_DOMAINS, f"no example domains found under {EXAMPLES_DIR}"

_IDS = [p.name for p in EXAMPLE_DOMAINS]
_domain = pytest.mark.parametrize("domain", EXAMPLE_DOMAINS, ids=_IDS)


def _kb_docs(domain: Path) -> dict[str, str]:
    """Map ``doc_id`` -> full file text for every doc in the domain's KB."""
    docs: dict[str, str] = {}
    for path in sorted((domain / "data" / "kb").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
        assert match, f"{path.name}: no YAML frontmatter"
        front = yaml.safe_load(match.group(1)) or {}
        doc_id = front.get("doc_id")
        assert doc_id, f"{path.name}: frontmatter has no doc_id"
        # The search tool indexes by doc_id while golden rows are written against
        # filenames; a mismatch makes a row uncitable in a way nothing else
        # reports.
        assert doc_id == path.stem, f"{path.name}: doc_id {doc_id!r} != filename stem"
        docs[doc_id] = text
    assert docs, f"{domain.name}: knowledge base is empty"
    return docs


def _rows(domain: Path) -> list[dict]:
    return load_golden_set(domain / "data" / "golden_set.jsonl")


@_domain
@pytest.mark.unit
def test_golden_set_loads_and_carries_every_required_field(domain: Path) -> None:
    rows = _rows(domain)
    assert rows, f"{domain.name}: golden set is empty"
    for row in rows:
        missing = [f for f in REQUIRED_FIELDS if f not in row]
        assert not missing, f"{row.get('example_id')}: missing {missing}"
    ids = [r["example_id"] for r in rows]
    assert len(set(ids)) == len(ids), f"{domain.name}: duplicate example_ids"


@_domain
@pytest.mark.unit
def test_bucket_counts_satisfy_every_configured_mode(domain: Path) -> None:
    """The domain must have enough rows per bucket for each mode it declares.

    ``select_subset`` takes the first N of each bucket, so a domain one row
    short in one bucket does not error -- it silently returns a smaller set and
    the mode quietly evaluates fewer rows than its config says.
    """
    rows = _rows(domain)
    config_path = default_runtime_config_path(domain / "scaffold")
    config = RuntimeYAML.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")))

    available = collections.Counter(r["category"] for r in rows)
    for mode_name, mode in config.eval.modes.items():
        for bucket, wanted in mode.buckets.items():
            assert available[bucket] >= wanted, (
                f"{domain.name}: mode {mode_name} wants {wanted} {bucket} rows, "
                f"golden set has {available[bucket]}"
            )
        selected = select_subset(rows, buckets=mode.buckets)
        assert len(selected) == mode.rows, (
            f"{domain.name}: mode {mode_name} declares rows={mode.rows} but its "
            f"buckets select {len(selected)}"
        )


@_domain
@pytest.mark.unit
def test_every_cited_doc_exists(domain: Path) -> None:
    docs = _kb_docs(domain)
    for row in _rows(domain):
        for field in ("expected_doc_ids", "expected_citations"):
            for doc_id in row[field]:
                assert doc_id in docs, f"{row['example_id']}: {field} names unknown doc {doc_id!r}"


@_domain
@pytest.mark.unit
def test_every_must_include_string_is_present_in_a_cited_doc(domain: Path) -> None:
    """A row cannot demand a string its own sources do not contain.

    Otherwise the row is unanswerable: no retrieval and no amount of scaffold
    improvement can satisfy it, and it drags the bucket's score down forever.

    Refusal rows are checked against their ``reference_answer`` instead, because
    they cite no documents by definition -- for them the expected facts describe
    the shape of the refusal, not the content of a source.
    """
    docs = _kb_docs(domain)
    for row in _rows(domain):
        if row["should_refuse"]:
            source, where = row["reference_answer"], "its reference_answer"
        else:
            source = "\n".join(docs[d] for d in row["expected_doc_ids"] if d in docs)
            where = f"any of its cited docs {row['expected_doc_ids']}"
        for needle in row["must_include"]:
            assert needle in source, (
                f"{row['example_id']}: must_include {needle!r} does not appear in {where}"
            )


@_domain
@pytest.mark.unit
def test_no_row_has_an_empty_must_include(domain: Path) -> None:
    """Every row needs expected facts, refusal rows included.

    Not a style rule. ``_build_dataset`` projects ``must_include`` onto
    mlflow's ``expected_facts``, and the Correctness judge requires either
    ``expected_facts`` or ``expected_response``; given an empty list it raises
    "Missing input fields" rather than scoring. Four refusal rows shipped with
    ``must_include: []``, and the first live run of this example errored
    correctness on 2 of 8 cases and exited 2 as unjudgeable -- the harness
    caught it, but only after paying for the run.
    """
    for row in _rows(domain):
        assert row["must_include"], (
            f"{row['example_id']}: must_include is empty, so expected_facts is "
            "empty and the Correctness judge will error instead of scoring"
        )


@_domain
@pytest.mark.unit
def test_every_must_not_include_is_a_real_trap(domain: Path) -> None:
    """Each forbidden string must exist somewhere else in the KB.

    That is what makes it a trap: some other document really does contain the
    plausible-but-wrong value, so the row measures whether the agent picked the
    right source. A forbidden string that appears nowhere cannot be emitted by a
    grounded agent, so the row passes unconditionally.
    """
    docs = _kb_docs(domain)
    for row in _rows(domain):
        others = {k: v for k, v in docs.items() if k not in row["expected_doc_ids"]}
        for needle in row["must_not_include"]:
            assert any(needle in text for text in others.values()), (
                f"{row['example_id']}: must_not_include {needle!r} appears in no "
                "other KB doc, so nothing can trigger it -- fake trap"
            )


@_domain
@pytest.mark.unit
def test_refusal_rows_are_exactly_the_out_of_scope_rows(domain: Path) -> None:
    for row in _rows(domain):
        out_of_scope = row["category"] == "out_of_scope"
        assert bool(row["should_refuse"]) == out_of_scope, (
            f"{row['example_id']}: should_refuse={row['should_refuse']} "
            f"but category={row['category']}"
        )
        if out_of_scope:
            # A row that should be refused cannot also expect citations: the
            # groundedness scorer does not apply to it (docs/decisions.md D10),
            # so expectations here would never be checked.
            assert not row["expected_doc_ids"], f"{row['example_id']}: refusal row cites docs"
            assert not row["expected_citations"], f"{row['example_id']}: refusal row expects cites"


@_domain
@pytest.mark.unit
def test_the_scaffold_and_its_config_load(domain: Path) -> None:
    """The example must compose a runtime prompt, not merely parse.

    This is the check that would have caught the example shipping without its
    own ``harness/config.yaml``: the loader resolves that file as a sibling of
    ``scaffold/``, so its absence makes the documented command fail outright.
    """
    scaffold = domain / "scaffold"
    config_path = default_runtime_config_path(scaffold)
    assert config_path.is_file(), (
        f"{domain.name}: expected a runtime config at {config_path} "
        "(the loader resolves it as a sibling of scaffold/)"
    )
    snapshot = load_harness(scaffold, config_path)
    assert snapshot.config.skills, f"{domain.name}: no skills configured"
    assert snapshot.config.tools, f"{domain.name}: no tools configured"


@_domain
@pytest.mark.unit
def test_the_example_domain_does_not_inherit_the_shipped_judge_domain(domain: Path) -> None:
    """An example must tell the refusal judge which domain it is grading.

    Without ``judge_domain_context`` the judge falls back to the built-in
    NeoVolt description, and every correct refusal in another domain is graded
    against the wrong notion of "in scope". The failure is invisible in the
    aggregate -- it just makes the refusal score wrong.
    """
    from anvil.eval.scorers import DEFAULT_JUDGE_DOMAIN_CONTEXT, DEFAULT_JUDGE_DOMAIN_NAME

    config_path = default_runtime_config_path(domain / "scaffold")
    config = RuntimeYAML.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")))

    assert config.judge_domain_name, f"{domain.name}: judge_domain_name is not set"
    assert config.judge_domain_context, f"{domain.name}: judge_domain_context is not set"
    assert config.judge_domain_name != DEFAULT_JUDGE_DOMAIN_NAME
    assert config.judge_domain_context != DEFAULT_JUDGE_DOMAIN_CONTEXT


@_domain
@pytest.mark.unit
def test_the_documented_run_command_matches_the_real_flags(domain: Path) -> None:
    """The README's command must parse against the actual CLI.

    A README is the one file guaranteed to be read and the one nothing else
    verifies, so the flags it names are checked against the live argparse
    definition rather than trusted.
    """
    import importlib.util

    readme = (domain / "README.md").read_text(encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_ev", REPO_ROOT / "scripts" / "evaluate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    known = {
        action.option_strings[0]
        for action in module._arg_parser()._actions
        if action.option_strings
    }

    for flag in sorted(set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]+", readme))):
        assert flag in known, f"{domain.name}/README.md documents unknown flag {flag}"
