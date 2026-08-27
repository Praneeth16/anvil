"""Chunk-level retrieval for ``search_knowledge_base`` (issue #26).

Whole-document BM25 used to return ``body[:3000]`` — a prefix of articles
whose answers routinely sit mid-body (measured live during the MultiHopRAG
migration: correctness failures with groundedness 1.0). Documents now index
as deterministic <=1500-char paragraph-packed chunks, BM25 ranks chunks,
and a hit's snippet is the document's best chunks within a 3000-char budget
— the old prefix's volume, spent where the query points. Measured over the
golden set's 159 must-include facts, visibility from the expected doc's
snippet is 71.7%, vs 66.7% for the prefix and 54.1% for a single best
chunk (the rejected alternative — prefix volume was doing real work).

The contract stays doc-level: ``k`` counts documents, snippets cite their
parent ``doc_id``, and the trace keeps the ``doc_uri``/``page_content``
keys the groundedness judge reads.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from anvil.tools.search_knowledge_base import (
    CHUNK_CHAR_LIMIT,
    SNIPPET_CHAR_BUDGET,
    _assemble_snippet,
    _bm25_scores,
    _chunk_body,
    _IndexedChunk,
    _load_kb_index,
    _search,
    format_hits,
    make_kb_executor,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_doc(kb: Path, name: str, body: str, *, title: str | None = None) -> None:
    kb.mkdir(parents=True, exist_ok=True)
    (kb / f"{name}.md").write_text(
        f"---\ndoc_id: {name}\ntitle: {title or name}\n---\n\n{body}",
        encoding="utf-8",
    )


def _para(words: str, n: int) -> str:
    """A paragraph of roughly ``n`` chars of repeatable, tokenizable text."""
    return (words + " ") * (n // (len(words) + 1) + 1)


# ---------------------------------------------------------------------------
# _chunk_body
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_chunking_packs_paragraphs_up_to_the_limit() -> None:
    paras = [f"para{i} " + "word " * 100 for i in range(6)]  # ~505 chars each
    body = "\n\n".join(paras)
    chunks = _chunk_body(body, limit=1200)
    # 505 fits twice in 1200 but not three times -> pairs.
    assert len(chunks) == 3
    for chunk in chunks:
        assert len(chunk) <= 1200
    # Paragraphs stay whole and in order.
    for i, para in enumerate(paras):
        assert para.strip() in chunks[i // 2]


@pytest.mark.unit
def test_chunking_hard_splits_an_oversized_paragraph_on_whitespace() -> None:
    body = "word " * 600  # one ~3000-char paragraph, no blank lines
    chunks = _chunk_body(body, limit=1000)
    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk) <= 1000
        assert chunk == chunk.strip()
    # Whitespace splits: no chunk ends or starts mid-word where avoidable.
    assert all(" " in chunk for chunk in chunks)


@pytest.mark.unit
def test_chunking_splits_mid_word_only_when_no_whitespace_is_in_reach() -> None:
    chunks = _chunk_body("x" * 2500, limit=1000)
    assert [len(c) for c in chunks] == [1000, 1000, 500]


@pytest.mark.unit
def test_chunking_is_deterministic_and_preserves_the_text() -> None:
    body = "\n\n".join(_para(f"sentence number {i} with some words", 700) for i in range(8))
    first = _chunk_body(body)
    assert first == _chunk_body(body)
    assert all(first)  # no empty chunks
    # Everything the paragraphs carried is in some chunk (separators may
    # normalize, words may not vanish).
    joined = "\n\n".join(first)
    for i in range(8):
        assert f"sentence number {i}" in joined


# ---------------------------------------------------------------------------
# The index and the search
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_short_docs_index_as_one_chunk_and_search_returns_the_whole_body(
    tmp_path: Path,
) -> None:
    """NeoVolt/pyloom-shaped corpora (docs shorter than one chunk) see no
    behavior change beyond whitespace normalization."""
    kb = tmp_path / "kb"
    _write_doc(kb, "short", "Short policy body.\n\nSecond paragraph.")
    docs = _load_kb_index(kb)
    assert len(docs[0].chunks) == 1
    hits = _search(docs, "short policy", 3)
    assert len(hits) == 1
    assert hits[0].snippet == "Short policy body.\n\nSecond paragraph."
    assert hits[0].chunk_index == 0
    assert format_hits(hits) == (
        "=== doc_id: short ===\ntitle: short\n\nShort policy body.\n\nSecond paragraph."
    )


@pytest.mark.unit
def test_a_fact_deep_in_the_body_is_retrievable(tmp_path: Path) -> None:
    """The mhr_d_006 failure as a unit test: the answer term sits past the
    old 3000-char prefix, so the prefix design could never surface it."""
    kb = tmp_path / "kb"
    filler = _para("ordinary reporting about the match and the season", 900)
    body = "\n\n".join([filler] * 4 + ["The availability report came from Gaston Edul."])
    assert "Gaston Edul" not in body[:3000]  # the prefix hides it, by construction
    _write_doc(kb, "deep", body)
    docs = _load_kb_index(kb)

    hits = _search(docs, "Gaston Edul availability report", 3)
    assert hits and hits[0].doc_id == "deep"
    assert "Gaston Edul" in hits[0].snippet


@pytest.mark.unit
def test_k_counts_documents_not_chunks(tmp_path: Path) -> None:
    """One doc owning the two best chunks must not starve the second doc."""
    kb = tmp_path / "kb"
    _write_doc(
        kb,
        "aaa",
        "\n\n".join(
            [
                "zebra zebra zebra " + _para("filler words here", 900),
                _para("more filler", 900),
                "zebra zebra " + _para("further filler", 900),
            ]
        ),
    )
    _write_doc(kb, "bbb", "zebra " + _para("other filler", 400))
    hits = _search(_load_kb_index(kb), "zebra", 2)
    assert {h.doc_id for h in hits} == {"aaa", "bbb"}
    # aaa ranks first (its best chunk mentions zebra more), bbb is not starved.
    assert hits[0].doc_id == "aaa"
    assert hits[1].doc_id == "bbb"


@pytest.mark.unit
def test_a_strong_mid_body_match_outranks_a_weak_prefix_match(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    _write_doc(kb, "prefixy", "quokka " + _para("plain words", 2000))
    _write_doc(
        kb,
        "deep",
        "\n\n".join(
            [_para("plain words", 2000), "quokka quokka quokka quokka quokka " + _para("tail", 300)]
        ),
    )
    hits = _search(_load_kb_index(kb), "quokka", 2)
    assert hits[0].doc_id == "deep"


# ---------------------------------------------------------------------------
# Snippet assembly
# ---------------------------------------------------------------------------


def _chunk(index: int, text: str) -> _IndexedChunk:
    return _IndexedChunk(
        text=text, chunk_index=index, token_counts=Counter(text.split()), length=len(text)
    )


@pytest.mark.unit
def test_assemble_snippet_selects_by_score_but_presents_in_document_order() -> None:
    chunks = [
        (0.9, _chunk(5, "e" * 1400)),
        (0.8, _chunk(2, "b" * 1400)),
        (0.7, _chunk(3, "c" * 1400)),  # would bust the 3000 budget
    ]
    snippet = _assemble_snippet(chunks)
    assert len(snippet) <= SNIPPET_CHAR_BUDGET
    # Picked the top two by score (5 and 2), presented in document order (2, 5),
    # with an elision marker for the gap (2 -> 5 is not contiguous).
    assert snippet.index("b" * 1400) < snippet.index("e" * 1400)
    assert "[...]" in snippet
    assert "c" * 1400 not in snippet


@pytest.mark.unit
def test_assemble_snippet_joins_contiguous_chunks_without_a_marker() -> None:
    snippet = _assemble_snippet([(0.9, _chunk(2, "b" * 100)), (0.8, _chunk(3, "c" * 100))])
    assert snippet == "b" * 100 + "\n\n" + "c" * 100


@pytest.mark.unit
def test_assemble_snippet_always_includes_the_best_chunk() -> None:
    big = _chunk(0, "x" * (SNIPPET_CHAR_BUDGET + 500))  # pathological, over budget alone
    snippet = _assemble_snippet([(0.9, big)])
    assert snippet == big.text


# ---------------------------------------------------------------------------
# The executor and the trace contract
# ---------------------------------------------------------------------------


@pytest.fixture
def local_mlruns(tmp_path: Path):
    """Point mlflow at an isolated local file store for the test."""
    import mlflow

    store = tmp_path / "mlruns"
    old_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"file://{store}")
    yield store
    mlflow.set_tracking_uri(old_uri)


def test_executor_output_and_retriever_span_carry_the_chunk_contract(
    tmp_path: Path, local_mlruns: Path
) -> None:
    """The groundedness judge reads ``page_content`` + ``metadata.doc_uri``
    from the RETRIEVER span; chunk-level retrieval must keep both keys, with
    the chunk text (not a prefix) as the content."""
    import mlflow
    from mlflow.entities import SpanType

    kb = tmp_path / "kb"
    _write_doc(
        kb,
        "deep",
        "\n\n".join(
            [_para("ordinary words", 2000), "The wombat fact lives here. " + _para("tail", 300)]
        ),
    )
    executor = make_kb_executor(kb)

    mlflow.set_experiment("test_chunk_trace_contract")
    with mlflow.start_span(name="root", span_type=SpanType.CHAIN):
        formatted = executor("search_knowledge_base", json.dumps({"query": "wombat", "k": 1}))

    assert "=== doc_id: deep ===" in formatted
    assert "The wombat fact lives here." in formatted

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    retriever = next(s for s in trace.data.spans if s.span_type == "RETRIEVER")
    (chunk,) = retriever.outputs
    assert chunk["metadata"]["doc_uri"] == "deep"
    assert chunk["metadata"]["chunk_index"] == 1
    assert "The wombat fact lives here." in chunk["page_content"]


# ---------------------------------------------------------------------------
# Whole-KB smoke over the real corpus
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_real_kb_chunks_are_in_budget_substrings_of_their_parent() -> None:
    docs = _load_kb_index(REPO_ROOT / "data" / "kb")
    assert all(
        len(c.text) <= CHUNK_CHAR_LIMIT for d in docs for c in d.chunks
    ), "a chunk exceeded CHUNK_CHAR_LIMIT"
    queries = [
        "When did the Diamondbacks clinch a playoff berth?",
        "What was reported about Lionel Messi's availability for Inter Miami?",
        "Who won the Nobel Prize and for what discovery?",
    ]
    bodies = {d.doc_id: "\n\n".join(c.text for c in d.chunks) for d in docs}
    seen_snippet = False
    for query in queries:
        for hit in _search(docs, query, 3):
            seen_snippet = True
            assert len(hit.snippet) <= SNIPPET_CHAR_BUDGET
            # Every passage is real text from the parent doc — retrieval
            # selects, it never fabricates.
            for passage in hit.snippet.split("\n\n"):
                if passage == "[...]":
                    continue
                assert passage in bodies[hit.doc_id]
    assert seen_snippet


@pytest.mark.unit
def test_real_kb_index_is_deterministic_across_loads() -> None:
    first = _load_kb_index(REPO_ROOT / "data" / "kb")
    second = _load_kb_index(REPO_ROOT / "data" / "kb")
    for d1, d2 in zip(first, second, strict=True):
        assert d1.doc_id == d2.doc_id
        assert [c.text for c in d1.chunks] == [c.text for c in d2.chunks]


@pytest.mark.unit
def test_bm25_ranks_chunks_not_docs() -> None:
    """Sanity on the scoring seam: a chunk with the term beats one without."""
    from anvil.tools.search_knowledge_base import _tokenise

    def scorable(text: str, index: int) -> _IndexedChunk:
        tokens = _tokenise(text)
        return _IndexedChunk(text, index, Counter(tokens), len(tokens))

    scores = _bm25_scores(
        _tokenise("quokka"),
        [scorable("quokka quokka " + "filler " * 50, 0), scorable("filler " * 52, 1)],
    )
    assert scores[0] > 0
    assert scores[1] == 0
