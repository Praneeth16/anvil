"""``search_knowledge_base`` tool: chunk-level BM25 retriever over ``data/kb/``.

Schema (locked, must stay identical when a UC Function + Vector
Search backend swaps in for Phase 4)::

    search_knowledge_base(query: str, k: int = 3)
        -> list[{doc_id, title, snippet}]

Retrieval is **chunk-level**: documents are split into deterministic
<=1500-char chunks at index time, BM25 ranks chunks, and each hit's
``snippet`` is the document's best-scoring chunk. ``k`` counts DOCUMENTS —
doc score is its best chunk's score — so one article cannot monopolize the
result set and starve the second source a multi-hop question needs. The
doc-level shape is what keeps golden rows (``expected_doc_ids``), the
citation contract, and the groundedness judge's trace extraction
(``metadata.doc_uri``) unchanged.

Backend = ``"bm25"`` for the demo: zero dependencies, fully offline,
deterministic. ``"vector_search"`` raises ``NotImplementedError`` and
is the documented Phase-4 plug-in point.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import mlflow
import yaml
from mlflow.entities import SpanType

from anvil.runtime.agent import ToolExecutor

SEARCH_TOOL_NAME = "search_knowledge_base"
DEFAULT_K = 3
# Whole-document ranking used to return body[:SNIPPET_CHAR_LIMIT] — 500, then
# 3000 chars — a prefix of an article whose answer routinely sits mid-body
# (every MultiHopRAG doc is 5.2k-61k chars, median 8.4k; measured live: rows
# failed correctness with groundedness 1.0 because the fact was past the
# prefix). Retrieval is chunk-level now: docs split into <=1500-char
# paragraph-packed chunks at index time, BM25 ranks chunks, and a hit's
# snippet is the document's best chunks, accumulated in score order up to
# SNIPPET_CHAR_BUDGET and presented in document order. The budget is the old
# prefix's size, spent where the query points instead of where the article
# starts: measured over the golden set's 159 must-include facts, visibility
# from the expected doc's snippet went 66.7% (3000 prefix) -> 71.7% (this),
# while a single best chunk — the other candidate design — DROPS to 54.1%
# because prefix volume was doing real work. NeoVolt and pyloom docs are
# shorter than one chunk, so they index as exactly one chunk per doc and
# their results change only in that the snippet is whitespace-normalized.
CHUNK_CHAR_LIMIT = 1500
# Per-document snippet cap: the most context a hit may occupy, filled with
# the document's best chunks rather than its prefix. Same number as the old
# prefix limit, so the context cost of a k=3 search is unchanged.
SNIPPET_CHAR_BUDGET = 3000
EMPTY_RESULT_TEXT = "No matching policy documents."

_BM25_K1 = 1.5
_BM25_B = 0.75
_MIN_TOKEN_LEN = 3
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_FRONTMATTER_DELIM = "---"

Backend = Literal["bm25", "vector_search"]


@dataclass(frozen=True)
class KbHit:
    doc_id: str
    title: str
    snippet: str
    score: float
    chunk_index: int


@dataclass(frozen=True)
class _IndexedChunk:
    """One passage of a document, ranked as a unit by BM25."""

    text: str
    chunk_index: int
    token_counts: Counter[str]
    length: int


@dataclass(frozen=True)
class _IndexedDoc:
    doc_id: str
    title: str
    chunks: list[_IndexedChunk]


class _Scorable(Protocol):
    """What ``_bm25_scores`` needs of a ranked unit (a chunk).

    Property declarations, not attributes: the ranked units are frozen
    dataclasses, and a frozen attribute is read-only, which a settable
    protocol member would reject.
    """

    @property
    def token_counts(self) -> Counter[str]: ...

    @property
    def length(self) -> int: ...


def _strip_frontmatter(text: str) -> tuple[dict, str]:
    """Split a markdown file's optional YAML frontmatter from its body."""
    if not text.startswith(_FRONTMATTER_DELIM):
        return {}, text
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip() != _FRONTMATTER_DELIM:
        return {}, text
    for i in range(1, len(lines)):
        if lines[i].rstrip() == _FRONTMATTER_DELIM:
            fm_text = "".join(lines[1:i])
            body = "".join(lines[i + 1 :]).lstrip("\n")
            fm = yaml.safe_load(fm_text) or {}
            if not isinstance(fm, dict):
                fm = {}
            return fm, body
    return {}, text


def _tokenise(text: str) -> list[str]:
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if len(tok) >= _MIN_TOKEN_LEN]


def _hard_split(text: str, limit: int) -> list[str]:
    """Split an over-``limit`` paragraph at the last whitespace before the limit.

    Falls back to a mid-word cut when no whitespace is in reach — the
    alternative is an unbounded loop or an over-limit chunk, and a rare ugly
    cut beats both.
    """
    pieces: list[str] = []
    rest = text
    while len(rest) > limit:
        cut = rest.rfind(" ", 0, limit + 1)
        if cut <= 0:
            cut = limit
        pieces.append(rest[:cut])
        rest = rest[cut:].lstrip()
    if rest:
        pieces.append(rest)
    return pieces


def _chunk_body(body: str, limit: int = CHUNK_CHAR_LIMIT) -> list[str]:
    """Split ``body`` into deterministic <=``limit``-char chunks.

    Paragraphs (blank-line separated) are packed greedily; a paragraph that
    exceeds the limit on its own is hard-split (see :func:`_hard_split`). No
    overlap: MultiHopRAG evidence is sentence-scale, and every chunk boundary
    is a pure function of the file bytes — the tool re-indexes per process,
    so anything less deterministic would make a round's retrieval depend on
    process history.
    """
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        for piece in _hard_split(para, limit):
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) <= limit:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return chunks


def _load_kb_index(kb_dir: Path) -> list[_IndexedDoc]:
    if not kb_dir.is_dir():
        raise FileNotFoundError(f"KB directory not found: {kb_dir}")

    docs: list[_IndexedDoc] = []
    for path in sorted(kb_dir.glob("*.md")):
        fm, body = _strip_frontmatter(path.read_text(encoding="utf-8"))
        # Bind before narrowing: two separate ``.get`` calls give the type
        # checker nothing to narrow, so ``doc_id``/``title`` were annotated
        # ``str`` while a KB doc missing the frontmatter field could still put
        # ``None`` through.
        raw_doc_id = fm.get("doc_id")
        doc_id = raw_doc_id if isinstance(raw_doc_id, str) else path.stem
        raw_title = fm.get("title")
        title = raw_title if isinstance(raw_title, str) else doc_id
        chunks: list[_IndexedChunk] = []
        for i, text in enumerate(_chunk_body(body.strip())):
            # The title ranks with every chunk, exactly as it ranked with the
            # whole document before chunking — title terms boost all of a
            # doc's chunks equally, so the weighting's effect is unchanged.
            tokens = _tokenise(f"{title}\n{text}")
            chunks.append(
                _IndexedChunk(
                    text=text,
                    chunk_index=i,
                    token_counts=Counter(tokens),
                    length=len(tokens),
                )
            )
        if not any(c.length for c in chunks):
            continue
        docs.append(_IndexedDoc(doc_id=doc_id, title=title, chunks=chunks))
    if not docs:
        raise ValueError(f"KB directory has no usable markdown docs: {kb_dir}")
    return docs


def _bm25_scores(query_tokens: list[str], items: Sequence[_Scorable]) -> list[float]:
    n_docs = len(items)
    avgdl = sum(d.length for d in items) / n_docs

    df: dict[str, int] = {}
    for term in set(query_tokens):
        df[term] = sum(1 for d in items if term in d.token_counts)

    scores: list[float] = []
    for d in items:
        score = 0.0
        norm = 1 - _BM25_B + _BM25_B * d.length / avgdl
        for term in query_tokens:
            tf = d.token_counts.get(term, 0)
            if tf == 0:
                continue
            idf = math.log(((n_docs - df[term] + 0.5) / (df[term] + 0.5)) + 1.0)
            score += idf * (tf * (_BM25_K1 + 1)) / (tf + _BM25_K1 * norm)
        scores.append(score)
    return scores


def _assemble_snippet(scored_chunks: list[tuple[float, _IndexedChunk]]) -> str:
    """The document's best chunks within ``SNIPPET_CHAR_BUDGET``, in document order.

    ``scored_chunks`` is the doc's positive-scoring chunks, best first. The
    best chunk is always included; each further chunk joins only while the
    budget holds, separators included, so the result never exceeds
    ``SNIPPET_CHAR_BUDGET``. Selection is by score but presentation follows
    the article (``chunk_index`` order), so the model reads passages as the
    author wrote them; non-contiguous chunks are joined with an elision
    marker.
    """
    picked: list[_IndexedChunk] = []
    spent = 0
    for _, chunk in scored_chunks:
        # Budget the separator too (worst case is the gap marker), so the
        # assembled snippet provably stays within SNIPPET_CHAR_BUDGET.
        cost = len(chunk.text) if not picked else len(chunk.text) + len("\n\n[...]\n\n")
        if picked and spent + cost > SNIPPET_CHAR_BUDGET:
            break
        picked.append(chunk)
        spent += cost
    picked.sort(key=lambda c: c.chunk_index)
    parts: list[str] = []
    for i, chunk in enumerate(picked):
        if i > 0 and chunk.chunk_index != picked[i - 1].chunk_index + 1:
            parts.append("[...]")
        parts.append(chunk.text)
    return "\n\n".join(parts)


def _search(docs: list[_IndexedDoc], query: str, k: int) -> list[KbHit]:
    """Rank documents by their best chunk; return the top ``k`` documents.

    Chunk-level scoring with doc-level results: the snippet holds the chunks
    the answer is likeliest in (see :func:`_assemble_snippet`), but ``k``
    still counts documents, so a single article cannot fill the result set
    and starve the second source a multi-hop question needs.
    """
    query_tokens = _tokenise(query)
    if not query_tokens:
        return []
    flat = [(chunk, doc) for doc in docs for chunk in doc.chunks]
    scores = _bm25_scores(query_tokens, [chunk for chunk, _ in flat])
    per_doc: dict[str, list[tuple[float, _IndexedChunk]]] = {}
    doc_by_id: dict[str, _IndexedDoc] = {}
    for score, (chunk, doc) in zip(scores, flat, strict=True):
        if score <= 0:
            continue
        per_doc.setdefault(doc.doc_id, []).append((score, chunk))
        doc_by_id[doc.doc_id] = doc
    for hits in per_doc.values():
        hits.sort(key=lambda entry: entry[0], reverse=True)
    ranked = sorted(
        per_doc.items(),
        key=lambda item: item[1][0][0],  # doc score = its best chunk's score
        reverse=True,
    )
    return [
        KbHit(
            doc_id=doc_by_id[doc_id].doc_id,
            title=doc_by_id[doc_id].title,
            snippet=_assemble_snippet(scored_chunks),
            score=scored_chunks[0][0],
            chunk_index=scored_chunks[0][1].chunk_index,
        )
        for doc_id, scored_chunks in ranked[:k]
    ]


def format_hits(hits: list[KbHit]) -> str:
    if not hits:
        return EMPTY_RESULT_TEXT
    blocks: list[str] = []
    for h in hits:
        blocks.append(f"=== doc_id: {h.doc_id} ===\ntitle: {h.title}\n\n{h.snippet}")
    return "\n\n".join(blocks)


class _KbToolExecutor:
    """Callable that dispatches ``search_knowledge_base`` calls."""

    def __init__(self, docs: list[_IndexedDoc]) -> None:
        self._docs = docs

    def __call__(self, name: str, arguments_json: str) -> str:
        if name != SEARCH_TOOL_NAME:
            raise RuntimeError(
                f"KbToolExecutor cannot dispatch tool {name!r}. "
                f"This executor only handles {SEARCH_TOOL_NAME!r}."
            )
        args = json.loads(arguments_json) if arguments_json else {}
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                f"{SEARCH_TOOL_NAME}: 'query' is required and must be a non-empty string"
            )
        k_raw = args.get("k", DEFAULT_K)
        try:
            k = int(k_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{SEARCH_TOOL_NAME}: 'k' must be an integer (got {k_raw!r})"
            ) from exc
        if k <= 0:
            raise ValueError(f"{SEARCH_TOOL_NAME}: 'k' must be > 0 (got {k})")
        # Emit a RETRIEVER span so MLflow's RetrievalGroundedness
        # scorer (and any future trace-level citation extractor) can
        # read the retrieved chunks. ``mlflow.start_span`` is a no-op
        # outside an active trace in MLflow 3.10.
        # NOTE: ``extract_retrieval_context_from_trace`` reads
        # ``chunk["metadata"]["doc_uri"]`` (NOT ``doc_id``) for source
        # attribution — keep the key as ``doc_uri``.
        with mlflow.start_span(name=SEARCH_TOOL_NAME, span_type=SpanType.RETRIEVER) as span:
            span.set_inputs({"query": query, "k": k})
            hits = _search(self._docs, query, k)
            span.set_outputs(
                [
                    {
                        "page_content": h.snippet,
                        "metadata": {
                            "doc_uri": h.doc_id,
                            "title": h.title,
                            "score": h.score,
                            "chunk_index": h.chunk_index,
                        },
                    }
                    for h in hits
                ]
            )
        return format_hits(hits)


def make_kb_executor(kb_dir: Path | str, backend: Backend = "bm25") -> ToolExecutor:
    """Build a ToolExecutor that dispatches ``search_knowledge_base`` calls."""
    if backend == "vector_search":
        raise NotImplementedError(
            "vector_search backend lands in Phase 4. "
            "Use backend='bm25' for now — schema is identical, swap is later transparent."
        )
    if backend != "bm25":
        raise ValueError(f"Unknown backend: {backend!r}")
    docs = _load_kb_index(Path(kb_dir))
    return _KbToolExecutor(docs)
