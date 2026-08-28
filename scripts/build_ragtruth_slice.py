"""Vendor a stratified RAGTruth slice for judge validation (issue #16).

RAGTruth (ParticleMedia/RAGTruth, MIT) is the only public dataset found
carrying both span-level hallucination labels and an ``incorrect_refusal``
quality flag — the two human references the groundedness and refusal judges
can be validated against. This script fetches it at a pinned commit, joins
responses to their sources, samples a stratified slice, and writes it to
``eval/judge_validation/ragtruth/``.

The slice is NOT a domain: no scaffold, no KB, no golden-set shape. It is
judge-validation data — rows carry the responding model's prompt, response,
reference passage, and the human labels, nothing fabricated.

Verified against the pinned files on 2026-08-28:

* ``response.jsonl`` — 17,790 rows: id, source_id, model, temperature,
  labels, split, quality, response. ``quality`` is a plain string:
  ``good`` (17,617) / ``incorrect_refusal`` (144) / ``truncated`` (29).
  Span labels total exactly 14,289 (the paper's count), so an empty
  ``labels`` list means the response was annotated and supported.
* ``source_info.jsonl`` — 2,965 rows: source_id, task_type
  (QA 989 / Data2txt 1,033 / Summary 943), source (dataset name),
  source_info (a ``{question, passages}`` dict for QA rows, the reference
  text as a string otherwise), prompt (what the model saw).

Usage::

    .venv/bin/python scripts/build_ragtruth_slice.py            # fetch + build
    .venv/bin/python scripts/build_ragtruth_slice.py --check    # byte-diff vs committed
    .venv/bin/python scripts/build_ragtruth_slice.py --input-dir build/ragtruth_raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

RAGTRUTH_REVISION = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
RAW_BASE = (
    "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/"
    f"{RAGTRUTH_REVISION}/dataset"
)
RAW_FILES = ("response.jsonl", "source_info.jsonl")

OUT_DIR = REPO_ROOT / "eval" / "judge_validation" / "ragtruth"

# 50 per (task_type, supported) cell = 300 answer rows, plus every
# incorrect-refusal row — the only inappropriate-refusal examples that
# exist in any public dataset, so none are left on the table.
ANSWER_CELL_QUOTA = 50
TASK_TYPES = ("QA", "Summary", "Data2txt")
EXPECTED_LABEL_SPAN_TOTAL = 14289
EXPECTED_QUALITIES = {"good", "incorrect_refusal", "truncated"}

_ID_PREFIX = {"QA": "rt_qa", "Summary": "rt_sum", "Data2txt": "rt_d2t"}

ATTRIBUTION = """\
# RAGTruth

The rows in this directory are a stratified slice of **RAGTruth**
(`ParticleMedia/RAGTruth` on GitHub), pinned at revision
`c103204b9ce28d6bbad859304bf30de72b8ed8fe`.

- **Paper:** Niu et al., "RAGTruth: A Hallucination Corpus for Developing
  Trustworthy Retrieval-Augmented Language Models" (ACL 2024).
- **Source:** https://github.com/ParticleMedia/RAGTruth
- **License:** MIT, https://github.com/ParticleMedia/RAGTruth/blob/main/LICENSE

## Derivation

Sampled by `scripts/build_ragtruth_slice.py` (seed-pinned, reproducible via
`--check`): 50 rows per (task_type, supported) cell = 300 answer rows,
plus every `incorrect_refusal` row. `truncated` rows are excluded. The
slice validates the harness's judges against human labels; it is not an
eval domain and carries no fabricated expectations.
"""


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 — pinned public URL
        dest.write_bytes(resp.read())


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _require_keys(row: dict, keys: tuple[str, ...], *, where: str) -> None:
    missing = [k for k in keys if k not in row]
    if missing:
        raise RuntimeError(
            f"RAGTruth schema changed: {where} row is missing {missing}; "
            f"actual keys are {sorted(row)}. Re-verify the mapping before "
            "trusting any number this script produces."
        )


def build_slice(rows_dir: Path, *, seed: int, out_dir: Path) -> dict[str, int]:
    responses = _load_jsonl(rows_dir / "response.jsonl")
    sources = _load_jsonl(rows_dir / "source_info.jsonl")
    if not responses or not sources:
        raise RuntimeError(f"empty input under {rows_dir}")

    _require_keys(
        responses[0],
        ("id", "source_id", "model", "labels", "split", "quality", "response"),
        where="response",
    )
    _require_keys(
        sources[0],
        ("source_id", "task_type", "source", "source_info", "prompt"),
        where="source_info",
    )
    for row in responses:
        for label in row["labels"]:
            _require_keys(label, ("start", "end", "text", "label_type"), where="label")

    qualities = Counter(row["quality"] for row in responses)
    unknown = set(qualities) - EXPECTED_QUALITIES
    if unknown:
        raise RuntimeError(
            f"RAGTruth quality values changed: unexpected {sorted(unknown)}; "
            f"expected a subset of {sorted(EXPECTED_QUALITIES)}"
        )
    span_total = sum(len(row["labels"]) for row in responses)
    if span_total != EXPECTED_LABEL_SPAN_TOTAL:
        raise RuntimeError(
            f"expected {EXPECTED_LABEL_SPAN_TOTAL} total span labels (the paper's "
            f"count), found {span_total} — empty labels would no longer provably "
            "mean 'annotated and supported'"
        )

    passages = {row["source_id"]: row for row in sources}
    rng = random.Random(seed)

    cells: dict[tuple[str, bool], list[dict]] = {}
    refusal_rows: list[dict] = []
    for row in responses:
        if row["quality"] == "truncated":
            continue
        source = passages.get(row["source_id"])
        if source is None:
            raise RuntimeError(f"response {row['id']!r} names unknown source_id")
        # source_info's SHAPE depends on the task, and each shape maps
        # differently. QA: a {question, passages} dict — the question is
        # what the refusal judge should see, the passages are the evidence.
        # Summary: the article as a plain string, with the instruction in
        # ``prompt``. Data2txt: a structured business record (name, hours,
        # attributes...) — the "passage" is that record, serialized so the
        # groundedness judge reads the same facts the responding model was
        # given. Anything else is schema drift and must fail here, not
        # surface as a dict handed where a passage belongs.
        info = source["source_info"]
        task = source["task_type"]
        if task == "QA":
            if not isinstance(info, dict) or not isinstance(
                info.get("question"), str
            ) or not isinstance(info.get("passages"), str):
                raise RuntimeError(
                    f"QA source {source['source_id']!r}: expected source_info "
                    f"{{question, passages}} strings, got {sorted(info) if isinstance(info, dict) else type(info).__name__}"
                )
            query, row_passages = info["question"], [info["passages"]]
        elif task == "Summary":
            if not isinstance(info, str):
                raise RuntimeError(
                    f"Summary source {source['source_id']!r}: expected "
                    f"source_info as a string, got {type(info).__name__}"
                )
            query, row_passages = source["prompt"], [info]
        elif task == "Data2txt":
            if not isinstance(info, dict):
                raise RuntimeError(
                    f"Data2txt source {source['source_id']!r}: expected "
                    f"source_info as a record dict, got {type(info).__name__}"
                )
            query = source["prompt"]
            row_passages = [json.dumps(info, indent=2, sort_keys=True)]
        else:
            raise RuntimeError(f"unknown task_type {task!r} on {source['source_id']!r}")
        supported = not row["labels"]
        record = {
            "ragtruth_id": row["id"],
            "task_type": source["task_type"],
            "query": query,
            "response": row["response"],
            "passages": row_passages,
            "labels": {
                "supported": supported,
                # RAGTruth has no unanswerable-question rows: every question
                # was answerable, so should_refuse is False everywhere in this
                # slice. Correct refusals are the one refusal-judge case no
                # public dataset supplies — recorded as a gap in D14.
                "should_refuse": False,
                "refusal_incorrect": row["quality"] == "incorrect_refusal",
            },
            "source_dataset": source["source"],
            "model": row["model"],
            "split": row["split"],
        }
        if row["quality"] == "incorrect_refusal":
            refusal_rows.append(record)
        else:
            cells.setdefault((source["task_type"], supported), []).append(record)

    rng.shuffle(refusal_rows)
    slice_rows: list[dict] = []
    counts: dict[str, int] = {}
    for task_type in TASK_TYPES:
        for supported in (True, False):
            pool = cells.get((task_type, supported), [])
            if len(pool) < ANSWER_CELL_QUOTA:
                raise RuntimeError(
                    f"stratum ({task_type}, supported={supported}) has {len(pool)} "
                    f"rows, fewer than the {ANSWER_CELL_QUOTA} quota"
                )
            rng.shuffle(pool)
            picked = pool[:ANSWER_CELL_QUOTA]
            for i, record in enumerate(picked, 1):
                tag = "s" if supported else "u"
                record["example_id"] = f"{_ID_PREFIX[task_type]}_{tag}_{i:03d}"
                record["stratum"] = f"{task_type}/{'supported' if supported else 'unsupported'}"
            slice_rows.extend(picked)
            counts[f"{task_type}/{'supported' if supported else 'unsupported'}"] = len(picked)

    for i, record in enumerate(refusal_rows, 1):
        record["example_id"] = f"rt_ref_{i:03d}"
        record["stratum"] = "refusal/incorrect"
    slice_rows.extend(refusal_rows)
    counts["refusal/incorrect"] = len(refusal_rows)

    ids = [row["example_id"] for row in slice_rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate example_ids in slice")
    for row in slice_rows:
        if not row["passages"] or not all(row["passages"]):
            raise RuntimeError(f"{row['example_id']}: empty passages")
        if row["labels"]["refusal_incorrect"] and row["labels"]["should_refuse"]:
            raise RuntimeError(f"{row['example_id']}: inconsistent refusal labels")

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for row in slice_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "ATTRIBUTION.md").write_text(ATTRIBUTION, encoding="utf-8")
    report = {
        "source": f"ParticleMedia/RAGTruth@{RAGTRUTH_REVISION}",
        "seed": seed,
        "n_rows": len(slice_rows),
        "strata": counts,
        "rows_sha256": hashlib.sha256((out_dir / "rows.jsonl").read_bytes()).hexdigest()[:32],
    }
    (out_dir / "slice_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return counts


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument(
        "--input-dir",
        default=None,
        help="directory with response.jsonl + source_info.jsonl already fetched",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="re-derive into a temp dir and byte-diff against the committed slice",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    out_dir = Path(args.out)

    with tempfile.TemporaryDirectory() as tmp:
        rows_dir = Path(args.input_dir) if args.input_dir else Path(tmp) / "raw"
        if args.input_dir is None:
            for name in RAW_FILES:
                print(f"fetching {name} @ {RAGTRUTH_REVISION[:8]}")
                _fetch(f"{RAW_BASE}/{name}", rows_dir / name)

        if args.check:
            check_dir = Path(tmp) / "check"
            counts = build_slice(rows_dir, seed=args.seed, out_dir=check_dir)
            for name in ("rows.jsonl", "ATTRIBUTION.md", "slice_report.json"):
                want = (check_dir / name).read_bytes()
                got_path = out_dir / name
                if not got_path.is_file() or got_path.read_bytes() != want:
                    raise RuntimeError(f"--check failed: {got_path} differs from re-derivation")
            print(f"--check ok: {sum(counts.values())} rows byte-identical to {out_dir}")
            return 0

        counts = build_slice(rows_dir, seed=args.seed, out_dir=out_dir)
    total = sum(counts.values())
    print(f"wrote {total} rows to {out_dir}: {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
