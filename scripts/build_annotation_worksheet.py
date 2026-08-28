"""Build the blind annotation worksheet for the human-ceiling study (issue #16).

A judge cannot fairly be held to better agreement with humans than humans
reach with each other, so judge-vs-human kappa only means something next to
the human-vs-human ceiling. This emits ~200 stratified, blinded items from
the vendored RAGTruth slice for two independent annotators, and the
protocol they follow. Their filled worksheets feed
``scripts/compute_alpha.py``, and the alpha becomes the denominator in
"judge agreement as a fraction of the human ceiling".

The items are drawn from the SAME slice the judges were measured on, so the
two studies are directly comparable; labels stay in ``rows.jsonl`` and
never enter the worksheet.

Usage::

    .venv/bin/python scripts/build_annotation_worksheet.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SLICE_PATH = REPO_ROOT / "eval" / "judge_validation" / "ragtruth" / "rows.jsonl"
OUT_CSV = REPO_ROOT / "eval" / "judge_validation" / "annotation_worksheet.csv"
PROTOCOL_PATH = REPO_ROOT / "eval" / "judge_validation" / "ANNOTATION_PROTOCOL.md"

WORKSHEET_SIZE = 200

PROTOCOL = """\
# Annotation protocol — human ceiling for judge agreement

Two independent annotators, one adjudicator (who may be one of the two after
the independent pass). ~200 items, blind. Budget ~90 seconds per item; split
into sessions of 50.

## The verdict

For each item, read the query, the response, and the passage(s). Mark exactly
one verdict per item:

- **supported** — every factual claim in the response is entailed by the
  passage(s), or is directly quoted from them. Reasonable summarization and
  paraphrase are supported.
- **unsupported** — any factual claim in the response contradicts the
  passage(s), or asserts something the passage(s) do not establish
  (invented names, dates, numbers, causal claims, quotes). One bad claim
  makes the item unsupported.

Do not grade fluency, completeness, or whether the response answers the
question well — only whether the passage(s) carry what the response asserts.
If the passages themselves are missing or empty, mark **unsupported** and
flag the item in the notes column.

## Procedure

1. Each annotator fills their own column of `annotation_worksheet.csv`
   (`verdict_a` / `verdict_b`) independently. No conferring, no shared
   screen, no looking at the other column.
2. Run `.venv/bin/python scripts/compute_alpha.py worksheet_a.csv
   worksheet_b.csv` (or on one file with both columns filled) to get
   Krippendorff's alpha and the raw agreement.
3. Adjudicate every disagreement: read the item together, decide the final
   label, record it in `adjudicated`. The adjudicated labels are the local
   human reference; the alpha from step 2 is the ceiling the judges are
   read against.
4. Record the outcome in `docs/decisions.md` under D14: alpha, n, and the
   judge kappas restated as a fraction of it.

The expected range, from the only applicable published number (FActScore's
~72% crowd-vs-expert agreement on atomic facts), is alpha roughly 0.5–0.6.
Far above that suggests the items are too easy to be a fair ceiling; far
below suggests the label definitions above are ambiguous — fix the
definitions and redo a 50-item pilot before the full pass.
"""


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slice", default=str(SLICE_PATH))
    p.add_argument("--out", default=str(OUT_CSV))
    p.add_argument("--protocol", default=str(PROTOCOL_PATH))
    p.add_argument("--size", type=int, default=WORKSHEET_SIZE)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    rows = [json.loads(line) for line in Path(args.slice).read_text().splitlines()]
    answer_rows = [r for r in rows if not r["labels"]["refusal_incorrect"]]

    # Stratified by slice stratum, proportional to the slice's own mix.
    rng = random.Random(args.seed)
    strata: dict[str, list[dict]] = {}
    for row in answer_rows:
        strata.setdefault(row["stratum"], []).append(row)
    picked: list[dict] = []
    for stratum in sorted(strata):
        pool = strata[stratum]
        rng.shuffle(pool)
        share = round(len(pool) / len(answer_rows) * args.size)
        picked.extend(pool[:share])
    rng.shuffle(picked)
    picked = picked[: args.size]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["item_id", "task_type", "query", "response", "passages", "verdict_a", "verdict_b", "adjudicated", "notes"]
        )
        for row in picked:
            writer.writerow(
                [
                    row["example_id"],
                    row["task_type"],
                    row["query"],
                    row["response"],
                    "\n\n---\n\n".join(row["passages"]),
                    "",
                    "",
                    "",
                    "",
                ]
            )
    Path(args.protocol).write_text(PROTOCOL, encoding="utf-8")
    print(f"wrote {len(picked)} blind items to {out}")
    print(f"wrote the protocol to {args.protocol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
