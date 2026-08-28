"""Krippendorff's alpha for the human-ceiling annotation study (issue #16).

Reads the annotation worksheet with both verdict columns filled (or two
single-column files), aligns items by id, and prints raw agreement plus
Krippendorff's alpha — the human ceiling the judge kappas are read
against. Blank cells mean "not scored" and are skipped per-item, which is
exactly the overlap design the alpha implementation supports.

Usage::

    .venv/bin/python scripts/compute_alpha.py annotation_worksheet.csv
    .venv/bin/python scripts/compute_alpha.py worksheet_a.csv worksheet_b.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from anvil.eval.agreement import krippendorffs_alpha_nominal  # noqa: E402

VERDICTS = {"supported", "unsupported"}


def _load(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return {row["item_id"]: row for row in csv.DictReader(fh)}


def _verdict(raw: str, *, item_id: str, column: str) -> str | None:
    value = raw.strip().lower()
    if not value:
        return None
    if value not in VERDICTS:
        raise SystemExit(
            f"{item_id} has verdict {raw!r} in {column}; expected one of {sorted(VERDICTS)}"
        )
    return value


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("worksheet_a", type=Path)
    p.add_argument("worksheet_b", type=Path, nargs="?")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    a = _load(args.worksheet_a)

    ratings: list[tuple[str | None, str | None]] = []
    if args.worksheet_b is None:
        for item_id, row in a.items():
            ratings.append(
                (
                    _verdict(row.get("verdict_a", ""), item_id=item_id, column="verdict_a"),
                    _verdict(row.get("verdict_b", ""), item_id=item_id, column="verdict_b"),
                )
            )
    else:
        b = _load(args.worksheet_b)
        missing = set(a) ^ set(b)
        if missing:
            raise SystemExit(f"worksheets name different items: {sorted(missing)[:5]}")
        for item_id in a:
            va = a[item_id].get("verdict_a") or a[item_id].get("verdict", "")
            vb = b[item_id].get("verdict_a") or b[item_id].get("verdict", "")
            ratings.append(
                (
                    _verdict(va, item_id=item_id, column="worksheet_a"),
                    _verdict(vb, item_id=item_id, column="worksheet_b"),
                )
            )

    paired = [r for r in ratings if all(r)]
    if not paired:
        raise SystemExit("no items scored by both annotators")
    agree = sum(1 for x, y in paired if x == y)
    alpha = krippendorffs_alpha_nominal(paired)
    print(f"items scored by both: {len(paired)}")
    print(f"raw agreement: {agree}/{len(paired)} = {agree / len(paired):.3f}")
    print(f"Krippendorff's alpha (nominal): {alpha:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
