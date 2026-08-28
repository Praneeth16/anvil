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
