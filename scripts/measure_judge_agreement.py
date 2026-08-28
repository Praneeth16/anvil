"""Measure judge-vs-human agreement on the vendored RAGTruth slice (issue #16).

Scores the slice with the same scorer objects the live eval builds — same
``build_scorers`` construction, same judge endpoint — and pairs each verdict
with the human label:

* ``retrieval_groundedness`` vs span labels (any hallucinated span means the
  human marked the response unsupported). Answer rows only; the judge's own
  applicability rule excludes refusals, exactly as in the eval.
* ``refusal_appropriateness`` vs the ``incorrect_refusal`` quality flag.
  RAGTruth carries no correct-refusal rows, so this judge is measured on
  appropriate answers vs inappropriate refusals only — the gap is recorded
  in the report.
* ``correctness`` is not measurable here (RAGTruth has no reference facts)
  and says so in the report rather than being quietly absent.

The refusal judge is domain-scoped: grading open-domain RAGTruth rows under
the shipped MultiHopRAG domain context would measure domain mismatch, not
judge error. It is therefore run with a domain override describing this
slice, and the override is captured in the report's scorer fingerprint.

Usage::

    .venv/bin/python scripts/measure_judge_agreement.py \
        --profile fe-vm-lakebase-praneeth
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import mlflow  # noqa: E402
from mlflow.entities import SpanType  # noqa: E402

from anvil.cli import ExitCode, run_cli  # noqa: E402
from anvil.eval.agreement import bootstrap_kappa_ci, cohens_kappa, rates  # noqa: E402
from anvil.eval.cache import compute_scorer_fingerprint  # noqa: E402
from anvil.eval.scorers import (  # noqa: E402
    GROUNDEDNESS_SCORER_NAME,
    REFUSAL_SCORER_NAME,
    build_scorers,
)
from anvil.runtime.client import build_databricks_client  # noqa: E402
from anvil.runtime.loader import default_runtime_config_path, load_endpoints  # noqa: E402
from anvil.runtime.models import ScorerConfig  # noqa: E402

SLICE_PATH = REPO_ROOT / "eval" / "judge_validation" / "ragtruth" / "rows.jsonl"
OUT_PATH = REPO_ROOT / "eval" / "runs" / "judge_agreement_ragtruth.json"

# The refusal judge grades "should this have been refused" against a domain
# description. The shipped default describes the MultiHopRAG news corpus;
# under it, every off-corpus RAGTruth question looks out of scope and an
# incorrect refusal reads as appropriate — domain mismatch, not judge error.
RAGTRUTH_JUDGE_DOMAIN_NAME = "open-domain RAG (RAGTruth slice)"
RAGTRUTH_JUDGE_DOMAIN_CONTEXT = (
    "an open-domain retrieval-augmented assistant evaluated on the RAGTruth "
    "slice: news summarization, question answering over provided passages, "
    "and data-to-text tasks. Any question the provided passages can answer "
    "is in scope; refusing an answerable question is inappropriate."
)

# Above this share of judge-call errors the measurement is partial and the
# report must not be written — same posture as the eval's judgability floor.
MAX_JUDGE_ERROR_RATE = 0.05


def _synthesize_trace(query: str, response: str, passages: list[str], doc_uri: str):
    """A minimal two-span trace carrying the retrieval context.

    Mirrors the live path exactly: root CHAIN span with the query as inputs
    and the response as outputs, one RETRIEVER child whose outputs use the
    ``page_content`` / ``metadata.doc_uri`` keys
    ``extract_retrieval_context_from_trace`` reads. The groundedness judge
    derives request, response and context from the trace, so the trace —
    not the scorer kwargs — is what carries them.
    """
    with mlflow.start_span(name="anvil.predict", span_type=SpanType.CHAIN) as span:
        span.set_inputs({"query": query})
        with mlflow.start_span(
            name="search_knowledge_base", span_type=SpanType.RETRIEVER
        ) as retriever:
            retriever.set_inputs({"query": query, "k": len(passages)})
            retriever.set_outputs(
                [
                    {
                        "page_content": passage,
                        "metadata": {"doc_uri": doc_uri, "title": doc_uri},
                    }
                    for passage in passages
                ]
            )
        span.set_outputs({"response": response})
    trace_id = mlflow.get_last_active_trace_id()
    assert trace_id is not None, "span context lost: no active trace id"
    return mlflow.get_trace(trace_id)


def _named(scorers: list, name: str):
    for scorer in scorers:
        if getattr(scorer, "name", None) == name or getattr(scorer, "__name__", None) == name:
            return scorer
    raise RuntimeError(f"scorer {name!r} not found in build_scorers output")


def _groundedness_verdict(result) -> bool | None:
    """Map a groundedness Feedback to a boolean, preserving inapplicability."""
    if result is None:
        return None
    if result.value == "yes":
        return True
    if result.value == "no":
        return False
    raise RuntimeError(f"unexpected groundedness verdict value: {result.value!r}")


def _score_row(row: dict, groundedness_scorer, refusal_scorer) -> dict:
    """Score one slice row with both judges. Pure orchestration: the judges
    themselves are injected, so tests drive this with fakes."""
    out: dict = {"example_id": row["example_id"], "stratum": row["stratum"], "errors": []}
    doc_uri = f"ragtruth:{row['ragtruth_id']}"

    if not row["labels"]["refusal_incorrect"]:
        trace = _synthesize_trace(row["query"], row["response"], row["passages"], doc_uri)
        try:
            result = groundedness_scorer.run(
                inputs={"query": row["query"]},
                outputs=row["response"],
                expectations={"expected_doc_ids": [doc_uri]},
                trace=trace,
            )
            verdict = _groundedness_verdict(result)
            if verdict is not None:
                out["groundedness"] = {
                    "judge": verdict,
                    "human": row["labels"]["supported"],
                }
        except Exception as exc:  # noqa: BLE001 — recorded, counted, excluded
            out["errors"].append(f"groundedness: {exc.__class__.__name__}: {exc}")

    try:
        result = refusal_scorer.run(
            inputs={"query": row["query"]},
            outputs=row["response"],
            expectations={"should_refuse": row["labels"]["should_refuse"]},
            trace=None,
        )
        if not isinstance(result.value, bool):
            raise RuntimeError(f"unexpected refusal verdict value: {result.value!r}")
        out["refusal"] = {
            "judge": result.value,
            "human": not row["labels"]["refusal_incorrect"],
        }
    except Exception as exc:  # noqa: BLE001 — recorded, counted, excluded
        out["errors"].append(f"refusal: {exc.__class__.__name__}: {exc}")
    return out


def _judge_report(pairs: list[tuple[bool, bool]], strata: dict[str, list[tuple[bool, bool]]]) -> dict:
    report: dict = {
        **rates(pairs),
        "kappa": cohens_kappa(pairs),
        "kappa_ci95": bootstrap_kappa_ci(pairs),
    }
    report["per_stratum"] = {
        stratum: {**rates(cell), "kappa": cohens_kappa(cell)} if cell else {}
        for stratum, cell in sorted(strata.items())
    }
    return report


def measure(rows: list[dict], scorers: list, *, workers: int = 4) -> dict:
    """Score every row and compute the agreement report."""
    groundedness_scorer = _named(scorers, GROUNDEDNESS_SCORER_NAME)
    refusal_scorer = _named(scorers, REFUSAL_SCORER_NAME)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        scored = list(
            pool.map(lambda row: _score_row(row, groundedness_scorer, refusal_scorer), rows)
        )

    errors = [e for s in scored for e in s["errors"]]
    # Calls ATTEMPTED, not calls that produced a verdict: refusal is
    # attempted on every row, groundedness on every answer row (including
    # the ones that errored — an errored call is still a call the ceiling
    # must see).
    n_calls = len(scored) + sum(
        1
        for s in scored
        if "groundedness" in s or any(e.startswith("groundedness") for e in s["errors"])
    )
    if errors and len(errors) / max(n_calls, 1) > MAX_JUDGE_ERROR_RATE:
        raise RuntimeError(
            f"{len(errors)} judge-call errors over {n_calls} attempted calls exceeds "
            f"the {MAX_JUDGE_ERROR_RATE:.0%} ceiling — the measurement is partial; "
            f"first error: {errors[0]}"
        )

    def pairs_of(items):
        return [(item["human"], item["judge"]) for item in items]

    def strata_of(items):
        out: dict[str, list[tuple[bool, bool]]] = {}
        for item in items:
            out.setdefault(item["stratum"], []).append((item["human"], item["judge"]))
        return out

    grounded_items = [
        {**s["groundedness"], "stratum": s["stratum"]} for s in scored if "groundedness" in s
    ]
    refusal_items = [
        {**s["refusal"], "stratum": s["stratum"]} for s in scored if "refusal" in s
    ]
    return {
        "judges": {
            GROUNDEDNESS_SCORER_NAME: _judge_report(pairs_of(grounded_items), strata_of(grounded_items)),
            REFUSAL_SCORER_NAME: _judge_report(pairs_of(refusal_items), strata_of(refusal_items)),
            "correctness": {
                "status": "not measurable against RAGTruth",
                "reason": (
                    "RAGTruth carries no reference answers or expected facts, "
                    "and the correctness judge grades against exactly those. "
                    "Validating it needs a dataset with expert-verified "
                    "reference answers; see docs/decisions.md."
                ),
            },
        },
        "n_errors": len(errors),
        "errors": errors[:20],
    }


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default="DEFAULT")
    p.add_argument("--slice", default=str(SLICE_PATH))
    p.add_argument("--out", default=str(OUT_PATH))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--scaffold",
        default=str(REPO_ROOT / "scaffold"),
        help="scaffold whose sibling harness/config.yaml supplies the judge endpoint",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    rows = [json.loads(line) for line in Path(args.slice).read_text().splitlines()]
    print(f"loaded {len(rows)} slice rows from {args.slice}")

    client = build_databricks_client(profile=args.profile)
    _, judge_model = load_endpoints(
        default_runtime_config_path(Path(args.scaffold))
    )
    scorers = build_scorers(
        judge_client=client,
        judge_model=judge_model,
        judge_domain_name=RAGTRUTH_JUDGE_DOMAIN_NAME,
        judge_domain_context=RAGTRUTH_JUDGE_DOMAIN_CONTEXT,
    )
    print(f"judges built against {judge_model}; scoring with {args.workers} workers")

    report = measure(rows, scorers, workers=args.workers)
    report.update(
        {
            "measured_at": datetime.now(UTC).isoformat(),
            "judge_model": judge_model,
            "slice_path": str(args.slice),
            "slice_sha256": hashlib.sha256(Path(args.slice).read_bytes()).hexdigest()[:32],
            "scorer_fingerprint": compute_scorer_fingerprint(
                [ScorerConfig(name=n) for n in ("correctness", "retrieval_groundedness", "refusal_appropriateness")],
                judge_domain_name=RAGTRUTH_JUDGE_DOMAIN_NAME,
                judge_domain_context=RAGTRUTH_JUDGE_DOMAIN_CONTEXT,
            ),
        }
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, judge in report["judges"].items():
        if "kappa" in judge:
            lo, hi = judge["kappa_ci95"]
            print(
                f"{name}: n={judge['n']} kappa={judge['kappa']:.3f} "
                f"[{lo:.3f}, {hi:.3f}] fp={judge['fp_rate']:.3f} fn={judge['fn_rate']:.3f}"
            )
    print(f"report written to {out_path}")
    return ExitCode.OK


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
