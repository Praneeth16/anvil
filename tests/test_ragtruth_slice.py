"""The RAGTruth vendoring script against a synthetic fixture (issue #16).

The real files are verified by the script's own hard-fail checks at build
time (schema keys, quality vocabulary, the paper's 14,289 span total); what
these tests pin is the *logic* — join, strata, refusal mapping, exclusion,
determinism, --check — against rows small enough to see whole. Module
constants are monkeypatched so a 20-row fixture can stand in for 17,790.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def module():
    spec = importlib.util.spec_from_file_location(
        "build_ragtruth_slice_script", REPO_ROOT / "scripts" / "build_ragtruth_slice.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    return mod


def _response(rid: str, source_id: str, quality: str = "good", n_labels: int = 0) -> dict:
    return {
        "id": rid,
        "source_id": source_id,
        "model": "m",
        "temperature": 0.7,
        "labels": [
            {"start": 0, "end": 1, "text": "x", "label_type": "Evident Conflict"}
            for _ in range(n_labels)
        ],
        "split": "test",
        "quality": quality,
        "response": f"response {rid}",
    }


def _source(sid: str, task_type: str) -> dict:
    # QA: {question, passages} dict. Summary: reference text as a string.
    # Data2txt: a structured record dict (serialized downstream).
    info: str | dict
    if task_type == "QA":
        info = {"question": f"question for {sid}", "passages": f"passage for {sid}"}
    elif task_type == "Summary":
        info = f"passage for {sid}"
    else:
        info = {"name": f"business {sid}", "city": "springfield"}
    return {
        "source_id": sid,
        "task_type": task_type,
        "source": "unit-test",
        "source_info": info,
        "prompt": f"prompt for {sid}",
    }


def _write_fixture(rows_dir: Path, responses: list[dict], sources: list[dict]) -> None:
    rows_dir.mkdir(parents=True, exist_ok=True)
    (rows_dir / "response.jsonl").write_text(
        "\n".join(json.dumps(r) for r in responses) + "\n", encoding="utf-8"
    )
    (rows_dir / "source_info.jsonl").write_text(
        "\n".join(json.dumps(s) for s in sources) + "\n", encoding="utf-8"
    )


def _full_pool() -> tuple[list[dict], list[dict]]:
    """3 rows per (task_type, supported) cell + 2 incorrect_refusal + 1 truncated."""
    responses: list[dict] = []
    sources: list[dict] = []
    n = 0
    for task_type in ("QA", "Summary", "Data2txt"):
        for supported in (True, False):
            for _ in range(3):
                n += 1
                sources.append(_source(f"s{n}", task_type))
                responses.append(_response(f"r{n}", f"s{n}", n_labels=0 if supported else 1))
    sources.extend([_source("sR1", "QA"), _source("sR2", "Summary"), _source("sT", "QA")])
    responses.extend(
        [
            _response("rR1", "sR1", quality="incorrect_refusal"),
            _response("rR2", "sR2", quality="incorrect_refusal", n_labels=1),
            _response("rT", "sT", quality="truncated"),
        ]
    )
    return responses, sources


def _build(module, rows_dir: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(module, "ANSWER_CELL_QUOTA", 2)
    monkeypatch.setattr(module, "EXPECTED_LABEL_SPAN_TOTAL", 10)
    return module.build_slice(rows_dir, seed=42, out_dir=out_dir)


@pytest.mark.unit
def test_slice_strata_and_refusal_mapping(
    tmp_path: Path, module, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses, sources = _full_pool()
    _write_fixture(tmp_path / "raw", responses, sources)
    counts = _build(module, tmp_path / "raw", tmp_path / "out", monkeypatch)

    # 2 per cell x 6 cells + both refusal rows; the truncated row is gone.
    assert counts["QA/supported"] == 2
    assert counts["QA/unsupported"] == 2
    assert counts["refusal/incorrect"] == 2
    assert sum(counts.values()) == 14

    rows = [json.loads(line) for line in (tmp_path / "out" / "rows.jsonl").read_text().splitlines()]
    refusal = next(r for r in rows if r["ragtruth_id"] == "rR1")
    assert refusal["labels"] == {
        "supported": True,
        "should_refuse": False,
        "refusal_incorrect": True,
    }
    assert refusal["stratum"] == "refusal/incorrect"
    assert all(r["ragtruth_id"] != "rT" for r in rows)
    # The query is what the responding model saw; the passage is the
    # reference for the SAME source id (rows are sampled, so anchor on the
    # join, not on a specific id surviving the shuffle).
    for row in rows:
        sid = row["query"].rsplit(" ", 1)[-1]
        assert sid in row["passages"][0], f"{row['example_id']}: passage from another source"
    # And the artifacts the check mode diffs exist.
    assert (tmp_path / "out" / "ATTRIBUTION.md").read_text().startswith("# RAGTruth")
    report = json.loads((tmp_path / "out" / "slice_report.json").read_text())
    assert report["n_rows"] == 14


@pytest.mark.unit
def test_slice_is_deterministic_under_the_seed(
    tmp_path: Path, module, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses, sources = _full_pool()
    _write_fixture(tmp_path / "raw", responses, sources)
    _build(module, tmp_path / "raw", tmp_path / "a", monkeypatch)
    _build(module, tmp_path / "raw", tmp_path / "b", monkeypatch)
    assert (tmp_path / "a" / "rows.jsonl").read_bytes() == (tmp_path / "b" / "rows.jsonl").read_bytes()


@pytest.mark.unit
def test_a_short_stratum_fails_loudly(
    tmp_path: Path, module, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses, sources = _full_pool()
    # One supported QA row left against a quota of two (r1-r3 are QA/supported).
    responses = [r for r in responses if r["id"] not in ("r1", "r2")]
    _write_fixture(tmp_path / "raw", responses, sources)
    with pytest.raises(RuntimeError, match="fewer than the 2 quota"):
        _build(module, tmp_path / "raw", tmp_path / "out", monkeypatch)


@pytest.mark.unit
def test_schema_drift_fails_with_the_actual_keys(
    tmp_path: Path, module, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses, sources = _full_pool()
    for r in responses:
        del r["quality"]
    _write_fixture(tmp_path / "raw", responses, sources)
    with pytest.raises(RuntimeError, match="actual keys are"):
        _build(module, tmp_path / "raw", tmp_path / "out", monkeypatch)


@pytest.mark.unit
def test_an_unknown_quality_value_fails(
    tmp_path: Path, module, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses, sources = _full_pool()
    responses[0]["quality"] = "surprising"
    _write_fixture(tmp_path / "raw", responses, sources)
    with pytest.raises(RuntimeError, match="unexpected \['surprising'\]"):
        _build(module, tmp_path / "raw", tmp_path / "out", monkeypatch)


@pytest.mark.unit
def test_the_span_total_guard_fails_when_labels_no_longer_mean_supported(
    tmp_path: Path, module, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses, sources = _full_pool()
    monkeypatch.setattr(module, "ANSWER_CELL_QUOTA", 2)
    monkeypatch.setattr(module, "EXPECTED_LABEL_SPAN_TOTAL", 999)
    _write_fixture(tmp_path / "raw", responses, sources)
    with pytest.raises(RuntimeError, match="span labels"):
        module.build_slice(tmp_path / "raw", seed=42, out_dir=tmp_path / "out")


@pytest.mark.unit
def test_check_mode_byte_diffs_the_committed_slice(
    tmp_path: Path, module, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses, sources = _full_pool()
    _write_fixture(tmp_path / "raw", responses, sources)
    monkeypatch.setattr(module, "ANSWER_CELL_QUOTA", 2)
    monkeypatch.setattr(module, "EXPECTED_LABEL_SPAN_TOTAL", 10)

    out = tmp_path / "out"
    assert module.main(["--input-dir", str(tmp_path / "raw"), "--out", str(out)]) == 0
    assert (
        module.main(["--input-dir", str(tmp_path / "raw"), "--out", str(out), "--check"]) == 0
    )
    # Perturb the committed slice: the check must notice.
    (out / "rows.jsonl").write_text(
        (out / "rows.jsonl").read_text().replace("response r2", "tampered"), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="--check failed"):
        module.main(["--input-dir", str(tmp_path / "raw"), "--out", str(out), "--check"])
