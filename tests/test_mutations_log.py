"""Tests for the mutations JSONL log."""

from __future__ import annotations

from pathlib import Path

from anvil.loop.mutations_log import (
    MutationRecord,
    append_mutation,
    load_mutations,
    mutations_log_path,
)


def test_round_trip_one_record(tmp_path: Path) -> None:
    rec = MutationRecord.new(
        round_id=1,
        git_branch="anvil/exp-round-1",
        git_commit_sha="a" * 40,
        parent_commit_sha="b" * 40,
        files_added=["scaffold/rules/foo.md"],
        diff_summary="add_rule rules/foo.md: seed",
        baseline_score=0.78,
        mutated_score=0.81,
        score_delta=0.03,
        decision="keep",
        mlflow_eval_run_id="run_abc",
        parse_status="ok",
    )
    append_mutation(tmp_path, rec)
    loaded = load_mutations(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].round_id == 1
    assert loaded[0].decision == "keep"
    assert loaded[0].score_delta == 0.03
    assert loaded[0].mutation_id.startswith("mut_")


def test_append_multiple_rounds(tmp_path: Path) -> None:
    for round_id in range(1, 4):
        rec = MutationRecord.new(
            round_id=round_id,
            git_branch=f"anvil/exp-round-{round_id}",
            git_commit_sha="0" * 40,
            parent_commit_sha="0" * 40,
            decision="revert",
            parse_status="ok",
        )
        append_mutation(tmp_path, rec)

    loaded = load_mutations(tmp_path)
    assert [r.round_id for r in loaded] == [1, 2, 3]
    assert all(r.decision == "revert" for r in loaded)


def test_mutation_id_is_unique_per_record() -> None:
    a = MutationRecord.new(
        round_id=1,
        git_branch="b",
        git_commit_sha="x",
        parent_commit_sha="y",
        parse_status="ok",
    )
    b = MutationRecord.new(
        round_id=1,
        git_branch="b",
        git_commit_sha="x",
        parent_commit_sha="y",
        parse_status="ok",
    )
    assert a.mutation_id != b.mutation_id


def test_load_returns_empty_when_file_missing(tmp_path: Path) -> None:
    assert load_mutations(tmp_path) == []
    assert mutations_log_path(tmp_path) == tmp_path / "eval" / "mutations.jsonl"
