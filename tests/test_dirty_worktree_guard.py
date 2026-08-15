from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from anvil.loop.git_ops import check_clean_worktree


def _load_run_round_script() -> ModuleType:
    script = Path(__file__).parents[1] / "scripts" / "run_round.py"
    spec = importlib.util.spec_from_file_location("run_round_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_clean_worktree_rejects_dirty_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)

    check_clean_worktree()

    dirty_file = tmp_path / "uncommitted.txt"
    dirty_file.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"Working tree is dirty.*Uncommitted files"):
        check_clean_worktree()

    dirty_file.unlink()
    check_clean_worktree()


def test_allow_dirty_skips_check(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_run_round_script()

    def fail_if_checked() -> None:
        pytest.fail("clean-worktree check should be skipped")

    monkeypatch.setattr(module, "check_clean_worktree", fail_if_checked)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )

    assert module.main(["--allow-dirty"]) == 2
    assert "WARNING: --allow-dirty specified" in capsys.readouterr().out
