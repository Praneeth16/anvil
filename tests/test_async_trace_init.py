"""Tests for the process-start MLflow trace logging default."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("explicit_value", "expected"),
    [(None, "false"), ("true", "true")],
)
def test_import_anvil_sets_async_trace_logging_default(
    explicit_value: str | None, expected: str
) -> None:
    env = os.environ.copy()
    env.pop("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", None)
    if explicit_value is not None:
        env["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = explicit_value

    repo_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(repo_root / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import anvil; import os; "
            "print(os.environ['MLFLOW_ENABLE_ASYNC_TRACE_LOGGING'])",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.strip() == expected
