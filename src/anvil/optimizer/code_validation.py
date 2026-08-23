"""Safety checks for optimizer-generated Python candidates."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from types import ModuleType
from typing import Final

from anvil.agents.memory_system import find_memory_system_subclass


class CodeValidationError(Exception):
    """Raised when a code candidate fails validation."""


# Kept as public module-level data so later code-mode work can extend the
# policy without changing the AST walker.
FORBIDDEN_STRING_PATTERNS: Final[tuple[str, ...]] = (
    r"(?<![A-Za-z0-9_])test_",
    r"(?<![A-Za-z0-9_])eval(?=$|[/\\])",
    r"(?<![A-Za-z0-9_])solution(?![A-Za-z0-9_])",
    r"(?<![A-Za-z0-9_])golden_set(?![A-Za-z0-9_])",
    r"(?<![A-Za-z0-9_])answer_key(?![A-Za-z0-9_])",
    r"(?<![A-Za-z0-9_])ground_truth(?![A-Za-z0-9_])",
)
FORBIDDEN_IMPORT_PREFIXES: Final[tuple[str, ...]] = ("test_", "eval_")
FORBIDDEN_IMPORT_TERMS: Final[tuple[str, ...]] = ("solution",)


def _forbidden_string(value: str) -> str | None:
    for pattern in FORBIDDEN_STRING_PATTERNS:
        if re.search(pattern, value, flags=re.IGNORECASE):
            return pattern
    return None


def _forbidden_import(module_name: str) -> bool:
    name = module_name.lower()
    if name.startswith(FORBIDDEN_IMPORT_PREFIXES):
        return True
    return any(
        re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", name)
        for term in FORBIDDEN_IMPORT_TERMS
    )


def check_ast_denylist(file_path: Path | str) -> None:
    """Reject references to test, evaluation, solution, or answer data."""
    path = Path(file_path)
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise CodeValidationError(f"could not parse code candidate {path}: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _forbidden_string(node.value) is not None:
                raise CodeValidationError(
                    f"forbidden reference in string literal at line {node.lineno}"
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _forbidden_import(alias.name):
                    raise CodeValidationError(
                        f"forbidden import {alias.name!r} at line {node.lineno}"
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _forbidden_import(module):
                raise CodeValidationError(
                    f"forbidden import {module!r} at line {node.lineno}"
                )


def validate_imports(file_path: Path | str) -> ModuleType:
    """Load a candidate from an isolated temporary working directory.

    Returns the imported module so a caller can inspect what the candidate
    actually defines without importing it a second time.
    """
    path = Path(file_path)
    if not path.is_file():
        raise CodeValidationError(f"code candidate is not a file: {path}")

    module_name = f"_anvil_candidate_{uuid.uuid4().hex}"
    previous_cwd = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="anvil-code-validation-") as temp_dir:
            isolated_path = Path(temp_dir) / path.name
            shutil.copy2(path, isolated_path)
            os.chdir(temp_dir)
            spec = importlib.util.spec_from_file_location(module_name, isolated_path)
            if spec is None or spec.loader is None:
                raise CodeValidationError(f"could not create import spec for {path}")
            module = importlib.util.module_from_spec(spec)
            # Deliberately do not register the candidate in sys.modules.
            spec.loader.exec_module(module)
            return module
    except CodeValidationError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise CodeValidationError(
            f"code candidate {path} failed to import: {exc.__class__.__name__}: {exc}"
        ) from exc
    finally:
        os.chdir(previous_cwd)


def check_constructor_contract(module: ModuleType) -> None:
    """Reject a candidate whose ``__init__`` the eval could not call.

    ``anvil.eval.runner._load_memory_system`` instantiates every candidate as
    ``cls(llm_client=..., model=...)``. A subclass that names those parameters
    differently, or makes them positional-only, raises ``TypeError`` at that
    call -- inside the eval, after the optimizer session has been paid for,
    where judgeability reads an exception as an infrastructure failure and
    aborts the round. An invalid candidate should be *rejected*, which is a
    different outcome with a different consequence: the round reverts and the
    loop keeps going.

    Checked by binding the arguments rather than by comparing parameter names,
    so a candidate that accepts them via ``**kwargs`` passes -- it can in fact
    be constructed, which is the only thing this guards.
    """
    try:
        cls = find_memory_system_subclass(module)
    except ValueError as exc:
        # Zero or several concrete subclasses. Same failure mode one step
        # earlier: `write_agent` writes the module `agent_module` points at, so
        # a candidate the eval cannot find an agent in was never valid -- it
        # just failed later, inside the eval, as an infrastructure error.
        raise CodeValidationError(f"invalid code candidate: {exc}") from exc
    try:
        inspect.signature(cls).bind(llm_client=None, model="")
    except (TypeError, ValueError) as exc:
        raise CodeValidationError(
            f"{cls.__name__}.__init__ cannot be called as "
            f"cls(llm_client=..., model=...): {exc}. Every MemorySystem "
            f"candidate must accept those keyword arguments -- inherit "
            f"MemorySystem.__init__ or declare a compatible signature."
        ) from exc


def validate_code_candidate(file_path: Path | str) -> None:
    """Run the cheap AST policy check before attempting an isolated import."""
    check_ast_denylist(file_path)
    module = validate_imports(file_path)
    check_constructor_contract(module)
