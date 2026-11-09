"""Boundary rules for the core (Task 1, Step 3).

The core is stdlib only, does no I/O and reads no configuration. Only ``testing.py`` and
``discovery.py`` open connections. The integrations import the core and two shared helpers.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

THIRD_PARTY = (
    "crewai",
    "litellm",
    "langchain",
    "langchain_core",
    "langgraph",
    "openai",
    "anthropic",
    "cryptography",
    "pydantic",
    "fastapi",
    "sqlalchemy",
    "httpx",
    "opentelemetry",
)
ENV_ATTRIBUTES = {"environ", "getenv", "putenv"}
CONNECTION_MODULES = ("socket", "urllib", "http.client", "http.server")
CONNECTION_OWNERS = {"testing.py", "discovery.py"}
SHARED_HELPERS = {"_names", "_pydantic"}


@pytest.fixture(scope="module")
def package_root() -> Path:
    import llm_provider  # ModuleNotFoundError until the package exists

    return Path(llm_provider.__file__).resolve().parent


def _module_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _core_files(root: Path) -> list[Path]:
    """Every module except ``integrations/`` and ``testing.py``."""
    return [
        p
        for p in _module_files(root)
        if p.relative_to(root).parts[0] != "integrations" and p.relative_to(root).as_posix() != "testing.py"
    ]


def _integration_files(root: Path) -> list[Path]:
    return [p for p in _module_files(root) if p.relative_to(root).parts[0] == "integrations"]


def _package_parts(root: Path, path: Path) -> list[str]:
    """The dotted package a file sits in, as parts: ``["llm_provider", "telemetry"]``."""
    return [root.name, *path.relative_to(root).parts[:-1]]


def _imported_modules(tree: ast.AST, package: list[str]) -> list[str]:
    """Absolute dotted names a module imports, relative imports resolved.

    ``from a import b`` yields ``a`` and ``a.b``, so ``from http import client`` is seen.
    """
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                module = ".".join([*base, *(node.module.split(".") if node.module else [])])
            else:
                module = node.module or ""
            found.append(module)
            found.extend(f"{module}.{alias.name}" for alias in node.names)
    return found


def _is_or_under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _environment_reads(tree: ast.AST) -> list[str]:
    """``os.environ``, ``os.getenv``, ``os.putenv`` (also through an alias or ``from os import``)."""
    os_names = {"os"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            os_names.update(a.asname for a in node.names if a.name == "os" and a.asname)
    hits: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in ENV_ATTRIBUTES
            and isinstance(node.value, ast.Name)
            and node.value.id in os_names
        ):
            hits.append(f"os.{node.attr}")
        elif isinstance(node, ast.ImportFrom) and node.module == "os" and node.level == 0:
            hits.extend(f"from os import {a.name}" for a in node.names if a.name in ENV_ATTRIBUTES)
    return hits


def _parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_core_import_pulls_in_no_third_party_module() -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-c", "import llm_provider, sys, json; print(json.dumps(sorted(sys.modules)))"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr  # ModuleNotFoundError: llm_provider until it is installed
    loaded = json.loads(result.stdout)
    leaked = sorted(m for m in loaded for banned in THIRD_PARTY if _is_or_under(m, banned))
    assert not leaked, f"importing llm_provider loaded third-party modules: {leaked}"


def test_core_never_imports_integrations_or_reads_the_environment(package_root: Path) -> None:
    problems: list[str] = []
    for path in _core_files(package_root):
        tree = _parse(path)
        rel = path.relative_to(package_root).as_posix()
        for name in _imported_modules(tree, _package_parts(package_root, path)):
            if _is_or_under(name, f"{package_root.name}.integrations") or _is_or_under(name, "dotenv"):
                problems.append(f"{rel} imports {name}")
        problems.extend(f"{rel} uses {hit}" for hit in _environment_reads(tree))
    assert not problems, "\n".join(problems)


def test_only_testing_and_discovery_open_connections(package_root: Path) -> None:
    problems: list[str] = []
    for path in _core_files(package_root):
        rel = path.relative_to(package_root).as_posix()
        if rel in CONNECTION_OWNERS:
            continue
        for name in _imported_modules(_parse(path), _package_parts(package_root, path)):
            if any(_is_or_under(name, banned) for banned in CONNECTION_MODULES):
                problems.append(f"{rel} imports {name}")
    assert not problems, "\n".join(problems)


def test_integrations_import_only_the_core_and_shared_helpers(package_root: Path) -> None:
    own = f"{package_root.name}.integrations"
    problems: list[str] = []
    for path in _integration_files(package_root):
        rel = path.relative_to(package_root).as_posix()
        tree = _parse(path)
        names = _imported_modules(tree, _package_parts(package_root, path))
        for name in names:
            if _is_or_under(name, own) and name != own:
                if name.split(".")[2] not in SHARED_HELPERS:
                    problems.append(f"{rel} imports {name}")
            if _is_or_under(name, "pydantic") and path.name != "_pydantic.py":
                problems.append(f"{rel} imports {name}")
        if path.name == "_names.py":
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = ["." * node.level + (node.module or "")]
                else:
                    continue
                problems.extend(
                    f"{rel} imports {m}, which is not standard library"
                    for m in modules
                    if m.startswith(".") or m.split(".")[0] not in sys.stdlib_module_names
                )
    assert not problems, "\n".join(problems)
