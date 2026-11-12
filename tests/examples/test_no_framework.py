"""``examples/no_framework.py`` runs against the stub (Task 4 Step 1; AC-05.5)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from llm_provider import CallCompleted, CallStarted, Usage

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "no_framework.py"


def load_example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("no_framework_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_no_framework_example_runs() -> None:
    example = load_example()

    sink = example.main()

    started, completed = sink.events
    assert isinstance(started, CallStarted)
    assert isinstance(completed, CallCompleted)
    assert completed.call_id == started.call_id
    assert completed.usage == Usage(11, 7, 18)  # what the stub reports by default
    assert (completed.provider, completed.origin) == ("demo", "user")
