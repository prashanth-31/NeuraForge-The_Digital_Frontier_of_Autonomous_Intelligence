"""Compat shim ensuring old import path still locates the active tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_REAL_TEST_PATH = (
	Path(__file__).resolve().parents[2] / "tests" / "test_tool_policy.py"
)
spec = importlib.util.spec_from_file_location(__name__, _REAL_TEST_PATH)
if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
	raise ImportError(f"Unable to load tool policy tests from {_REAL_TEST_PATH}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.modules[__name__] = module
