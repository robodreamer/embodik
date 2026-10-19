"""Shared pytest configuration for source-tree tests."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TEST_DIR = Path(__file__).resolve().parent
for _path in (_REPO_ROOT, _TEST_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
