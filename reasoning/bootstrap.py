"""Ensure PathPocket root + this folder are on sys.path (self-contained)."""

from __future__ import annotations

import sys
from pathlib import Path

REASONING_ROOT = Path(__file__).resolve().parent
PATHPOCKET_ROOT = REASONING_ROOT.parent

for p in (REASONING_ROOT, PATHPOCKET_ROOT):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
