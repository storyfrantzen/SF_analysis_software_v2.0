#!/usr/bin/env python3
"""Compatibility entry point for the top-level :mod:`visualizer` package.

New code should use ``python -m visualizer`` or import from ``visualizer.app``.
This wrapper keeps existing scripts and documented farm commands working.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from visualizer.app import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    raise SystemExit(main())
