from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator


def ensure_repo_paths() -> None:
    """
    Make imports stable regardless of working directory.
    Adds repo root and scripts/ to sys.path.
    """
    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


@contextmanager
def timed(timings_ms: Dict[str, float], stage: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings_ms[stage] = (time.perf_counter() - t0) * 1000.0
