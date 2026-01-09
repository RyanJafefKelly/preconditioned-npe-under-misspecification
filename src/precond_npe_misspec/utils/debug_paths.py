from __future__ import annotations

from datetime import datetime
from pathlib import Path


def ensure_debug_outdir(experiment: str, *, seed: int) -> str:
    """Create and return a results path for ad-hoc debugger runs."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    project_root = Path(__file__).resolve().parents[3]
    outdir = project_root / "results" / "debug" / experiment / f"seed-{seed}" / timestamp
    outdir.mkdir(parents=True, exist_ok=True)
    return str(outdir)

