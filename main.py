from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_ULTRALYTICS = PROJECT_ROOT / "ultralytics"
if LOCAL_ULTRALYTICS.exists() and str(LOCAL_ULTRALYTICS) not in sys.path:
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))

from ui.main_window import run_app


if __name__ == "__main__":
    raise SystemExit(run_app())
