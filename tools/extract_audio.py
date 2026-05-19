"""Compatibility wrapper for the THEval audio-extraction CLI."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from theval.cli.extract_audio import main


if __name__ == "__main__":
    raise SystemExit(main())
