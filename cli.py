"""A shim, so that `python3 cli.py analyze ...` from a checkout still works.

The command line itself lives in `src/photoai/cli.py`. This file exists because
the README, the generated delete script and several years of muscle memory all
say `python3 cli.py`, and a working tree is not always an installed package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from photoai.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
