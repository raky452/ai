"""PyInstaller entry script (a package's __main__ can't be frozen directly with relative imports)."""

import os
import sys

# A windowed (no-console) exe has no stdout/stderr; give print() somewhere harmless to go.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115

from tf_assistant.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
