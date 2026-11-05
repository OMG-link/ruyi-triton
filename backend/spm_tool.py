"""``python -m triton.backends.triton_shared.spm_tool status | release-stale``.

Shows who holds the board's scratchpad and, on the K3, clears the records of
blocks whose owner thread no longer exists. See spm.py.
"""

import sys

from triton.backends.triton_shared.spm import main

if __name__ == "__main__":
    sys.exit(main())
