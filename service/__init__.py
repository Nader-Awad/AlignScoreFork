"""AlignScore service package bootstrap.

Sets PYTORCH_ENABLE_MPS_FALLBACK=1 by default on macOS when the service is
configured to use MPS (explicitly or via ALIGNSCORE_DEVICE=auto). This avoids
runtime NotImplementedError for missing MPS kernels by falling back to CPU for
unsupported ops.
"""

from __future__ import annotations

import os
import platform

# Respect explicit user configuration; otherwise enable fallback on macOS when
# the configured device is 'mps' or 'auto'. This must run before torch import.
if os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") is None:
    target = os.getenv("ALIGNSCORE_DEVICE", "auto").lower()
    if platform.system() == "Darwin" and target in {"auto", "mps"}:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
